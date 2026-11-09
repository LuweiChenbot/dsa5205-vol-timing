"""Daily rolling estimation with explicit, calendar-based label maturity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features.feature_build import PastScaler
from src.features.targets import forward_return
from src.models.config import FEATURE_SETS


def check_dates(index: pd.Index, name: str) -> None:
    if not isinstance(index, pd.DatetimeIndex) or index.empty:
        raise ValueError(f"{name} must have a nonempty DatetimeIndex")
    if index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{name} dates must be nonmissing, unique and increasing")
    if index.tz is not None or not index.equals(index.normalize()):
        raise ValueError(f"{name} must contain timezone-naive session dates")


@dataclass
class ModelData:
    features: pd.DataFrame
    target: pd.Series
    calendar: pd.DataFrame
    outcomes: pd.DataFrame
    horizon: int = 1


def prepare_data(panel: pd.DataFrame, features: pd.DataFrame, *, horizon: int = 1,
                 allow_incomplete_tail: bool = False) -> ModelData:
    """Validate the snapshot and map each decision to execution and maturity.

    Keep the complete panel here: truncating it at the last decision would lose
    that decision's execution and outcome. No rows are imputed or silently dropped.
    An explicitly allowed incomplete tail has NaT/NaN outcomes and is logged by
    walk_forward; it cannot enter training or silently become a shorter target.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    check_dates(panel.index, "panel")
    check_dates(features.index, "features")
    if panel.columns.has_duplicates or features.columns.has_duplicates:
        raise ValueError("duplicate column names")
    if set(features.columns) != set(FEATURE_SETS["full"]):
        raise ValueError("features must contain exactly the frozen 14 feature columns")
    features = features.loc[:, list(FEATURE_SETS["full"])].astype(float).copy()
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("features contain missing or nonfinite values")
    for column in ("dec_cond", "ts_backw"):
        if not features[column].isin([0.0, 1.0]).all():
            raise ValueError(f"{column} must be binary")
    required = {"spy_tr_cc", "bil_tr_cc", "fwd_spy_xs"}
    if not required.issubset(panel.columns):
        raise ValueError(f"panel is missing {sorted(required - set(panel.columns))}")
    positions = panel.index.get_indexer(features.index)
    if (positions < 0).any() or (np.diff(positions) != 1).any():
        raise ValueError("features must cover consecutive panel sessions with no gaps")
    if positions[-1] + 1 >= len(panel):
        raise ValueError("panel must include execution after the last decision")
    completed = positions + horizon + 1 < len(panel)
    if not completed.all() and not allow_incomplete_tail:
        raise ValueError("panel must extend two sessions past the last decision" if horizon == 1
                         else "panel must extend horizon+1 sessions past the last decision")
    ends = pd.Series(pd.NaT, index=features.index, dtype="datetime64[ns]")
    ends.loc[completed] = panel.index[positions[completed] + horizon + 1]
    calendar = pd.DataFrame({
        "execution_date": panel.index[positions + 1],
        "label_end_date": ends.to_numpy(),
    }, index=features.index.rename("decision_date"))
    daily = panel.iloc[positions[0] + 2:][["spy_tr_cc", "bil_tr_cc"]].to_numpy(dtype=float)
    if not np.isfinite(daily).all() or (daily <= -1).any():
        raise ValueError("realised asset returns are missing, nonfinite or <= -100%")
    if horizon == 1:
        realised = panel[["spy_tr_cc", "bil_tr_cc"]].shift(-2).loc[features.index].to_numpy(dtype=float)
    else:
        realised = np.column_stack([forward_return(panel[f"{asset}_tr_cc"], horizon).loc[features.index]
                                    for asset in ("spy", "bil")])
    expected = realised[:, 0] - realised[:, 1]
    target = (panel.loc[features.index, "fwd_spy_xs"].astype(float).rename("actual_excess_return")
              if horizon == 1 else pd.Series(expected, index=features.index, name="actual_excess_return"))
    if not np.isfinite(target.loc[completed]).all() or not np.allclose(
        target.loc[completed], expected[completed], rtol=1e-8, atol=1e-11
    ):
        raise ValueError("fwd_spy_xs must equal SPY minus BIL return at label_end_date")
    outcomes = calendar.assign(actual_excess_return=target.to_numpy(),
                               spy_return=realised[:, 0], bil_return=realised[:, 1])
    return ModelData(features, target, calendar, outcomes, horizon)


@dataclass
class ForecastRun:
    predictions: pd.DataFrame
    coverage: pd.DataFrame
    coefficients: pd.DataFrame


def walk_forward(
    data: ModelData,
    dates: pd.DatetimeIndex,
    models: dict[str, tuple[str, tuple[float, ...]]],
    train_window: int,
    *,
    outcome_cutoff: pd.Timestamp | None = None,
    keep_coefficients: bool = True,
) -> ForecastRun:
    """Forecast each requested date with exactly train_window completed labels.

    models maps model ID to (feature set, alphas). A rolling historical mean is
    always included. outcome_cutoff purges validation targets crossing the design
    boundary; it is NOT applied to the evaluation sample's trailing outcomes.
    """
    check_dates(dates, "forecast")
    if not dates.isin(data.features.index).all():
        raise ValueError("forecast dates are missing from features")
    if isinstance(train_window, bool) or not isinstance(train_window, int) or train_window < 2:
        raise ValueError("train_window must be an integer of at least 2")
    if not models:
        raise ValueError("at least one ridge model is required")
    for model_id, (feature_set, alphas) in models.items():
        if model_id == "historical_mean" or feature_set not in FEATURE_SETS:
            raise ValueError("invalid model ID or feature set")
        if not alphas or len(set(alphas)) != len(alphas) or any(not np.isfinite(a) or a <= 0 for a in alphas):
            raise ValueError("each model requires unique, finite, positive alphas")

    label_dates = data.calendar.label_end_date.dropna().to_numpy()
    column_positions = {name: data.features.columns.get_indexer(columns)
                        for name, columns in FEATURE_SETS.items()}
    predictions, coverage, coefficients = [], [], []
    for date in dates:
        # Side=right permits labels realised at today's close: the decision is later.
        stop = int(np.searchsorted(label_dates, date.to_datetime64(), side="right"))
        timing = data.calendar.loc[date]
        status = "predicted"
        if pd.isna(timing.label_end_date):
            status = "incomplete_outcome_calendar"
        elif outcome_cutoff is not None and timing.label_end_date > outcome_cutoff:
            status = "outcome_after_design_end"
        elif stop < train_window:
            status = "insufficient_history"
        coverage.append({"decision_date": date, "label_end_date": timing.label_end_date,
                         "available_train_rows": stop, "status": status})
        if status != "predicted":
            continue

        start = stop - train_window
        X = data.features.iloc[start:stop]
        y = data.target.iloc[start:stop].to_numpy(dtype=float)
        if not np.isfinite(y).all():
            raise ValueError(f"nonfinite training label at decision {date.date()}")
        # Scaling is columnwise, so one full-set scaler equals a separate fit for
        # each subset. It is fitted anew on ONLY the matured training window.
        scaler = PastScaler().fit(X)
        Xz = scaler.transform(X).to_numpy()
        xz = scaler.transform(data.features.loc[[date]]).to_numpy()
        audit = {"decision_date": date, "execution_date": timing.execution_date,
                 "label_end_date": timing.label_end_date, "horizon": data.horizon,
                 "n_train": train_window, "train_start_date": X.index[0],
                 "train_end_date": X.index[-1],
                 "train_last_label_end_date": pd.Timestamp(label_dates[stop - 1]),
                 "decline": int(data.features.at[date, "dec_cond"])}
        predictions.append({**audit, "model_id": "historical_mean", "feature_set": "none",
                            "alpha": np.nan, "prediction": float(y.mean())})
        for model_id, (feature_set, alphas) in models.items():
            cols = column_positions[feature_set]
            for alpha in alphas:
                model = Ridge(alpha=float(alpha), fit_intercept=True, solver="cholesky")
                model.fit(Xz[:, cols], y)
                prediction = float(model.predict(xz[:, cols])[0])
                if not np.isfinite(prediction):
                    raise ValueError(f"nonfinite prediction for {model_id} at {date}")
                predictions.append({**audit, "model_id": model_id, "feature_set": feature_set,
                                    "alpha": float(alpha), "prediction": prediction})
                if keep_coefficients:
                    coefficients.append({"decision_date": date, "model_id": model_id,
                                         "alpha": float(alpha), "intercept": float(model.intercept_),
                                         **dict(zip(FEATURE_SETS[feature_set], model.coef_))})
    return ForecastRun(pd.DataFrame(predictions), pd.DataFrame(coverage), pd.DataFrame(coefficients))
