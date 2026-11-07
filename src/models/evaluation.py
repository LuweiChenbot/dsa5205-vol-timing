"""Validation-only selection and matched forecast diagnostics (not trading P&L)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.config import FEATURE_SETS
from src.models.walk_forward import check_dates


def score_predictions(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    actual = outcomes["actual_excess_return"]
    if actual.index.has_duplicates or not np.isfinite(actual).all():
        raise ValueError("outcomes must be unique and finite")
    scored = predictions.merge(actual.rename_axis("decision_date").reset_index(),
                               on="decision_date", how="left", validate="many_to_one")
    if scored.actual_excess_return.isna().any():
        raise ValueError("missing outcome for a prediction")
    scored["squared_error"] = (scored.prediction - scored.actual_excess_return) ** 2
    scored["absolute_error"] = (scored.prediction - scored.actual_excess_return).abs()
    return scored


def select_alphas(candidates: pd.DataFrame, outcomes: pd.DataFrame,
                  design_end: str) -> tuple[dict[str, float], pd.DataFrame]:
    """Freeze one alpha per set; never consult outcomes beyond design_end."""
    cutoff = pd.Timestamp(design_end)
    eligible = candidates[(candidates.decision_date <= cutoff) &
                          (candidates.label_end_date <= cutoff)]
    if eligible.empty:
        raise ValueError("no eligible validation predictions")
    scored = score_predictions(eligible, outcomes.loc[outcomes.label_end_date <= cutoff])
    scored = scored[scored.model_id != "historical_mean"]
    rows = []
    reference = None
    for (feature_set, alpha), part in scored.groupby(["feature_set", "alpha"], sort=True):
        dates = pd.DatetimeIndex(part.decision_date).sort_values()
        if dates.has_duplicates or (reference is not None and not dates.equals(reference)):
            raise ValueError("validation candidates must share exactly the same unique dates")
        reference = dates
        rows.append({"feature_set": feature_set, "alpha": alpha, "n": len(part),
                     "mse": part.squared_error.mean(), "mae": part.absolute_error.mean(),
                     "first_decision": dates[0], "last_decision": dates[-1],
                     "last_label_end": part.label_end_date.max()})
    table = pd.DataFrame(rows)
    if table.empty or set(table.feature_set) != set(FEATURE_SETS):
        raise ValueError("validation requires all four feature sets")
    grids = [tuple(group.alpha.sort_values()) for _, group in table.groupby("feature_set")]
    if len(set(grids)) != 1:
        raise ValueError("validation feature sets must use the same alpha grid")
    selected = {}
    for name in FEATURE_SETS:
        ranked = table[table.feature_set == name].sort_values(["mse", "alpha"], ascending=[True, False])
        selected[name] = float(ranked.iloc[0].alpha)
    table["selected"] = [alpha == selected[name] for name, alpha in zip(table.feature_set, table.alpha)]
    return selected, table


def selected_predictions(candidates: pd.DataFrame, alphas: dict[str, float]) -> pd.DataFrame:
    """Extract the frozen choices and the matched-penalty term-structure model."""
    pieces = [candidates[candidates.model_id == "historical_mean"]]
    for name, alpha in alphas.items():
        pieces.append(candidates[(candidates.model_id == f"ridge_{name}") & (candidates.alpha == alpha)])
    matched = candidates[(candidates.model_id == "ridge_vix") &
                         (candidates.alpha == alphas["vix_level"])].copy()
    matched["model_id"] = "ridge_vix_matched"
    pieces.append(matched)
    return pd.concat(pieces, ignore_index=True).sort_values(["decision_date", "model_id"]).reset_index(drop=True)


def forecast_metrics(scored: pd.DataFrame) -> pd.DataFrame:
    """Pooled and decision-year metrics on identical dates for every model."""
    mean = scored.loc[scored.model_id == "historical_mean", ["decision_date", "squared_error"]]
    joined = scored.merge(mean.rename(columns={"squared_error": "mean_squared_error"}),
                          on="decision_date", validate="many_to_one")
    rows = []
    for model_id, model in joined.groupby("model_id", sort=True):
        periods = [("all", model), *[(str(year), part) for year, part in
                                   model.groupby(model.decision_date.dt.year)]]
        for year, part in periods:
            for segment, mask in (("all", np.ones(len(part), dtype=bool)),
                                  ("decline", part.decline == 1),
                                  ("non_decline", part.decline == 0)):
                s = part.loc[mask]
                n = len(s)
                denom = s.mean_squared_error.sum()
                mse = s.squared_error.mean()
                rows.append({"model_id": model_id, "decision_year": year, "segment": segment,
                             "n": n, "mse": mse, "rmse": np.sqrt(mse),
                             "mae": s.absolute_error.mean(),
                             "oos_r2": 1 - s.squared_error.sum() / denom if denom > 0 else np.nan,
                             "direction_accuracy": float((np.sign(s.prediction) ==
                                                          np.sign(s.actual_excess_return)).mean()) if n else np.nan,
                             "mean_prediction": s.prediction.mean(),
                             "mean_actual": s.actual_excess_return.mean()})
    return pd.DataFrame(rows)


COMPARISONS = (
    ("vix_level_information", "ridge_base", "ridge_vix_level"),
    ("term_structure_tuned", "ridge_vix_level", "ridge_vix"),
    ("term_structure_same_alpha", "ridge_vix_level", "ridge_vix_matched"),
    ("interactions_tuned", "ridge_vix", "ridge_full"),
)


def paired_comparisons(scored: pd.DataFrame) -> pd.DataFrame:
    """A negative MSE/MAE difference favours the added features; no causal claim."""
    rows = []
    for name, smaller, larger in COMPARISONS:
        a, b = [scored[scored.model_id == m].set_index("decision_date").sort_index()
                for m in (smaller, larger)]
        if a.index.has_duplicates or not a.index.equals(b.index) or a.empty:
            raise ValueError("comparisons require identical, nonempty, unique forecast dates")
        for segment, mask in (("all", np.ones(len(a), dtype=bool)),
                              ("decline", a.decline == 1), ("non_decline", a.decline == 0)):
            aa, bb = a.loc[mask], b.loc[mask]
            smaller_mse = aa.squared_error.mean()
            delta = bb.squared_error.mean() - smaller_mse
            rows.append({"comparison": name, "smaller_model": smaller, "larger_model": larger,
                         "segment": segment, "n": len(aa), "smaller_mse": smaller_mse,
                         "larger_mse": bb.squared_error.mean(), "mse_difference": delta,
                         "mse_change_pct": 100 * delta / smaller_mse if smaller_mse > 0 else np.nan,
                         "mae_difference": bb.absolute_error.mean() - aa.absolute_error.mean()})
    return pd.DataFrame(rows)


def event_diagnostics(predictions: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Descriptive ranking of 10-day barrier labels by the one-day forecast.

    AUC uses resolved (+1/-1) events only. Unresolved events remain in the joined
    table and their count is reported; predictions are NOT probabilities.
    """
    needed = ["ev_start", "fwd_ev_label"]
    check_dates(events.index, "events")
    if events.columns.has_duplicates or not set(needed).issubset(events.columns):
        raise ValueError("invalid event table")
    if not pd.api.types.is_bool_dtype(events.ev_start) or events.ev_start.isna().any():
        raise ValueError("ev_start must contain nonmissing booleans")
    if not pd.Index(predictions.decision_date.unique()).isin(events.index).all():
        raise ValueError("event table does not cover all prediction dates")
    starts = events.loc[events.ev_start, ["fwd_ev_label"]]
    joined = predictions.merge(starts.rename_axis("decision_date").reset_index(),
                               on="decision_date", how="inner", validate="many_to_one")
    if not joined.fwd_ev_label.isin([-1, 0, 1]).all():
        raise ValueError("event starts must have completed barrier labels")
    rows = []
    for model_id in predictions.model_id.unique():
        group = joined[joined.model_id == model_id]
        resolved = group[group.fwd_ev_label != 0]
        auc = (roc_auc_score(resolved.fwd_ev_label == 1, resolved.prediction)
               if resolved.fwd_ev_label.nunique() == 2 else np.nan)
        rows.append({"model_id": model_id, "n_events": len(group),
                     "n_recovery": int((group.fwd_ev_label == 1).sum()),
                     "n_continuation": int((group.fwd_ev_label == -1).sum()),
                     "n_unresolved": int((group.fwd_ev_label == 0).sum()),
                     "resolved_auc": auc,
                     "mean_score_recovery": group.loc[group.fwd_ev_label == 1, "prediction"].mean(),
                     "mean_score_continuation": group.loc[group.fwd_ev_label == -1, "prediction"].mean(),
                     "mean_score_unresolved": group.loc[group.fwd_ev_label == 0, "prediction"].mean()})
    return pd.DataFrame(rows), joined
