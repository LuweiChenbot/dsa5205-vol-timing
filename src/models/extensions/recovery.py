"""Direct three-class recovery probabilities on decline-trigger dates."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from src.features.events import H, K, barrier_labels, event_starts
from src.features.feature_build import PastScaler
from src.models.walk_forward import ForecastRun, ModelData, check_dates
from src.models.extensions.config import CLASS_LABELS, FEATURE_SETS, PROBABILITY_COLUMNS


@dataclass
class RecoveryData:
    features: pd.DataFrame
    calendar: pd.DataFrame
    outcomes: pd.DataFrame


def prepare_recovery(panel: pd.DataFrame, data: ModelData, events: pd.DataFrame) -> RecoveryData:
    """Reuse the upstream barrier code, but label every trigger without censoring.

    Episode IDs use the original causal event-start rule on the full calendar.
    They do not use future prices or wait to discover how long a decline lasts.
    All labels wait for their full ten-session maturity, even early barrier hits.
    """
    if data.horizon != H:
        raise ValueError("recovery data requires the ten-day calendar")
    check_dates(events.index, "events")
    if not events.index.equals(panel.index) or "dec_cond" not in events or events.columns.has_duplicates:
        raise ValueError("events must cover the panel calendar and include dec_cond")
    cond = events.dec_cond
    if not pd.api.types.is_bool_dtype(cond) or cond.isna().any():
        raise ValueError("event decline flags must be nonmissing booleans")
    if not np.array_equal(cond.loc[data.features.index].to_numpy(), data.features.dec_cond.astype(bool).to_numpy()):
        raise ValueError("saved features and events disagree on decline triggers")
    if "spy_adj_close" not in panel or not np.isfinite(panel.spy_adj_close).all() or (panel.spy_adj_close <= 0).any():
        raise ValueError("recovery labels require finite positive SPY total-return prices")
    trigger = data.features.dec_cond.astype(bool)
    full_trigger = trigger.reindex(panel.index, fill_value=False)
    width = K * data.features.dec_vol.reindex(panel.index) * np.sqrt(H)
    if (width.loc[trigger.index[trigger]] <= 0).any():
        raise ValueError("trigger-day volatility must be positive")
    labels = barrier_labels(panel, full_trigger, width, H).fwd_ev_label
    starts = event_starts(cond, H)
    episode_dates = pd.Series(panel.index, index=panel.index).where(starts).ffill()
    dates = trigger.index[trigger]
    calendar = data.calendar.loc[dates].copy()
    calendar["episode_start_date"] = episode_dates.loc[dates]
    if calendar.episode_start_date.isna().any():
        raise ValueError("every trigger must belong to a causal episode")
    outcomes = calendar.assign(label=labels.loc[dates])
    complete = outcomes.label_end_date.notna()
    if not outcomes.loc[complete, "label"].isin(CLASS_LABELS).all():
        raise ValueError("completed recovery labels must be -1, 0 or +1")
    if outcomes.loc[~complete, "label"].notna().any():
        raise ValueError("incomplete recovery horizons must not have labels")
    return RecoveryData(data.features.loc[dates].copy(), calendar, outcomes)


def episode_weights(episodes: pd.Series) -> np.ndarray:
    """One total unit of weight per episode, computed on the supplied rows only."""
    if episodes.isna().any():
        raise ValueError("missing episode ID")
    return (1.0 / episodes.groupby(episodes).transform("size")).to_numpy(dtype=float)


def class_prior(labels: np.ndarray, weights: np.ndarray) -> np.ndarray:
    counts = np.array([weights[labels == value].sum() for value in CLASS_LABELS])
    return (counts + 1.0) / (counts.sum() + len(CLASS_LABELS))


def walk_recovery(data: RecoveryData, dates: pd.DatetimeIndex,
                  models: dict[str, tuple[str, tuple[float, ...]]], *,
                  min_episodes: int = 12, outcome_cutoff: pd.Timestamp | None = None) -> ForecastRun:
    check_dates(dates, "recovery forecast")
    if not dates.isin(data.features.index).all():
        raise ValueError("recovery forecasts are only defined on saved decline-trigger dates")
    if isinstance(min_episodes, bool) or not isinstance(min_episodes, int) or min_episodes < 1:
        raise ValueError("min_episodes must be a positive integer")
    if not models:
        raise ValueError("at least one classifier is required")
    for model_id, (name, cs) in models.items():
        if model_id == "class_prior" or name not in FEATURE_SETS or not cs or len(set(cs)) != len(cs):
            raise ValueError("invalid classifier plan")
        if any(not np.isfinite(c) or c <= 0 for c in cs):
            raise ValueError("C must be positive and finite")
    rows, coverage, coefficients = [], [], []
    for date in dates:
        timing = data.calendar.loc[date]
        mature = data.calendar.label_end_date.le(date)
        training = data.outcomes.loc[mature]
        n = len(training)
        groups = training.episode_start_date.nunique()
        status = ("incomplete_outcome_calendar" if pd.isna(timing.label_end_date) else
                  "outcome_after_design_end" if outcome_cutoff is not None and timing.label_end_date > outcome_cutoff
                  else "predicted")
        coverage.append({"decision_date": date, "label_end_date": timing.label_end_date,
                         "available_train_rows": n, "n_episodes": groups, "status": status})
        if status != "predicted":
            continue
        if not training.label.isin(CLASS_LABELS).all():
            raise ValueError("invalid matured recovery labels")
        y = training.label.to_numpy(dtype=int)
        weights = episode_weights(training.episode_start_date)
        prior = class_prior(y, weights)
        fallback = groups < min_episodes or set(y) != set(CLASS_LABELS)
        audit = {"decision_date": date, "execution_date": timing.execution_date,
                 "label_end_date": timing.label_end_date, "episode_start_date": timing.episode_start_date,
                 "horizon": H, "n_train": n, "n_episodes": groups,
                 "train_weight_sum": float(weights.sum()),
                 "train_start_date": training.index[0] if n else pd.NaT,
                 "train_end_date": training.index[-1] if n else pd.NaT,
                 "train_last_label_end_date": training.label_end_date.max() if n else pd.NaT,
                 **{f"n_{name}": int((y == value).sum()) for name, value in
                    zip(("continuation", "unresolved", "recovery"), CLASS_LABELS)}}
        rows.append({**audit, "model_id": "class_prior", "feature_set": "none", "C": np.nan,
                     "fit_status": "prior", **dict(zip(PROBABILITY_COLUMNS, prior))})
        if not fallback:
            X = data.features.loc[training.index, list(FEATURE_SETS["term_structure"])]
            scaler = PastScaler().fit(X)
            Xz = scaler.transform(X)
            xz = scaler.transform(data.features.loc[[date], X.columns])
        for model_id, (name, cs) in models.items():
            for c in cs:
                probabilities = prior
                if not fallback:
                    columns = list(FEATURE_SETS[name])
                    model = LogisticRegression(C=float(c), penalty="l2", solver="lbfgs",
                                               max_iter=2000, tol=1e-8)
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(Xz[columns], y, sample_weight=weights)
                    if tuple(model.classes_) != CLASS_LABELS:
                        raise ValueError("unexpected classifier class ordering")
                    probabilities = model.predict_proba(xz[columns])[0]
                    for label, intercept, coef in zip(model.classes_, model.intercept_, model.coef_):
                        coefficients.append({"decision_date": date, "model_id": model_id, "C": c,
                                             "class_label": int(label), "intercept": float(intercept),
                                             **dict(zip(columns, coef))})
                rows.append({**audit, "model_id": model_id, "feature_set": name, "C": float(c),
                             "fit_status": "prior_fallback" if fallback else "fit",
                             **dict(zip(PROBABILITY_COLUMNS, probabilities))})
    return ForecastRun(pd.DataFrame(rows), pd.DataFrame(coverage), pd.DataFrame(coefficients))


def select_cs(candidates: pd.DataFrame, outcomes: pd.DataFrame,
              design_end: str) -> tuple[dict[str, float], pd.DataFrame]:
    cutoff = pd.Timestamp(design_end)
    eligible = candidates[(candidates.decision_date <= cutoff) & (candidates.label_end_date <= cutoff) &
                          (candidates.model_id != "class_prior")]
    actual = outcomes.loc[outcomes.label_end_date <= cutoff, ["label"]]
    scored = eligible.merge(actual.rename_axis("decision_date").reset_index(), on="decision_date",
                             how="left", validate="many_to_one")
    if scored.empty or not scored.label.isin(CLASS_LABELS).all():
        raise ValueError("no complete classification validation sample")
    reference, rows = None, []
    for (name, c), part in scored.groupby(["feature_set", "C"]):
        dates = pd.DatetimeIndex(part.decision_date).sort_values()
        if dates.has_duplicates or (reference is not None and not dates.equals(reference)):
            raise ValueError("classification candidates require identical unique dates")
        reference = dates
        probabilities = part[list(PROBABILITY_COLUMNS)].to_numpy()
        if (not np.isfinite(probabilities).all() or (probabilities < 0).any() or
                (probabilities > 1).any() or not np.allclose(probabilities.sum(axis=1), 1)):
            raise ValueError("invalid class probabilities")
        # Labels -1,0,1 correspond to columns 0,1,2. Clip only for numerical log(0).
        correct = probabilities[np.arange(len(part)), part.label.to_numpy(dtype=int) + 1]
        weights = episode_weights(part.episode_start_date)
        loss = float(np.average(-np.log(np.clip(correct, np.finfo(float).eps, 1)), weights=weights))
        rows.append({"feature_set": name, "C": c, "n": len(part),
                     "n_episodes": part.episode_start_date.nunique(), "weighted_log_loss": loss,
                     "last_label_end": part.label_end_date.max()})
    table = pd.DataFrame(rows)
    if set(table.feature_set) != set(FEATURE_SETS):
        raise ValueError("all three classifier feature sets must be present")
    if len({tuple(g.C.sort_values()) for _, g in table.groupby("feature_set")}) != 1:
        raise ValueError("all classifier feature sets must use the same C grid")
    selected = {name: float(table[table.feature_set == name].sort_values(
        ["weighted_log_loss", "C"]).iloc[0].C) for name in FEATURE_SETS}
    table["selected"] = [c == selected[name] for name, c in zip(table.feature_set, table.C)]
    return selected, table


def selected_recovery(candidates: pd.DataFrame, cs: dict[str, float]) -> pd.DataFrame:
    pieces = [candidates[candidates.model_id == "class_prior"]]
    for name, c in cs.items():
        pieces.append(candidates[(candidates.model_id == f"logit_{name}") & (candidates.C == c)])
    matched = candidates[(candidates.model_id == "logit_term_structure") & (candidates.C == cs["vix_level"])].copy()
    matched["model_id"] = "logit_term_structure_matched"
    return pd.concat([*pieces, matched], ignore_index=True).sort_values(
        ["decision_date", "model_id"]).reset_index(drop=True)
