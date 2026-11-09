"""Typed-by-schema handoff for ten-day returns and decline-only probabilities."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.handoff import DATE_COLUMNS, load_predictions
from src.models.extensions.config import (HORIZON, MODEL_IDS, MODEL_SETS, OUTPUT_DIR,
                                          PROBABILITY_COLUMNS)
from src.models.walk_forward import check_dates

RECOVERY_DATES = (*DATE_COLUMNS, "episode_start_date")
COUNT_COLUMNS = ("n_train", "n_episodes", "n_continuation", "n_unresolved", "n_recovery")
RECOVERY_COLUMNS = (*RECOVERY_DATES, "model_id", "feature_set", "horizon", "C",
                    *COUNT_COLUMNS, "train_weight_sum", "fit_status", *PROBABILITY_COLUMNS)


def validate_recovery(frame: pd.DataFrame) -> None:
    if frame.empty or set(frame.columns) != set(RECOVERY_COLUMNS) or frame.columns.has_duplicates:
        raise ValueError("expected the complete recovery signal schema, without outcomes")
    if set(frame.model_id) != set(MODEL_IDS) or frame.duplicated(["decision_date", "model_id"]).any():
        raise ValueError("expected five recovery models with unique model/date pairs")
    for name in (*COUNT_COLUMNS, "train_weight_sum", "horizon", *PROBABILITY_COLUMNS):
        if not pd.api.types.is_numeric_dtype(frame[name]) or not np.isfinite(frame[name]).all():
            raise ValueError(f"{name} must be finite and numeric")
    if (frame[list(COUNT_COLUMNS)] < 0).any().any() or (frame[list(COUNT_COLUMNS)] % 1 != 0).any().any():
        raise ValueError("training counts must be nonnegative integers")
    if (frame.horizon != HORIZON).any() or (frame.n_episodes > frame.n_train).any():
        raise ValueError("invalid horizon or episode count")
    if (frame.n_continuation + frame.n_unresolved + frame.n_recovery != frame.n_train).any():
        raise ValueError("training class counts do not sum to n_train")
    if not np.allclose(frame.train_weight_sum, frame.n_episodes, rtol=0, atol=1e-10):
        raise ValueError("each training episode must carry one total unit of weight")
    p = frame[list(PROBABILITY_COLUMNS)].to_numpy()
    if (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1, rtol=0, atol=1e-10):
        raise ValueError("probabilities must be bounded and sum to one")
    for name in RECOVERY_DATES:
        if not pd.api.types.is_datetime64_any_dtype(frame[name]):
            raise ValueError(f"{name} must contain dates")
        dates = pd.DatetimeIndex(frame[name].dropna())
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError("expected timezone-naive session dates")
    known_dates = ["decision_date", "execution_date", "label_end_date", "episode_start_date"]
    if frame[known_dates].isna().any().any() or not (
        (frame.episode_start_date <= frame.decision_date) &
        (frame.decision_date < frame.execution_date) & (frame.execution_date < frame.label_end_date)
    ).all():
        raise ValueError("invalid episode, decision or execution timing")
    trained = frame.n_train > 0
    training_dates = ["train_start_date", "train_end_date", "train_last_label_end_date"]
    if frame.loc[trained, training_dates].isna().any().any() or frame.loc[~trained, training_dates].notna().any().any():
        raise ValueError("training dates must be present exactly when training rows exist")
    t = frame.loc[trained]
    if not ((t.train_start_date <= t.train_end_date) & (t.train_end_date < t.train_last_label_end_date) &
            (t.train_last_label_end_date <= t.decision_date)).all():
        raise ValueError("training includes unavailable outcomes")
    audit = [*RECOVERY_DATES[1:], *COUNT_COLUMNS, "horizon", "train_weight_sum"]
    if (frame.groupby("decision_date")[audit].nunique(dropna=False) != 1).any().any():
        raise ValueError("recovery models disagree on timing or training data")
    reference = None
    for model_id, part in frame.groupby("model_id"):
        dates = pd.DatetimeIndex(part.decision_date)
        check_dates(dates, model_id)
        if reference is not None and not dates.equals(reference):
            raise ValueError("recovery models have different forecast dates")
        reference = dates
        expected_set = "none" if model_id == "class_prior" else MODEL_SETS[model_id]
        if not part.feature_set.eq(expected_set).all():
            raise ValueError("wrong classifier feature set")
        if model_id == "class_prior":
            if not part.C.isna().all() or not part.fit_status.eq("prior").all():
                raise ValueError("invalid prior metadata")
        elif (not np.isfinite(part.C).all() or (part.C <= 0).any() or part.C.nunique() != 1 or
              not part.fit_status.isin(["fit", "prior_fallback"]).all()):
            raise ValueError("invalid frozen C or fit status")
    cs = frame.groupby("model_id").C.first()
    if cs["logit_vix_level"] != cs["logit_term_structure_matched"]:
        raise ValueError("matched classifier must share vix_level C")
    prior = frame[frame.model_id == "class_prior"].set_index("decision_date")
    fallback = frame[frame.fit_status == "prior_fallback"]
    if not np.allclose(fallback[list(PROBABILITY_COLUMNS)], prior.loc[fallback.decision_date, list(PROBABILITY_COLUMNS)]):
        raise ValueError("fallback probabilities must equal the contemporaneous prior")


def recovery_wide(frame: pd.DataFrame) -> pd.DataFrame:
    validate_recovery(frame)
    timing = frame.drop_duplicates("decision_date").set_index("decision_date")[
        ["execution_date", "label_end_date", "episode_start_date"]]
    wide = frame.pivot(index="decision_date", columns="model_id", values=list(PROBABILITY_COLUMNS))
    wide.columns = [f"{model_id}__{probability}" for probability, model_id in wide.columns]
    return timing.join(wide)


def load_recovery_predictions(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else OUTPUT_DIR / "recovery" / "predictions.csv"
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        expected = json.loads(manifest_path.read_text())["signal_sha256"].get(path.name)
        if expected is None or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("recovery signal checksum does not match manifest")
    frame = pd.read_csv(path, parse_dates=list(RECOVERY_DATES), float_precision="round_trip")
    validate_recovery(frame)
    return frame


def load_recovery_signal(model_id: str = "logit_term_structure", path: str | Path | None = None) -> pd.DataFrame:
    frame = load_recovery_predictions(path)
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown classifier {model_id!r}")
    return frame[frame.model_id == model_id].set_index("decision_date").copy()


def load_ten_day_signal(model_id: str = "ridge_full", path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else OUTPUT_DIR / "ridge_h10" / "predictions.csv"
    frame = load_predictions(path, expected_horizon=HORIZON)
    if model_id not in frame.model_id.unique():
        raise ValueError(f"unknown ridge model {model_id!r}")
    return frame[frame.model_id == model_id].set_index("decision_date").copy()
