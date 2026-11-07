"""Member 4's public interface: dated decimal-return forecasts, without outcomes."""
from __future__ import annotations

from pathlib import Path
from hashlib import sha256
import json

import numpy as np
import pandas as pd

from src.models.config import MODEL_DIR, MODEL_IDS, MODEL_SETS, PRIMARY_MODEL
from src.models.walk_forward import check_dates

DATE_COLUMNS = ("decision_date", "execution_date", "label_end_date", "train_start_date",
                "train_end_date", "train_last_label_end_date")
PREDICTION_COLUMNS = (*DATE_COLUMNS, "model_id", "feature_set", "horizon", "alpha",
                      "n_train", "decline", "prediction")


def validate_predictions(frame: pd.DataFrame) -> None:
    """Reject partial, mixed-model or outcome-contaminated signal exports."""
    if frame.empty or set(frame.columns) != set(PREDICTION_COLUMNS) or frame.columns.has_duplicates:
        raise ValueError("signal file must contain exactly the documented prediction columns")
    for column in DATE_COLUMNS:
        if not pd.api.types.is_datetime64_any_dtype(frame[column]) or frame[column].isna().any():
            raise ValueError(f"invalid dates in {column}")
        dates = pd.DatetimeIndex(frame[column])
        if dates.tz is not None or not dates.equals(dates.normalize()):
            raise ValueError(f"{column} must contain timezone-naive session dates")
    if frame.duplicated(["decision_date", "model_id"]).any():
        raise ValueError("duplicate model/decision_date")
    if set(frame.model_id) != set(MODEL_IDS):
        raise ValueError("signal file must contain all six documented model IDs")
    for column in ("prediction", "n_train", "horizon", "decline"):
        if not pd.api.types.is_numeric_dtype(frame[column]) or not np.isfinite(frame[column]).all():
            raise ValueError(f"{column} must be finite and numeric")
    if (frame.horizon != 1).any() or not frame.decline.isin([0, 1]).all():
        raise ValueError("expected one-day forecasts and binary decline flags")
    if (frame.n_train < 2).any() or (frame.n_train % 1 != 0).any() or frame.n_train.nunique() != 1:
        raise ValueError("all rows must use the same integer training window")
    if not ((frame.train_start_date <= frame.train_end_date) &
            (frame.train_end_date < frame.train_last_label_end_date) &
            (frame.train_last_label_end_date <= frame.decision_date) &
            (frame.decision_date < frame.execution_date) &
            (frame.execution_date < frame.label_end_date)).all():
        raise ValueError("invalid training, decision or execution timing")
    audit_columns = [*DATE_COLUMNS[1:], "n_train", "horizon", "decline"]
    if (frame.groupby("decision_date")[audit_columns].nunique() != 1).any().any():
        raise ValueError("models disagree on their calendar, training window or decline flag")
    reference = None
    for model_id, part in frame.groupby("model_id"):
        dates = pd.DatetimeIndex(part.decision_date)
        check_dates(dates, model_id)
        if reference is not None and not dates.equals(reference):
            raise ValueError("models have different forecast dates")
        reference = dates
        feature_set = "none" if model_id == "historical_mean" else MODEL_SETS[model_id]
        if not (part.feature_set == feature_set).all():
            raise ValueError(f"wrong feature set for {model_id}")
        if model_id == "historical_mean":
            if not part.alpha.isna().all():
                raise ValueError("historical mean must have no alpha")
        elif not np.isfinite(part.alpha).all() or (part.alpha <= 0).any() or part.alpha.nunique() != 1:
            raise ValueError("each ridge model must use one frozen, finite, positive alpha")
    alphas = frame.groupby("model_id").alpha.first()
    if alphas["ridge_vix_level"] != alphas["ridge_vix_matched"]:
        raise ValueError("matched term-structure models must share alpha")


def wide_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    validate_predictions(predictions)
    calendar = predictions.drop_duplicates("decision_date").set_index("decision_date")
    wide = predictions.pivot(index="decision_date", columns="model_id", values="prediction")
    wide.columns.name = None
    return calendar[["execution_date", "label_end_date", "decline"]].join(wide[list(MODEL_IDS)])


def load_predictions(path: str | Path | None = None) -> pd.DataFrame:
    """Load the complete evaluation signal file and validate its schema and timing."""
    path = Path(path) if path is not None else MODEL_DIR / "predictions.csv"
    manifest_path = path.parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        expected = manifest["signal_sha256"].get(path.name)
        if expected is None or sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("signal checksum does not match manifest; rerun or copy the complete output bundle")
    frame = pd.read_csv(path, parse_dates=list(DATE_COLUMNS), float_precision="round_trip")
    validate_predictions(frame)
    return frame


def load_signal(model_id: str = PRIMARY_MODEL, path: str | Path | None = None) -> pd.DataFrame:
    """Return one model indexed by decision_date, including execution/return dates.

    `prediction` is a decimal SPY-minus-BIL return estimate, NOT a probability or
    portfolio weight. Do not shift this frame before supplying decision-date
    weights to src.metrics.apply_weights: that helper already applies the lag.
    """
    frame = load_predictions(path)
    if model_id not in MODEL_IDS:
        raise ValueError(f"unknown model {model_id!r}; choose from {MODEL_IDS}")
    return frame[frame.model_id == model_id].set_index("decision_date").copy()
