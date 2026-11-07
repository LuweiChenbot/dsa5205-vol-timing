"""Predeclared model settings; no command-line tuning on evaluation outcomes."""
from __future__ import annotations

from dataclasses import dataclass

from src import config
from src.features.feature_build import FEATURE_SETS as UPSTREAM_SETS

FEATURE_SETS = {
    "base": tuple(UPSTREAM_SETS["base"]),
    "vix_level": tuple(c for c in UPSTREAM_SETS["vix"] if c not in {"ts_ratio", "ts_backw"}),
    "vix": tuple(UPSTREAM_SETS["vix"]),
    "full": tuple(UPSTREAM_SETS["full"]),
}
MODEL_SETS = {f"ridge_{name}": name for name in FEATURE_SETS}
MODEL_SETS["ridge_vix_matched"] = "vix"
MODEL_IDS = ("historical_mean", *MODEL_SETS)
PRIMARY_MODEL = "ridge_full"
MODEL_DIR = config.PROCESSED_DIR / "models"
REPORT_DIR = config.PROCESSED_DIR.parent.parent / "reports" / "models"
RESEARCH_LIMITATION = (
    "Earlier feature research examined 2017–2025 outcomes, as disclosed in "
    "docs/report_features_sections.md. The present forecasts obey information "
    "timing, but 2019–2025 is not a pristine, untouched research holdout."
)


@dataclass(frozen=True)
class ModelConfig:
    train_window: int = 756
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1000.0)
    validation_start: str = "2017-01-01"
    design_end: str = config.DESIGN_END
    test_start: str = config.TEST_START
    test_end: str = config.SAMPLE_END

    def __post_init__(self) -> None:
        import numpy as np
        import pandas as pd

        if isinstance(self.train_window, bool) or not isinstance(self.train_window, int) or self.train_window < 2:
            raise ValueError("train_window must be an integer of at least 2")
        if not self.alphas or len(set(self.alphas)) != len(self.alphas) or any(
            not np.isfinite(a) or a <= 0 for a in self.alphas
        ):
            raise ValueError("alphas must be unique, finite and positive")
        dates = [pd.Timestamp(d) for d in (
            self.validation_start, self.design_end, self.test_start, self.test_end
        )]
        if any(pd.isna(d) or d.tzinfo is not None or d != d.normalize() for d in dates):
            raise ValueError("split dates must be timezone-naive dates")
        if not (dates[0] <= dates[1] < dates[2] <= dates[3]):
            raise ValueError("require validation_start <= design_end < test_start <= test_end")
