"""Fixed choices for the explicitly exploratory extension."""
from dataclasses import dataclass, field

import numpy as np

from src.models.config import MODEL_DIR, REPORT_DIR, ModelConfig

FEATURE_SETS = {
    "price": ("dec_dd", "dec_mom"),
    "vix_level": ("dec_dd", "dec_mom", "vix_log"),
    "term_structure": ("dec_dd", "dec_mom", "vix_log", "ts_ratio"),
}
MODEL_SETS = {f"logit_{name}": name for name in FEATURE_SETS}
MODEL_SETS["logit_term_structure_matched"] = "term_structure"
MODEL_IDS = ("class_prior", *MODEL_SETS)
CLASS_LABELS = (-1, 0, 1)
PROBABILITY_COLUMNS = ("p_continuation", "p_unresolved", "p_recovery")
HORIZON = 10
OUTPUT_DIR = MODEL_DIR / "extensions"
EXT_REPORT_DIR = REPORT_DIR / "extensions"
EXPLORATORY_NOTICE = (
    "Exploratory extension designed after the original one-day evaluation was seen. "
    "Settings use design data only, but no untouched-holdout claim is made. "
    "No new evaluation-performance or trading analysis is performed here."
)


@dataclass(frozen=True)
class ExtensionConfig:
    periods: ModelConfig = field(default_factory=ModelConfig)
    ridge_alphas: tuple[float, ...] = (0.1, 1., 10., 100., 1000., 10000., 100000.)
    classifier_cs: tuple[float, ...] = (0.01, 0.1, 1., 10.)
    min_episodes: int = 12

    def __post_init__(self):
        for values in (self.ridge_alphas, self.classifier_cs):
            if not values or len(set(values)) != len(values) or any(not np.isfinite(v) or v <= 0 for v in values):
                raise ValueError("penalty grids must contain unique positive finite values")
        if isinstance(self.min_episodes, bool) or not isinstance(self.min_episodes, int) or self.min_episodes < 1:
            raise ValueError("min_episodes must be a positive integer")
