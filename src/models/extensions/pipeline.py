"""Offline exploratory model build: python -m src.models.extensions.pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

import pandas as pd

from src import config
from src.data.build import load_panel
from src.features.events import K, load_events
from src.features.feature_build import load_features
from src.models.config import FEATURE_SETS as RIDGE_SETS, MODEL_SETS as RIDGE_MODELS, RESEARCH_LIMITATION
from src.models.evaluation import select_alphas, selected_predictions
from src.models.handoff import PREDICTION_COLUMNS, validate_predictions, wide_predictions
from src.models.pipeline import file_hash, write_csv
from src.models.walk_forward import ModelData, prepare_data, walk_forward
from src.models.extensions.config import (EXPLORATORY_NOTICE, EXT_REPORT_DIR, FEATURE_SETS, HORIZON,
                                          MODEL_SETS, OUTPUT_DIR, ExtensionConfig)
from src.models.extensions.handoff import RECOVERY_COLUMNS, recovery_wide, validate_recovery
from src.models.extensions.recovery import prepare_recovery, select_cs, selected_recovery, walk_recovery


@dataclass
class FamilyRun:
    predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    candidates: pd.DataFrame
    selection: pd.DataFrame
    chosen: dict[str, float]
    coverage: pd.DataFrame
    coefficients: pd.DataFrame
    outcomes: pd.DataFrame


@dataclass
class ExtensionRun:
    ridge: FamilyRun
    recovery: FamilyRun
    sensitivity_candidates: pd.DataFrame
    sensitivity_selection: pd.DataFrame
    sensitivity_coverage: pd.DataFrame


def period_dates(index: pd.DatetimeIndex, settings: ExtensionConfig) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    p = settings.periods
    validation = index[(index >= p.validation_start) & (index <= p.design_end)]
    evaluation = index[(index >= p.test_start) & (index <= p.test_end)]
    if validation.empty or evaluation.empty:
        raise ValueError("both design validation and evaluation require forecast dates")
    return validation, evaluation


def validate_coverage(coverage: pd.DataFrame) -> None:
    """Only an explicitly logged missing-calendar tail may interrupt evaluation."""
    if not coverage.status.isin(["predicted", "incomplete_outcome_calendar"]).all():
        raise ValueError("unexpected gap in evaluation forecast coverage")
    missing = coverage.status == "incomplete_outcome_calendar"
    if not coverage.loc[missing, "label_end_date"].isna().all():
        raise ValueError("calendar exclusion with a known outcome date")
    if missing.any() and not missing.iloc[missing.to_numpy().argmax():].all():
        raise ValueError("missing outcome calendar is allowed only at the end")


def ridge_validation(data: ModelData, settings: ExtensionConfig):
    validation_dates, _ = period_dates(data.features.index, settings)
    grid = {f"ridge_{name}": (name, settings.ridge_alphas) for name in RIDGE_SETS}
    run = walk_forward(data, validation_dates, grid, settings.periods.train_window,
                       outcome_cutoff=pd.Timestamp(settings.periods.design_end), keep_coefficients=False)
    if run.predictions.empty:
        raise ValueError("insufficient history for ridge validation")
    chosen, selection = select_alphas(run.predictions, data.outcomes, settings.periods.design_end)
    return run, chosen, selection


def run_extensions(panel: pd.DataFrame, features: pd.DataFrame, events: pd.DataFrame,
                   settings: ExtensionConfig = ExtensionConfig()) -> ExtensionRun:
    p = settings.periods
    required = panel.index[(panel.index >= p.validation_start) & (panel.index <= p.test_end)]
    if required.empty or not required.isin(features.index).all():
        raise ValueError("features must cover every required session")
    data = prepare_data(panel, features, horizon=HORIZON, allow_incomplete_tail=True)
    validation, chosen, selection = ridge_validation(data, settings)
    _, dates = period_dates(features.index, settings)
    plan = {model_id: (name, (chosen["vix_level" if model_id == "ridge_vix_matched" else name],))
            for model_id, name in RIDGE_MODELS.items()}
    forecast = walk_forward(data, dates, plan, p.train_window)
    validate_coverage(forecast.coverage)
    signals = forecast.predictions.loc[:, list(PREDICTION_COLUMNS)].sort_values(
        ["decision_date", "model_id"]).reset_index(drop=True)
    design_signals = selected_predictions(validation.predictions, chosen).loc[:, list(PREDICTION_COLUMNS)]
    validate_predictions(signals, expected_horizon=HORIZON)
    validate_predictions(design_signals, expected_horizon=HORIZON)
    ridge = FamilyRun(signals, design_signals, validation.predictions, selection, chosen,
                      pd.concat([validation.coverage.assign(period="validation"),
                                 forecast.coverage.assign(period="evaluation")], ignore_index=True),
                      forecast.coefficients, data.outcomes)

    recovery_data = prepare_recovery(panel, data, events)
    validation_dates, evaluation_dates = period_dates(recovery_data.features.index, settings)
    grid = {f"logit_{name}": (name, settings.classifier_cs) for name in FEATURE_SETS}
    validation = walk_recovery(recovery_data, validation_dates, grid, min_episodes=settings.min_episodes,
                               outcome_cutoff=pd.Timestamp(p.design_end))
    if validation.predictions.empty:
        raise ValueError("no completed recovery validation outcomes")
    chosen, selection = select_cs(validation.predictions, recovery_data.outcomes, p.design_end)
    plan = {model_id: (name, (chosen["vix_level" if model_id.endswith("_matched") else name],))
            for model_id, name in MODEL_SETS.items()}
    forecast = walk_recovery(recovery_data, evaluation_dates, plan, min_episodes=settings.min_episodes)
    validate_coverage(forecast.coverage)
    signals = forecast.predictions.loc[:, list(RECOVERY_COLUMNS)].sort_values(
        ["decision_date", "model_id"]).reset_index(drop=True)
    design_signals = selected_recovery(validation.predictions, chosen).loc[:, list(RECOVERY_COLUMNS)]
    validate_recovery(signals)
    validate_recovery(design_signals)
    recovery = FamilyRun(signals, design_signals, validation.predictions, selection, chosen,
                         pd.concat([validation.coverage.assign(period="validation"),
                                    forecast.coverage.assign(period="evaluation")], ignore_index=True),
                         forecast.coefficients, recovery_data.outcomes)

    # This sensitivity analysis deliberately never runs on evaluation dates.
    sensitivity, _, selection = ridge_validation(prepare_data(panel, features), settings)
    return ExtensionRun(ridge, recovery, sensitivity.predictions, selection, sensitivity.coverage)


def publish(staging: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(staging.iterdir(), key=lambda p: (p.name == "manifest.json", p.name)):
        with tempfile.NamedTemporaryFile(dir=destination, prefix=".extension-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(path.read_bytes())
        temporary.replace(destination / path.name)


def write_extensions(result: ExtensionRun, settings: ExtensionConfig, *, output_dir: Path = OUTPUT_DIR,
                     report_dir: Path = EXT_REPORT_DIR, input_paths: tuple[Path, ...] = ()) -> None:
    source_paths = sorted((config.ROOT / "src" / "models").rglob("*.py")) + [
        config.ROOT / "src" / "features" / name for name in ("events.py", "targets.py", "feature_build.py")]
    source_paths += [config.ROOT / "src/config.py", config.ROOT / "src/data/build.py",
                     config.ROOT / "requirements.txt", config.ROOT / "docs/model_extensions_spec.md"]
    common = {
        "schema_version": 1, "exploratory_notice": EXPLORATORY_NOTICE,
        "research_limitation": RESEARCH_LIMITATION, "settings": asdict(settings),
        "python_version": platform.python_version(),
        "package_versions": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow")},
        "source_sha256": {str(p.relative_to(config.ROOT)): file_hash(p) for p in source_paths},
        "input_sha256": {str(p.relative_to(config.ROOT)) if p.is_relative_to(config.ROOT) else p.name:
                         file_hash(p) for p in input_paths},
    }
    with tempfile.TemporaryDirectory(prefix="model-extensions-") as temp:
        stage = Path(temp)
        for family, run in (("ridge_h10", result.ridge), ("recovery", result.recovery)):
            signals, reports = stage / family / "signals", stage / family / "reports"
            signals.mkdir(parents=True)
            reports.mkdir()
            for name, predictions in (("predictions", run.predictions), ("validation_predictions", run.validation_predictions)):
                write_csv(predictions, signals / f"{name}.csv")
                wide = (recovery_wide(predictions) if family == "recovery" else
                        wide_predictions(predictions, expected_horizon=HORIZON))
                write_csv(wide, signals / f"{name}_wide.csv", index=True)
                dates = pd.DatetimeIndex(predictions.decision_date.unique())
                outcome_name = "validation_outcomes" if name.startswith("validation") else "outcomes"
                write_csv(run.outcomes.loc[dates].rename_axis("decision_date"), signals / f"{outcome_name}.csv", index=True)
            for name, frame in (("penalty_selection", run.selection), ("coverage", run.coverage),
                                ("coefficients", run.coefficients), ("validation_candidates", run.candidates)):
                write_csv(frame, reports / f"{name}.csv")
            manifest = {**common, "family": family, "horizon": HORIZON,
                        "selected_penalties": run.chosen,
                        "feature_sets": FEATURE_SETS if family == "recovery" else RIDGE_SETS,
                        "units": "estimated class probability" if family == "recovery" else "ten-day decimal excess total return",
                        "selection_objective": "episode-weighted three-class log loss" if family == "recovery" else "MSE",
                        "training_weights": "one total weight per matured episode; unweighted training-only scaler" if family == "recovery" else "equal rows",
                        "probability_class_order": [-1, 0, 1] if family == "recovery" else None,
                        "barrier_multiplier": K if family == "recovery" else None,
                        "label_maturity": "close of decision+11 trading sessions, including early barrier hits",
                        "decision_timing": "after 16:15 America/New_York; execute next session close",
                        "signal_sha256": {p.name: file_hash(p) for p in sorted(signals.iterdir())},
                        "report_sha256": {p.name: file_hash(p) for p in sorted(reports.iterdir())}}
            (signals / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        sensitivity = stage / "ridge_h1_design_sensitivity"
        sensitivity.mkdir()
        for name, frame in (("penalty_selection", result.sensitivity_selection),
                            ("validation_candidates", result.sensitivity_candidates),
                            ("coverage", result.sensitivity_coverage)):
            write_csv(frame, sensitivity / f"{name}.csv")
        (sensitivity / "manifest.json").write_text(json.dumps({**common, "family": "ridge_h1_design_sensitivity",
            "scope": "design period only; original evaluation forecasts not replaced",
            "report_sha256": {p.name: file_hash(p) for p in sorted(sensitivity.iterdir())}}, indent=2, allow_nan=False) + "\n")
        summary = stage / "summary"
        summary.mkdir()
        _summary(result, summary / "summary.md")
        # No publication begins until every family has been completely staged.
        for family in ("ridge_h10", "recovery"):
            publish(stage / family / "reports", Path(report_dir) / family)
            publish(stage / family / "signals", Path(output_dir) / family)
        publish(sensitivity, Path(report_dir) / "ridge_h1_design_sensitivity")
        publish(summary, Path(report_dir))


def _summary(result: ExtensionRun, path: Path) -> None:
    lines = ["# Exploratory modelling extensions", "", EXPLORATORY_NOTICE, "", RESEARCH_LIMITATION,
             "", "No evaluation performance metrics or trading results are calculated in this extension.", ""]
    for family, run in (("Ten-day ridge", result.ridge), ("Recovery classifier", result.recovery)):
        lines.extend([f"## {family}", "", "Selected settings (design data only):", "",
                      run.selection[run.selection.selected].to_markdown(index=False), "",
                      run.coverage.groupby(["period", "status"]).size().rename("n").reset_index().to_markdown(index=False), ""])
        if family == "Recovery classifier":
            p = run.predictions.drop_duplicates("decision_date")
            lines.extend([f"Evaluation trigger dates: {len(p)}. Fallback rows across all fitted model IDs: "
                          f"{int(run.predictions.fit_status.eq('prior_fallback').sum())}.", "",
                          "Episode weighting reduces concentration; it does not turn overlapping labels into independent samples.", ""])
    lines.extend(["## One-day penalty sensitivity (design only)", "",
                  result.sensitivity_selection[result.sensitivity_selection.selected].to_markdown(index=False), "",
                  "The original one-day settings, predictions and performance tables remain unchanged.", "",
                  "Read docs/model_extensions.md for the target definitions, sparse-signal contract, missing-tail handling and Member 4 examples.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    settings = ExtensionConfig()
    print(EXPLORATORY_NOTICE, flush=True)
    panel, features, events = load_panel(), load_features(), load_events()
    expected = panel.index[(panel.index >= config.SAMPLE_START) & (panel.index <= config.SAMPLE_END)]
    if not features.index.equals(expected):
        raise ValueError("saved features must cover the complete configured sample")
    print("Fitting ten-day ridge, direct recovery classifiers and design-only penalty sensitivity...", flush=True)
    result = run_extensions(panel, features, events, settings)
    paths = tuple(next(p for p in (config.PROCESSED_DIR / f"{name}.parquet", config.PROCESSED_DIR / f"{name}.csv")
                       if p.exists()) for name in ("panel", "features", "events"))
    write_extensions(result, settings, input_paths=paths)
    print(f"Ten-day ridge alphas: {result.ridge.chosen}")
    print(f"Recovery classifier Cs: {result.recovery.chosen}")
    print(f"Outputs: {OUTPUT_DIR}; model-build summary: {EXT_REPORT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
