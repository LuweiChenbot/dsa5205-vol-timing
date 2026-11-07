"""Run the frozen offline experiment: python -m src.models.pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

import pandas as pd

from src import config
from src.data.build import load_panel
from src.features.events import load_events
from src.features.feature_build import load_features
from src.models.config import (FEATURE_SETS, MODEL_DIR, MODEL_SETS, PRIMARY_MODEL,
                               RESEARCH_LIMITATION, REPORT_DIR, ModelConfig)
from src.models.evaluation import (event_diagnostics, forecast_metrics, paired_comparisons,
                                   score_predictions, select_alphas, selected_predictions)
from src.models.handoff import PREDICTION_COLUMNS, validate_predictions, wide_predictions
from src.models.walk_forward import ModelData, prepare_data, walk_forward


@dataclass
class Experiment:
    data: ModelData
    predictions: pd.DataFrame
    validation_predictions: pd.DataFrame
    validation_candidates: pd.DataFrame
    selected_alphas: dict[str, float]
    selection: pd.DataFrame
    coverage: pd.DataFrame
    coefficients: pd.DataFrame
    metrics: pd.DataFrame
    comparisons: pd.DataFrame
    event_summary: pd.DataFrame
    event_predictions: pd.DataFrame


def run_experiment(panel: pd.DataFrame, features: pd.DataFrame, events: pd.DataFrame,
                   settings: ModelConfig = ModelConfig()) -> Experiment:
    data = prepare_data(panel, features)
    expected = panel.index[(panel.index >= settings.validation_start) & (panel.index <= settings.test_end)]
    if expected.empty or not expected.isin(features.index).all():
        raise ValueError("features do not cover every required validation/evaluation session")
    dates = features.index
    validation_dates = dates[(dates >= settings.validation_start) & (dates <= settings.design_end)]
    evaluation_dates = dates[(dates >= settings.test_start) & (dates <= settings.test_end)]
    if validation_dates.empty or evaluation_dates.empty:
        raise ValueError("validation and evaluation periods must both contain sessions")
    grid = {f"ridge_{name}": (name, settings.alphas) for name in FEATURE_SETS}
    validation = walk_forward(data, validation_dates, grid, settings.train_window,
                              outcome_cutoff=pd.Timestamp(settings.design_end), keep_coefficients=False)
    if validation.predictions.empty:
        raise ValueError("insufficient training history for any validation forecast")
    chosen, selection = select_alphas(validation.predictions, data.outcomes, settings.design_end)
    # Freeze the plan before the first evaluation forecast. Only training labels
    # already realised at each later decision are allowed to update coefficients.
    plan = {model_id: (feature_set, (chosen["vix_level" if model_id == "ridge_vix_matched"
                                           else feature_set],))
            for model_id, feature_set in MODEL_SETS.items()}
    evaluation = walk_forward(data, evaluation_dates, plan, settings.train_window)
    if not evaluation.coverage.status.eq("predicted").all():
        raise ValueError("incomplete evaluation coverage; increase training history, not test-time shortcuts")
    predictions = evaluation.predictions.loc[:, list(PREDICTION_COLUMNS)].sort_values(
        ["decision_date", "model_id"]).reset_index(drop=True)
    validation_predictions = selected_predictions(validation.predictions, chosen).loc[:, list(PREDICTION_COLUMNS)]
    validate_predictions(predictions)
    validate_predictions(validation_predictions)
    scored = score_predictions(predictions, data.outcomes)
    event_summary, event_rows = event_diagnostics(predictions, events)
    coverage = pd.concat([validation.coverage.assign(period="validation"),
                          evaluation.coverage.assign(period="evaluation")], ignore_index=True)
    return Experiment(data, predictions, validation_predictions, validation.predictions, chosen,
                      selection, coverage, evaluation.coefficients, forecast_metrics(scored),
                      paired_comparisons(scored), event_summary, event_rows)


def write_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    frame.to_csv(path, index=index, date_format="%Y-%m-%d", float_format="%.17g", lineterminator="\n")


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def write_outputs(result: Experiment, settings: ModelConfig,
                  output_dir: Path = MODEL_DIR, report_dir: Path = REPORT_DIR,
                  input_paths: tuple[Path, ...] = ()) -> None:
    """Stage complete artifacts before replacing output files; manifest is last.

    No raw inputs are changed. Hashes let consumers detect mismatched or partially
    copied artifacts. An interrupted publication cannot masquerade as a valid run.
    """
    output_dir, report_dir = Path(output_dir), Path(report_dir)
    with tempfile.TemporaryDirectory(prefix="vol-timing-models-") as temp:
        staging = Path(temp)
        signal_dir, reports = staging / "signals", staging / "reports"
        signal_dir.mkdir()
        reports.mkdir()
        for name, frame in (("predictions", result.predictions),
                            ("validation_predictions", result.validation_predictions)):
            write_csv(frame, signal_dir / f"{name}.csv")
            write_csv(wide_predictions(frame), signal_dir / f"{name}_wide.csv", index=True)
        dates = pd.DatetimeIndex(result.predictions.decision_date.unique())
        validation_dates = pd.DatetimeIndex(result.validation_predictions.decision_date.unique())
        write_csv(result.data.outcomes.loc[dates].rename_axis("decision_date"), signal_dir / "outcomes.csv", index=True)
        write_csv(result.data.outcomes.loc[validation_dates].rename_axis("decision_date"),
                  signal_dir / "validation_outcomes.csv", index=True)
        tables = {"alpha_selection": result.selection, "coverage": result.coverage,
                  "forecast_metrics": result.metrics, "feature_comparisons": result.comparisons,
                  "event_summary": result.event_summary, "event_predictions": result.event_predictions,
                  "coefficients": result.coefficients, "validation_candidates": result.validation_candidates}
        for name, frame in tables.items():
            write_csv(frame, reports / f"{name}.csv")
        _write_summary(result, settings, reports / "summary.md")
        _plot_results(result, reports)
        source_paths = sorted((config.ROOT / "src" / "models").glob("*.py")) + [
            config.ROOT / "src" / "config.py", config.ROOT / "src" / "features" / "feature_build.py",
            config.ROOT / "src" / "features" / "events.py", config.ROOT / "src" / "data" / "build.py",
            config.ROOT / "requirements.txt", config.ROOT / "docs" / "models_validation_spec.md"]
        manifest = {
            "schema_version": 1, "settings": asdict(settings), "feature_sets": FEATURE_SETS,
            "selected_alphas": result.selected_alphas,
            "matched_term_structure_alpha": result.selected_alphas["vix_level"],
            "primary_model": PRIMARY_MODEL, "selection_objective": "pooled validation MSE",
            "tie_break": "largest alpha for exact MSE ties", "target": "fwd_spy_xs",
            "units": "decimal excess total return; 0.001 means 0.1 percentage points",
            "timing": "decide after 16:15 America/New_York on t; execute close t+1; earn close t+1 to close t+2",
            "training": f"exactly {settings.train_window} matured labels; scaler and ridge refitted daily; intercept unpenalised",
            "research_limitation": RESEARCH_LIMITATION,
            "validation_use": "design/tuning data, not unbiased evaluation; selected alphas use this entire period",
            "evaluation_decisions": {"first": str(dates.min().date()), "last": str(dates.max().date()), "n": len(dates)},
            "last_evaluation_outcome": str(result.predictions.label_end_date.max().date()),
            "python_version": platform.python_version(),
            "package_versions": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "scipy",
                                                          "joblib", "threadpoolctl", "matplotlib", "pyarrow")},
            "input_sha256": {str(p.relative_to(config.ROOT)) if p.is_relative_to(config.ROOT) else p.name:
                             file_hash(p) for p in input_paths},
            "source_sha256": {str(p.relative_to(config.ROOT)): file_hash(p) for p in source_paths},
            "signal_sha256": {p.name: file_hash(p) for p in sorted(signal_dir.iterdir())},
            "report_sha256": {p.name: file_hash(p) for p in sorted(reports.iterdir())},
        }
        (signal_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        # Same-directory temporary files make each replacement atomic even when
        # the OS temporary directory is on another filesystem.
        for origin, destination in ((reports, report_dir), (signal_dir, output_dir)):
            for path in sorted(origin.iterdir(), key=lambda p: (p.name == "manifest.json", p.name)):
                with tempfile.NamedTemporaryFile(dir=destination, prefix=".model-", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(path.read_bytes())
                temporary.replace(destination / path.name)


def _write_summary(result: Experiment, settings: ModelConfig, path: Path) -> None:
    main = result.metrics[(result.metrics.decision_year == "all") & (result.metrics.segment == "all")]
    selected = result.selection[result.selection.selected][["feature_set", "alpha", "n", "mse"]]
    comparisons = result.comparisons[result.comparisons.segment.isin(["all", "decline"])][
        ["comparison", "segment", "n", "mse_change_pct"]]
    coverage = result.coverage.groupby(["period", "status"]).size().rename("n").reset_index()
    text = f"""# Ridge models and walk-forward results

Generated by `python -m src.models.pipeline`. Fixed {settings.train_window}-observation rolling
windows; decisions in {settings.test_start}–{settings.test_end}. These are forecast
diagnostics, not portfolio returns or evidence of profitability.

{RESEARCH_LIMITATION}

## Validation choices

Alpha chosen by pooled MSE on completed validation labels only; exact ties favour
larger alpha. Full ridge is the predeclared primary model. The matched term-structure
model uses the vix_level model's alpha.

{selected.to_markdown(index=False, floatfmt='.8g')}

## Coverage

{coverage.to_markdown(index=False)}

The last evaluation label ends on {result.predictions.label_end_date.max():%Y-%m-%d}.
Include trailing January 2026 sessions when Member 4 evaluates December decisions.

## Evaluation: all days

{main[['model_id', 'n', 'rmse', 'mae', 'oos_r2', 'direction_accuracy']].to_markdown(index=False, floatfmt='.6g')}

OOS R-squared is 1 minus model SSE divided by the rolling historical-mean SSE on
the same dates; negative values mean the model is worse. This is not the default
R-squared against the full evaluation-sample mean. The baseline therefore has zero.
Yearly and decline/non-decline results are in `forecast_metrics.csv`.

## Matched feature comparisons

{comparisons.to_markdown(index=False, floatfmt='.6g')}

Negative MSE change favours the larger feature set. Independently tuned comparisons
change both feature set and selected regularisation; `term_structure_same_alpha`
holds the penalty fixed. These descriptive differences have no significance claim.

## Event diagnostics

{result.event_summary[['model_id', 'n_events', 'n_recovery', 'n_continuation', 'n_unresolved', 'resolved_auc']].to_markdown(index=False, floatfmt='.6g')}

AUC ranks resolved recovery versus continuation events by the one-day forecast.
Unresolved events are counted and retained in `event_predictions.csv`, but excluded
from AUC. A one-day excess-return forecast is not a ten-day barrier probability;
this is a secondary, small-sample diagnostic, not a trained recovery classifier.

## Handoff

Read `docs/models_and_validation.md` for the complete protocol and Member 4 example.
Signal files contain no outcomes; outcomes are exported separately. Validation
predictions are design data selected using that same validation period, so their
performance is optimistic for the chosen alpha. Fit allocation rules on design
data and freeze them before evaluating 2019–2025; do not choose the best test model.
Coefficients are in scaled-feature units and are not accompanied by OLS p-values.
"""
    path.write_text(text, encoding="utf-8")


def _plot_results(result: Experiment, directory: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name in FEATURE_SETS:
        part = result.selection[result.selection.feature_set == name].sort_values("alpha")
        ax.plot(part.alpha, part.mse, marker="o", label=name)
    ax.set(xscale="log", xlabel="Ridge alpha", ylabel="Validation MSE (decimal return squared)",
           title="2017–2018 walk-forward validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "validation_mse.png", dpi=160)
    plt.close(fig)
    part = result.metrics[(result.metrics.decision_year != "all") &
                          (result.metrics.segment == "all") &
                          result.metrics.model_id.isin(["ridge_base", "ridge_vix_level", "ridge_vix", "ridge_full"])]
    pivot = part.pivot(index="decision_year", columns="model_id", values="oos_r2")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pivot.plot.bar(ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Decision year", ylabel="OOS R² vs rolling mean", title="Forecast performance by year")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(directory / "yearly_oos_r2.png", dpi=160)
    plt.close(fig)


def main() -> None:
    settings = ModelConfig()
    print("Running frozen ridge validation and evaluation on the processed snapshot...", flush=True)
    panel, features, events = load_panel(), load_features(), load_events()
    expected = panel.index[(panel.index >= config.SAMPLE_START) & (panel.index <= config.SAMPLE_END)]
    if not features.index.equals(expected):
        raise ValueError("features must exactly cover the configured 2014–2025 sample")
    result = run_experiment(panel, features, events, settings)
    inputs = tuple(next(p for p in (config.PROCESSED_DIR / f"{name}.parquet",
                                   config.PROCESSED_DIR / f"{name}.csv") if p.exists())
                   for name in ("panel", "features", "events"))
    write_outputs(result, settings, input_paths=inputs)
    print(f"Selected alphas: {result.selected_alphas}")
    print(f"Wrote {result.predictions.decision_date.nunique()} evaluation dates for six models to {MODEL_DIR}")
    print(f"Report: {REPORT_DIR / 'summary.md'}")
    print(RESEARCH_LIMITATION)


if __name__ == "__main__":
    main()
