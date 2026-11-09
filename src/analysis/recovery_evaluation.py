"""Score saved recovery probabilities without fitting or changing any model.

Run: python -m src.analysis.recovery_evaluation
Protocol: docs/recovery_evaluation_protocol.md
"""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.data.build import load_panel
from src.features.events import load_events
from src.models.extensions.config import OUTPUT_DIR, PROBABILITY_COLUMNS, MODEL_IDS
from src.models.extensions.handoff import load_recovery_predictions, validate_recovery

REPORT_DIR = config.PROCESSED_DIR.parent.parent / "reports" / "recovery_evaluation"
LABEL_NAMES = {-1: "continuation", 0: "unresolved", 1: "recovery"}
BOOTSTRAP_REPLICATES = 10000
BLOCK_EPISODES = 5
SEED = 5205


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def score_rows(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Require exact one-to-one outcome coverage, then add an unfitted uniform baseline."""
    validate_recovery(predictions)
    required = {"decision_date", "execution_date", "label_end_date", "episode_start_date", "label"}
    if set(outcomes.columns) != required or outcomes.columns.has_duplicates or outcomes.decision_date.duplicated().any():
        raise ValueError("outcomes must have the exact documented schema and unique decision dates")
    if not outcomes.label.isin(LABEL_NAMES).all():
        raise ValueError("all outcomes must be completed three-class labels")
    expected = pd.DatetimeIndex(predictions.decision_date.unique()).sort_values()
    if not pd.DatetimeIndex(outcomes.decision_date).sort_values().equals(expected):
        raise ValueError("outcomes must exactly match all prediction dates")
    scored = predictions.merge(outcomes, on="decision_date", how="left", validate="many_to_one", suffixes=("", "_actual"))
    for name in ("execution_date", "label_end_date", "episode_start_date"):
        if not scored[name].equals(scored[f"{name}_actual"]):
            raise ValueError(f"predictions/outcomes disagree on {name}")
        scored = scored.drop(columns=f"{name}_actual")
    uniform = scored[scored.model_id == "class_prior"].copy()
    uniform["model_id"] = "uniform"
    uniform["fit_status"] = "unfitted_uniform"
    uniform[list(PROBABILITY_COLUMNS)] = 1 / 3
    scored = pd.concat([scored, uniform], ignore_index=True)
    probabilities = scored[list(PROBABILITY_COLUMNS)].to_numpy()
    labels = scored.label.to_numpy(dtype=int) + 1
    correct = probabilities[np.arange(len(scored)), labels]
    scored["log_loss"] = -np.log(np.clip(correct, np.finfo(float).eps, 1))
    scored["brier"] = ((probabilities - np.eye(3)[labels]) ** 2).sum(axis=1)
    ties = probabilities == probabilities.max(axis=1, keepdims=True)
    scored["expected_correct"] = ties[np.arange(len(scored)), labels] / ties.sum(axis=1)
    return scored.sort_values(["decision_date", "model_id"]).reset_index(drop=True)


def metric_row(part: pd.DataFrame, *, weighting: str) -> dict:
    if weighting not in {"episode", "trigger"} or part.empty:
        raise ValueError("nonempty sample and supported weighting are required")
    weights = (1 / part.groupby("episode_start_date").decision_date.transform("size").to_numpy()
               if weighting == "episode" else np.ones(len(part)))
    result = {"n": len(part), "n_episodes": part.episode_start_date.nunique(),
              "log_loss": float(np.average(part.log_loss, weights=weights)),
              "brier": float(np.average(part.brier, weights=weights)),
              "accuracy": float(np.average(part.expected_correct, weights=weights))}
    recalls = []
    probabilities = part[list(PROBABILITY_COLUMNS)].to_numpy()
    maxima = probabilities == probabilities.max(axis=1, keepdims=True)
    predicted_mass = maxima / maxima.sum(axis=1, keepdims=True)
    for value, name in LABEL_NAMES.items():
        mask = part.label == value
        recall = float(np.average(part.loc[mask, "expected_correct"], weights=weights[mask])) if mask.any() else np.nan
        result[f"n_{name}"] = int(mask.sum())
        result[f"recall_{name}"] = recall
        result[f"expected_predictions_{name}"] = float(predicted_mass[:, value + 1].sum())
        recalls.append(recall)
    # A missing class makes a fixed-three-class macro recall undefined.
    result["macro_recall"] = float(np.mean(recalls))
    truth = part.label == 1
    result["recovery_vs_rest_auc"] = (float(roc_auc_score(truth, part.p_recovery, sample_weight=weights))
                                       if truth.nunique() == 2 else np.nan)
    return result


def metrics_table(scored: pd.DataFrame, *, period: str, include_subsets: bool = True) -> pd.DataFrame:
    rows = []
    for model_id, model in scored.groupby("model_id", sort=True):
        subsets = [("all_triggers", model)]
        if include_subsets:
            subsets += [("event_starts", model[model.decision_date == model.episode_start_date])]
        for subset, sample in subsets:
            if sample.empty:
                continue
            years = [("all", sample)]
            if include_subsets:
                years += [(str(year), group) for year, group in sample.groupby(sample.decision_date.dt.year)]
            for year, part in years:
                for weighting in ("episode", "trigger"):
                    rows.append({"period": period, "model_id": model_id, "subset": subset,
                                 "decision_year": year, "weighting": weighting,
                                 **metric_row(part, weighting=weighting)})
    return pd.DataFrame(rows)


def paired_intervals(scored: pd.DataFrame, *, block_size: int = BLOCK_EPISODES,
                     replicates: int = BOOTSTRAP_REPLICATES, seed: int = SEED) -> pd.DataFrame:
    """Percentile intervals on paired episode-mean loss differences; no model refits."""
    episodes = scored.groupby(["episode_start_date", "model_id"]).log_loss.mean().unstack().sort_index()
    n = len(episodes)
    if episodes.isna().any().any() or not isinstance(block_size, int) or not 1 <= block_size <= n:
        raise ValueError("paired complete episodes and a valid block length are required")
    if not isinstance(replicates, int) or replicates < 1:
        raise ValueError("positive integer bootstrap replicates required")
    rng = np.random.default_rng(seed)
    starts = rng.integers(n, size=(replicates, (n + block_size - 1) // block_size))
    indices = ((starts[:, :, None] + np.arange(block_size)) % n).reshape(replicates, -1)[:, :n]
    learned = [name for name in MODEL_IDS if name != "class_prior"]
    comparisons = [(model, baseline) for model in learned for baseline in ("class_prior", "uniform")]
    comparisons += [("logit_term_structure", "logit_price"),
                    ("logit_term_structure_matched", "logit_vix_level")]
    rows = []
    for model, comparator in comparisons:
        difference = (episodes[model] - episodes[comparator]).to_numpy()
        samples = difference[indices].mean(axis=1)
        lower, upper = np.quantile(samples, [.025, .975])
        baseline_loss = episodes[comparator].mean()
        rows.append({"model_id": model, "comparator": comparator, "n_episodes": n,
                     "loss_difference": float(difference.mean()),
                     "loss_change_pct": float(100 * difference.mean() / baseline_loss),
                     "ci95_lower": float(lower), "ci95_upper": float(upper),
                     "block_episodes": block_size, "replicates": replicates, "seed": seed})
    return pd.DataFrame(rows)


def read_outcomes(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["decision_date", "execution_date", "label_end_date", "episode_start_date"],
                       float_precision="round_trip")


def audit_bundle(folder: Path, panel: pd.DataFrame, events: pd.DataFrame) -> tuple[dict, dict[Path, str]]:
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    paths = {manifest_path: digest(manifest_path)}
    for name, expected in manifest["signal_sha256"].items():
        path = folder / name
        paths[path] = digest(path)
        if paths[path] != expected:
            raise ValueError(f"frozen artifact hash mismatch: {name}")
    for category in ("source_sha256", "input_sha256"):
        for name, expected in manifest[category].items():
            path = config.ROOT / name
            paths[path] = digest(path)
            if paths[path] != expected:
                raise ValueError(f"frozen provenance hash mismatch: {name}")
    predictions = load_recovery_predictions(folder / "predictions.csv")
    dates = pd.DatetimeIndex(predictions.decision_date.unique())
    periods = manifest["settings"]["periods"]
    expected = events.index[events.dec_cond & (events.index >= periods["test_start"]) &
                            (events.index <= periods["test_end"])]
    if not dates.equals(expected):
        raise ValueError("evaluation predictions do not cover all configured trigger dates")
    positions = panel.index.get_indexer(predictions.decision_date)
    if (positions < 0).any() or (positions + 11 >= len(panel)).any():
        raise ValueError("evaluation outcome calendar is incomplete")
    if (not np.array_equal(predictions.execution_date.to_numpy(), panel.index[positions + 1].to_numpy()) or
            not np.array_equal(predictions.label_end_date.to_numpy(), panel.index[positions + 11].to_numpy())):
        raise ValueError("prediction dates do not match the exchange calendar")
    for model, part in predictions.groupby("model_id"):
        if model != "class_prior":
            name = "vix_level" if model.endswith("_matched") else part.feature_set.iloc[0]
            if not part.C.eq(manifest["selected_penalties"][name]).all():
                raise ValueError("prediction C differs from the frozen selection")
    outcomes = read_outcomes(folder / "outcomes.csv")
    starts = outcomes[outcomes.decision_date == outcomes.episode_start_date]
    if not np.array_equal(starts.label.to_numpy(), events.loc[starts.decision_date, "fwd_ev_label"].to_numpy()):
        raise ValueError("dense labels disagree with original non-overlapping event labels")
    return manifest, paths


def evaluate_frozen(folder: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score artifacts only. This function has no estimator or selection calls."""
    panel, events = load_panel(), load_events()
    manifest, inputs = audit_bundle(folder, panel, events)
    evaluation = score_rows(load_recovery_predictions(folder / "predictions.csv"), read_outcomes(folder / "outcomes.csv"))
    validation = score_rows(load_recovery_predictions(folder / "validation_predictions.csv"),
                             read_outcomes(folder / "validation_outcomes.csv"))
    metrics = pd.concat([metrics_table(validation, period="validation", include_subsets=False),
                         metrics_table(evaluation, period="evaluation")], ignore_index=True)
    comparisons = paired_intervals(evaluation)
    counts = evaluation[evaluation.model_id == "class_prior"].groupby("episode_start_date").agg(
        n_triggers=("decision_date", "size"), first_decision=("decision_date", "min"),
        last_decision=("decision_date", "max"), last_label_end=("label_end_date", "max"))
    for value, name in LABEL_NAMES.items():
        counts[f"n_{name}"] = evaluation[evaluation.model_id == "class_prior"].groupby("episode_start_date").label.apply(
            lambda group: int((group == value).sum()))
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError(f"input changed while scoring: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"metrics.csv": metrics, "paired_loss_intervals.csv": comparisons,
               "episode_counts.csv": counts.reset_index(), "scored_predictions.csv": evaluation}
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False, float_format="%.17g", date_format="%Y-%m-%d", lineterminator="\n")
    _report(metrics, comparisons, manifest, output_dir / "report.md")
    source_paths = [Path(__file__), config.ROOT / "docs/recovery_evaluation_protocol.md"]
    audit = {"frozen_model_commit": "4d19649", "no_refitting_or_retuning": True,
             "selected_penalties": manifest["selected_penalties"],
             "primary_metric": "episode-weighted three-class log loss",
             "bootstrap": {"kind": "circular moving block on ordered episode losses", "block_episodes": BLOCK_EPISODES,
                           "replicates": BOOTSTRAP_REPLICATES, "seed": SEED, "confidence_level": .95},
             "research_limitation": manifest["research_limitation"],
             "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn")},
             "input_sha256": {str(p.relative_to(config.ROOT)): h for p, h in inputs.items()},
             "analysis_source_sha256": {str(p.relative_to(config.ROOT)): digest(p) for p in source_paths},
             "output_sha256": {name: digest(output_dir / name) for name in [*outputs, "report.md"]}}
    (output_dir / "manifest.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return metrics, comparisons


def _report(metrics: pd.DataFrame, comparisons: pd.DataFrame, manifest: dict, path: Path) -> None:
    main = metrics[(metrics.period == "evaluation") & (metrics.subset == "all_triggers") &
                   (metrics.decision_year == "all") & (metrics.weighting == "episode")]
    design = metrics[(metrics.period == "validation") & (metrics.weighting == "episode")]
    starts = metrics[(metrics.period == "evaluation") & (metrics.subset == "event_starts") &
                     (metrics.decision_year == "all") & (metrics.weighting == "episode")]
    columns = ["model_id", "log_loss", "brier", "accuracy", "macro_recall", "recovery_vs_rest_auc"]
    first = main.iloc[0]
    primary = comparisons[(comparisons.model_id == "logit_term_structure") & (comparisons.comparator == "class_prior")].iloc[0]
    count_columns = ["model_id", *[f"expected_predictions_{name}" for name in LABEL_NAMES.values()]]
    text = f"""# Frozen recovery classifier: 2019–2025 evaluation

Evaluated saved probabilities from model commit `4d19649`; no training, scaling,
penalty selection, threshold changes or probability reversal was performed.
Frozen C: {manifest['selected_penalties']}. The term-structure model remains the
designated direct-recovery model; this report does not replace it with a test winner.

{manifest['research_limitation']}

The classifier extension was also designed after the original one-day evaluation
was seen. This is a frozen-parameter retrospective evaluation, not a pristine
research holdout. It does not assess strategy profitability.

## Primary comparison: all triggers, equal total weight per episode

{int(first.n)} trigger decisions across {int(first.n_episodes)} represented episodes.
Class counts: {int(first.n_continuation)} continuation, {int(first.n_unresolved)}
unresolved, {int(first.n_recovery)} recovery. Unresolved labels remain a separate class.

{main[columns].to_markdown(index=False, floatfmt='.6f')}

Lower log loss and Brier score are better. Brier sums all three squared probability
errors (0–2 convention). Accuracy uses argmax and expected random tie-breaking;
uniform-baseline accuracy is 1/3. Macro recall averages all three class recalls
and is blank where a class has no observations. AUC ranks recovery against both
other classes combined, with the same episode weights. It is not the older
one-day model's recovery-versus-continuation diagnostic.

## Validation comparison (tuning data)

{design[['model_id', 'n', 'n_episodes', 'log_loss']].to_markdown(index=False, floatfmt='.6f')}

Validation selected C on these same observations and is therefore optimistic.
Changes in loss levels across periods also reflect different class mixtures and
market conditions; compare each model with its contemporaneous baselines.

## Direct answer for the designated recovery model

Term-structure classifier minus historical prior: {primary.loss_difference:+.6f}
log loss ({primary.loss_change_pct:+.3f}%); approximate paired interval
[{primary.ci95_lower:.6f}, {primary.ci95_upper:.6f}]. A positive difference is
worse. This comparison, rather than improvement over equal probabilities alone,
is the main check of whether the validation advantage carries over.

The following expected argmax prediction counts expose class collapse. They use
unweighted trigger counts; only tied probability maxima produce fractional counts.

{main[count_columns].to_markdown(index=False, floatfmt='.3f')}

Counts describe fixed argmax classification, not portfolio decisions. No threshold
or class-weight changes were made to improve these numbers.

## Paired uncertainty on the primary loss

{comparisons[['model_id', 'comparator', 'loss_difference', 'loss_change_pct', 'ci95_lower', 'ci95_upper']].to_markdown(index=False, floatfmt='.6f')}

Differences are named model minus comparator; negative favours the model. The
approximate 95% percentile intervals use {BOOTSTRAP_REPLICATES:,} paired circular
block resamples, with blocks of {BLOCK_EPISODES} consecutive episodes and seed
{SEED}. They preserve whole-episode weighting and some adjacent-episode dependence;
they do not assume {int(first.n)} independent daily observations. They are not
adjusted for multiple comparisons and do not account for earlier research exposure,
model-selection uncertainty or all forms of nonstationarity. An interval crossing
zero does not establish an improvement. Identical matched-model results are not
independent corroboration.

## Original event-start subset

{starts[['model_id', 'n', 'log_loss', 'brier', 'accuracy', 'recovery_vs_rest_auc']].to_markdown(index=False, floatfmt='.6f')}

This descriptive subset uses only decisions equal to their original episode start;
the labels were checked against the original event file. Episode counts for all
triggers can include an episode that started before the evaluation boundary.

## Artifacts and interpretation

`metrics.csv` includes trigger-weighted and episode-weighted results, per-class
recalls and decision-year diagnostics. Each yearly row recomputes episode weights
within its own sample; yearly averages do not simply average to the pooled result.
`paired_loss_intervals.csv`, `episode_counts.csv` and `scored_predictions.csv`
provide the underlying comparisons. The scored file includes future outcomes and
is an analysis artifact, never a trading-signal input.

The manifest records frozen input hashes and verifies that scoring left them
unchanged. Reproduce with `python -m src.analysis.recovery_evaluation`. Protocol:
`docs/recovery_evaluation_protocol.md`. Member 4 should use the original signal
loaders, define allocation and cost rules on design data, and retain baselines in
the backtest even if a model looks best in one reported metric.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    metrics, comparisons = evaluate_frozen(OUTPUT_DIR / "recovery", REPORT_DIR)
    primary = metrics[(metrics.period == "evaluation") & (metrics.subset == "all_triggers") &
                      (metrics.decision_year == "all") & (metrics.weighting == "episode")]
    print(primary[["model_id", "n", "n_episodes", "log_loss", "brier", "accuracy", "recovery_vs_rest_auc"]].to_string(index=False))
    print(comparisons[["model_id", "comparator", "loss_difference", "ci95_lower", "ci95_upper"]].to_string(index=False))
    print(f"Report: {REPORT_DIR / 'report.md'}; no model fitting or retuning performed.")


if __name__ == "__main__":
    main()
