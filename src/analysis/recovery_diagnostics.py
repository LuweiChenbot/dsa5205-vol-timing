"""Read-only diagnostics of the frozen recovery experiment; never fit or retune.

Run: python -m src.analysis.recovery_diagnostics
Protocol: docs/recovery_diagnostics_protocol.md
"""
from __future__ import annotations

from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.analysis.recovery_evaluation import (
    LABEL_NAMES, audit_bundle, digest, read_outcomes, score_rows,
)
from src.data.build import load_panel
from src.features.events import load_events
from src.models.extensions.config import EXT_REPORT_DIR, OUTPUT_DIR, PROBABILITY_COLUMNS
from src.models.extensions.handoff import load_recovery_predictions

REPORT_DIR = config.PROCESSED_DIR.parent.parent / "reports/recovery_diagnostics"
BIN_EDGES = np.array([0., .2, .4, .6, .8, 1.])
SIGN_TOLERANCE = 1e-10
PRIMARY = "logit_term_structure"


def weighted_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Inverse weighted empirical CDF at .1, .5 and .9, without interpolation."""
    if (len(values) == 0 or len(values) != len(weights) or not np.isfinite(values).all()
            or not np.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("finite observations and positive matching weights required")
    order = np.argsort(values, kind="stable")
    cdf = np.cumsum(weights[order]) / weights.sum()
    indices = np.minimum(np.searchsorted(cdf, [.1, .5, .9]), len(values) - 1)
    return values[order][indices]


def probability_diagnostics(scored: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Keep whole-sample episode weights fixed within class and calibration subsets."""
    summaries, distributions, confusion, bins = [], [], [], []
    for model_id, part in scored.groupby("model_id", sort=True):
        weights = 1 / part.groupby("episode_start_date").decision_date.transform("size").to_numpy()
        probabilities = part[list(PROBABILITY_COLUMNS)].to_numpy()
        maxima = probabilities == probabilities.max(axis=1, keepdims=True)
        predicted_mass = maxima / maxima.sum(axis=1, keepdims=True)
        labels = part.label.to_numpy(dtype=int)
        episodes = part.episode_start_date.to_numpy()
        for label, name in LABEL_NAMES.items():
            column = label + 1
            p, truth = probabilities[:, column], labels == label
            observed = float(np.average(truth, weights=weights))
            mean_p = float(np.average(p, weights=weights))
            margin = p - np.delete(probabilities, column, axis=1).max(axis=1)
            margin_quantiles = weighted_quantiles(margin, weights)
            # np.searchsorted assigns interior edges to the bin on their right;
            # clip includes the valid endpoint 1.0 in the final bin.
            bin_ids = np.clip(np.searchsorted(BIN_EDGES, p, side="right") - 1, 0, len(BIN_EDGES) - 2)
            ece, populated = 0., 0
            for bin_id, (left, right) in enumerate(zip(BIN_EDGES[:-1], BIN_EDGES[1:])):
                mask = bin_ids == bin_id
                mass = float(weights[mask].sum())
                predicted = float(np.average(p[mask], weights=weights[mask])) if mask.any() else np.nan
                frequency = float(np.average(truth[mask], weights=weights[mask])) if mask.any() else np.nan
                if mask.any():
                    ece += mass / weights.sum() * abs(predicted - frequency)
                    populated += 1
                bins.append({"model_id": model_id, "class_name": name, "bin_id": bin_id,
                             "left": left, "right": right, "right_inclusive": bin_id == len(BIN_EDGES) - 2,
                             "n_triggers": int(mask.sum()), "n_episodes": len(np.unique(episodes[mask])),
                             "episode_weight": mass, "mean_probability": predicted,
                             "observed_frequency": frequency, "forecast_minus_observed": predicted - frequency})
            summaries.append({"model_id": model_id, "class_name": name, "n_triggers": len(part),
                              "n_episodes": part.episode_start_date.nunique(), "n_actual": int(truth.sum()),
                              "observed_frequency": observed, "mean_probability": mean_p,
                              "forecast_minus_observed": mean_p - observed,
                              "one_vs_rest_auc": float(roc_auc_score(truth, p, sample_weight=weights))
                                  if truth.any() and not truth.all() else np.nan,
                              "binary_brier": float(np.average((p - truth) ** 2, weights=weights)),
                              "argmax_recall": float(np.average(predicted_mass[truth, column], weights=weights[truth]))
                                  if truth.any() else np.nan,
                              "expected_argmax_count": float(predicted_mass[:, column].sum()),
                              "margin_min": float(margin.min()), "margin_p10": margin_quantiles[0],
                              "margin_median": margin_quantiles[1], "margin_p90": margin_quantiles[2],
                              "margin_max": float(margin.max()), "ece_fixed_bins": ece,
                              "populated_bins": populated})
            for actual, actual_name in LABEL_NAMES.items():
                mask = labels == actual
                q = weighted_quantiles(p[mask], weights[mask]) if mask.any() else [np.nan] * 3
                distributions.append({"model_id": model_id, "forecast_class": name, "actual_class": actual_name,
                                      "n_triggers": int(mask.sum()), "n_episodes": len(np.unique(episodes[mask])),
                                      "episode_weight": float(weights[mask].sum()),
                                      "mean_probability": float(np.average(p[mask], weights=weights[mask])) if mask.any() else np.nan,
                                      "min_probability": float(p[mask].min()) if mask.any() else np.nan,
                                      "p10": q[0], "median": q[1], "p90": q[2],
                                      "max_probability": float(p[mask].max()) if mask.any() else np.nan})
                confusion.append({"model_id": model_id, "actual_class": actual_name, "predicted_class": name,
                                  "expected_count": float(predicted_mass[mask, column].sum()),
                                  "episode_weight": float((weights[mask] * predicted_mass[mask, column]).sum()),
                                  "row_fraction": float(np.average(predicted_mass[mask, column], weights=weights[mask]))
                                      if mask.any() else np.nan})
    return {"class_metrics.csv": pd.DataFrame(summaries), "probability_distributions.csv": pd.DataFrame(distributions),
            "confusion_matrices.csv": pd.DataFrame(confusion), "calibration_bins.csv": pd.DataFrame(bins)}


def candidate_episode_losses(candidates: pd.DataFrame, outcomes: pd.DataFrame,
                             manifest: dict, saved_validation: pd.DataFrame) -> pd.DataFrame:
    """Audit the complete saved grid, then average loss within each validation episode."""
    feature_sets = manifest["feature_sets"]
    dates = pd.DatetimeIndex(outcomes.decision_date).sort_values()
    periods = manifest["settings"]["periods"]
    if (outcomes.decision_date.duplicated().any() or not outcomes.label.isin(LABEL_NAMES).all()
            or not outcomes.decision_date.between(periods["validation_start"], periods["design_end"]).all()
            or not outcomes.label_end_date.le(pd.Timestamp(periods["design_end"])).all()):
        raise ValueError("candidate scoring requires unique, completed design-period outcomes")
    expected = {("class_prior", "none", None)} | {
        (f"logit_{name}", name, float(c)) for name in feature_sets for c in manifest["settings"]["classifier_cs"]}
    observed, pieces = set(), []
    for (model_id, feature, c), part in candidates.groupby(["model_id", "feature_set", "C"], dropna=False):
        observed.add((model_id, feature, None if pd.isna(c) else float(c)))
        if part.decision_date.duplicated().any() or not pd.DatetimeIndex(part.decision_date).sort_values().equals(dates):
            raise ValueError("all candidates must cover exactly the same validation dates")
        paired = part.merge(outcomes, on="decision_date", validate="one_to_one", suffixes=("", "_actual"))
        for column in ("execution_date", "label_end_date", "episode_start_date"):
            if not paired[column].equals(paired[f"{column}_actual"]):
                raise ValueError(f"candidate/outcome mismatch in {column}")
        p = paired[list(PROBABILITY_COLUMNS)].to_numpy()
        if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                or not np.allclose(p.sum(axis=1), 1., rtol=0., atol=1e-12)):
            raise ValueError("invalid candidate probabilities")
        correct = p[np.arange(len(paired)), paired.label.to_numpy(dtype=int) + 1]
        paired["log_loss"] = -np.log(np.clip(correct, np.finfo(float).eps, 1))
        loss = paired.groupby("episode_start_date").log_loss.mean().reset_index()
        pieces.append(loss.assign(model_id=model_id, feature_set=feature, C=c))
        if model_id == "class_prior" or c == manifest["selected_penalties"].get(feature):
            frozen = saved_validation[saved_validation.model_id == model_id].sort_values("decision_date")
            ordered = part.sort_values("decision_date")
            if (not np.array_equal(frozen.decision_date, ordered.decision_date)
                    or not np.array_equal(frozen[list(PROBABILITY_COLUMNS)], ordered[list(PROBABILITY_COLUMNS)])):
                raise ValueError("selected candidate differs from frozen validation forecasts")
    if observed != expected:
        raise ValueError("saved candidates do not exactly cover the frozen model/C grid")
    return pd.concat(pieces, ignore_index=True)


def selection_sensitivity(losses: pd.DataFrame, selected: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Rank existing candidates on full and omitted-episode scoring samples; do not select models."""
    episodes = sorted(losses.episode_start_date.unique())
    if len(episodes) < 2 or losses.duplicated(["episode_start_date", "model_id", "C"]).any():
        raise ValueError("at least two episodes with unique candidate losses are required")
    for _, part in losses.groupby(["model_id", "C"], dropna=False):
        if set(part.episode_start_date) != set(episodes) or not np.isfinite(part.log_loss).all():
            raise ValueError("every candidate needs complete finite episode losses")
    samples = []
    for omitted in [None, *episodes]:
        retained = losses if omitted is None else losses[losses.episode_start_date != omitted]
        table = retained.groupby(["model_id", "feature_set", "C"], dropna=False).log_loss.mean().reset_index()
        prior = table.loc[table.model_id == "class_prior", "log_loss"]
        if len(prior) != 1:
            raise ValueError("exactly one prior baseline is required")
        table["loss_minus_prior"] = table.log_loss - prior.iloc[0]
        table["rank_within_feature_set"] = np.nan
        table["originally_selected"] = False
        for feature, c in selected.items():
            ranked = table[table.feature_set == feature].sort_values(["log_loss", "C"])
            if ranked.empty or not ranked.C.eq(c).any():
                raise ValueError("frozen penalty missing from sensitivity candidates")
            table.loc[ranked.index, "rank_within_feature_set"] = np.arange(1, len(ranked) + 1)
            table.loc[ranked.index, "originally_selected"] = ranked.C.eq(c)
        table["sample"] = "full_validation" if omitted is None else "omit_one_episode"
        table["omitted_episode"] = pd.NaT if omitted is None else omitted
        table["n_episodes"] = len(episodes) - (omitted is not None)
        samples.append(table)
    ranks = pd.concat(samples, ignore_index=True)
    omitted = ranks[ranks["sample"] == "omit_one_episode"]
    counts = []
    for (feature, c), part in omitted[omitted.feature_set != "none"].groupby(["feature_set", "C"]):
        counts.append({"feature_set": feature, "C": c, "originally_selected": c == selected[feature],
                       "n_omissions": len(part), "n_ranked_first": int(part.rank_within_feature_set.eq(1).sum()),
                       "best_rank": int(part.rank_within_feature_set.min()), "worst_rank": int(part.rank_within_feature_set.max()),
                       "min_loss_minus_prior": float(part.loss_minus_prior.min()),
                       "max_loss_minus_prior": float(part.loss_minus_prior.max())})
    return {"validation_episode_losses.csv": losses, "validation_candidate_ranks.csv": ranks,
            "validation_rank_counts.csv": pd.DataFrame(counts)}


def coefficient_diagnostics(coefficients: pd.DataFrame, predictions: pd.DataFrame,
                            feature_sets: dict) -> dict[str, pd.DataFrame]:
    """Pairwise log-odds contrasts of saved fits in their changing scaled-feature units."""
    fitted = predictions[predictions.fit_status == "fit"]
    expected = {(row.model_id, row.decision_date, label) for row in fitted.itertuples() for label in LABEL_NAMES}
    keys = ["model_id", "decision_date", "class_label"]
    if coefficients.duplicated(keys).any() or set(coefficients[keys].itertuples(index=False, name=None)) != expected:
        raise ValueError("coefficient coverage must exactly match fitted model/date/classes")
    pieces = []
    feature_columns = set().union(*map(set, feature_sets.values()))
    for model, part in coefficients.groupby("model_id", sort=True):
        forecast = fitted[fitted.model_id == model].set_index("decision_date")
        if not np.array_equal(part.C.to_numpy(), forecast.loc[part.decision_date, "C"].to_numpy()):
            raise ValueError("coefficient penalties differ from saved forecasts")
        feature = forecast.feature_set.unique()
        if len(feature) != 1:
            raise ValueError("each model requires one feature set")
        columns = ["intercept", *feature_sets[feature[0]]]
        unused = list(feature_columns - set(columns))
        if not np.isfinite(part[columns]).all().all() or not part[unused].isna().all().all():
            raise ValueError("active coefficients must be finite and unused features empty")
        recovery = part[part.class_label == 1].set_index("decision_date")[columns].sort_index()
        for label in (-1, 0):
            current = part[part.class_label == label].set_index("decision_date")[columns].sort_index()
            contrast = (current - recovery).reset_index().melt(
                id_vars="decision_date", var_name="term", value_name="coefficient_contrast")
            pieces.append(contrast.assign(model_id=model, contrast=f"{LABEL_NAMES[label]}_minus_recovery"))
    if not pieces:
        raise ValueError("no fitted coefficients available for diagnostics")
    contrasts = pd.concat(pieces, ignore_index=True).sort_values(["model_id", "contrast", "term", "decision_date"])
    summary = []
    for (model, contrast, term), part in contrasts.groupby(["model_id", "contrast", "term"], sort=True):
        values = part.coefficient_contrast.to_numpy()
        signs = np.where(abs(values) <= SIGN_TOLERANCE, 0, np.sign(values))
        nonzero = signs[signs != 0]
        summary.append({"model_id": model, "contrast": contrast, "term": term, "n_fits": len(values),
                        "first_date": part.decision_date.iloc[0], "last_date": part.decision_date.iloc[-1],
                        "first": values[0], "last": values[-1], "min": float(values.min()), "max": float(values.max()),
                        "n_positive": int((signs > 0).sum()), "n_negative": int((signs < 0).sum()),
                        "n_near_zero": int((signs == 0).sum()),
                        "nonzero_sign_changes": int((nonzero[1:] != nonzero[:-1]).sum())})
    return {"coefficient_contrasts.csv": contrasts, "coefficient_stability.csv": pd.DataFrame(summary)}


def audit_reports(folder: Path, manifest: dict) -> dict[Path, str]:
    paths = {}
    for name, expected in manifest["report_sha256"].items():
        path = folder / name
        if digest(path) != expected:
            raise ValueError(f"frozen recovery report hash mismatch: {name}")
        paths[path] = expected
    return paths


def run_diagnostics(output_dir: Path = REPORT_DIR) -> dict[str, pd.DataFrame]:
    folder, report_folder = OUTPUT_DIR / "recovery", EXT_REPORT_DIR / "recovery"
    manifest, inputs = audit_bundle(folder, load_panel(), load_events())
    inputs.update(audit_reports(report_folder, manifest))
    protected = [config.PROCESSED_DIR, EXT_REPORT_DIR.parent, REPORT_DIR.parent / "recovery_evaluation"]
    if any(output_dir.resolve().is_relative_to(p.resolve()) for p in protected):
        raise ValueError("diagnostics must not overwrite frozen data, model reports or evaluation artifacts")
    if any(p.resolve().is_relative_to(output_dir.resolve()) for p in inputs):
        raise ValueError("diagnostic output directory must be separate from frozen input files")
    predictions = load_recovery_predictions(folder / "predictions.csv")
    evaluation = score_rows(predictions, read_outcomes(folder / "outcomes.csv"))
    validation = load_recovery_predictions(folder / "validation_predictions.csv")
    date_columns = ["decision_date", "execution_date", "label_end_date", "episode_start_date"]
    candidates = pd.read_csv(report_folder / "validation_candidates.csv", parse_dates=date_columns, float_precision="round_trip")
    losses = candidate_episode_losses(candidates, read_outcomes(folder / "validation_outcomes.csv"), manifest, validation)
    tables = probability_diagnostics(evaluation)
    tables.update(selection_sensitivity(losses, manifest["selected_penalties"]))
    full = tables["validation_candidate_ranks.csv"].query("sample == 'full_validation' and feature_set != 'none'")
    original = pd.read_csv(report_folder / "penalty_selection.csv", float_precision="round_trip")
    compare = full.merge(original, on=["feature_set", "C"], validate="one_to_one")
    if (len(compare) != len(original) or not np.allclose(compare.log_loss, compare.weighted_log_loss, rtol=0, atol=1e-14)
            or not np.array_equal(compare.rank_within_feature_set.eq(1), compare.selected)):
        raise ValueError("recomputed full validation ranks disagree with frozen selection")
    coefficients = pd.read_csv(report_folder / "coefficients.csv", parse_dates=["decision_date"], float_precision="round_trip")
    tables.update(coefficient_diagnostics(coefficients, predictions, manifest["feature_sets"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False, float_format="%.17g", date_format="%Y-%m-%d", lineterminator="\n")
    plot_calibration(tables["calibration_bins.csv"], output_dir / "calibration.png")
    write_report(tables, output_dir / "report.md")
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError(f"frozen input changed during diagnostics: {path}")
    sources = [Path(__file__), config.ROOT / "src/analysis/recovery_evaluation.py",
               config.ROOT / "docs/recovery_diagnostics_protocol.md"]
    audit = {"frozen_model_commit": "4d19649", "no_refitting_or_retuning": True,
             "selected_penalties_unchanged": manifest["selected_penalties"],
             "calibration_bin_edges": BIN_EDGES.tolist(), "coefficient_sign_tolerance": SIGN_TOLERANCE,
             "weighting": "one total weight per episode; fixed within probability subsets",
             "selection_sensitivity": "omit an episode from scoring saved candidates; no refits or new selections",
             "research_limitation": manifest["research_limitation"],
             "packages": {name: version(name) for name in ("numpy", "pandas", "scikit-learn", "matplotlib")},
             "input_sha256": {str(p.relative_to(config.ROOT)): h for p, h in inputs.items()},
             "analysis_source_sha256": {str(p.relative_to(config.ROOT)): digest(p) for p in sources},
             "output_sha256": {name: digest(output_dir / name) for name in [*tables, "calibration.png", "report.md"]}}
    (output_dir / "manifest.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    return tables


def plot_calibration(bins: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True)
    models = {"class_prior": "Historical prior", "logit_price": "Price", "logit_vix_level": "Price + VIX",
              PRIMARY: "Price + VIX + term structure"}
    for ax, name in zip(axes, LABEL_NAMES.values()):
        ax.plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
        for model, label in models.items():
            part = bins[(bins.model_id == model) & (bins.class_name == name)].sort_values("bin_id")
            ax.plot(part.mean_probability, part.observed_frequency, marker="o", markersize=4, label=label)
        ax.set(title=name.capitalize(), xlabel="Mean forecast probability", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Observed class frequency")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Frozen 2019–2025 probabilities: episode-weighted calibration", fontsize=12)
    fig.text(.5, .16, "Five fixed bins; sparse descriptive estimates, not fitted calibration. Counts are in calibration_bins.csv.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .19, 1, .94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(tables: dict[str, pd.DataFrame], path: Path) -> None:
    metrics = tables["class_metrics.csv"]
    continuation = metrics[metrics.class_name == "continuation"]
    primary = metrics[metrics.model_id == PRIMARY]
    matrix = tables["confusion_matrices.csv"].query("model_id == @PRIMARY").pivot(
        index="actual_class", columns="predicted_class", values="expected_count")
    names = list(LABEL_NAMES.values())
    matrix = matrix.reindex(index=names, columns=names)
    distributions = tables["probability_distributions.csv"].query("model_id == @PRIMARY and forecast_class == 'continuation'")
    counts = tables["validation_rank_counts.csv"]
    frozen = counts[counts.originally_selected]
    contrasts = tables["coefficient_stability.csv"].query("model_id == @PRIMARY and contrast == 'continuation_minus_recovery'")
    row = continuation[continuation.model_id == PRIMARY].iloc[0]
    text = f"""# Frozen recovery-model diagnostics

All inputs come from the original `4d19649` recovery snapshot. Forecasts, C values,
features, scaling, labels, thresholds and prior results remain unchanged. No model
was fitted, no calibrator was fitted, and no diagnostic winner was adopted.
Reproduce with `python -m src.analysis.recovery_diagnostics`.

These checks were requested after viewing the evaluation results. They diagnose
the existing experiment; they do not constitute a new untouched holdout or rescue
its primary result. On the frozen 2019–2025 evaluation, term-structure log loss
remains 1.017874 versus the historical prior's 0.977701. The previously reported
paired loss interval crosses zero. There is no established predictive improvement.

## 1. Why continuation never wins argmax

175 decline-trigger decisions represent 53 episodes. All probability summaries
use one total weight per episode. Unresolved outcomes stay in the analysis.

{continuation[['model_id', 'one_vs_rest_auc', 'mean_probability', 'observed_frequency', 'argmax_recall', 'margin_max']].to_markdown(index=False, floatfmt='.6f')}

For the designated term-structure model, continuation AUC is {row.one_vs_rest_auc:.6f}.
The largest continuation probability minus its strongest competitor is
{row.margin_max:.6f}; a negative value means continuation never has the highest
probability. This is a statement about the unchanged argmax rule, not a claim
that continuation probabilities are zero. AUC is descriptive ranking performance;
these additional class-specific estimates have no confidence intervals.

The term-structure model's continuation probabilities, grouped by actual outcome:

{distributions[['actual_class', 'n_triggers', 'mean_probability', 'min_probability', 'p10', 'median', 'p90', 'max_probability']].to_markdown(index=False, floatfmt='.6f')}

In this snapshot, continuation receives a slightly lower mean forecast on actual
continuations than on actual recoveries. Together with the AUC near 0.5, this
provides no convincing hidden continuation-ranking signal to recover merely by
changing the class decision rule. It does not prove that no signal exists.

Its confusion matrix below uses raw trigger counts (rows actual, columns
predicted). The CSV also provides episode-weighted masses and row-normalized
recalls. Exact ties split prediction mass evenly; missing actual classes yield
undefined recall. Neither thresholds nor probability order was changed.

{matrix.to_markdown(floatfmt='.3f')}

## 2. Probability calibration

For the designated model:

{primary[['class_name', 'mean_probability', 'observed_frequency', 'forecast_minus_observed', 'ece_fixed_bins', 'populated_bins']].to_markdown(index=False, floatfmt='.6f')}

Positive forecast-minus-observed means aggregate overprediction; negative means
underprediction. This is the episode-weighted population, not raw trigger counts.
Global bias can hide offsetting errors in different probability bins.
In this snapshot the largest absolute aggregate class bias is overprediction of
unresolved outcomes. Recovery is slightly underpredicted in aggregate despite
being the most frequent argmax prediction; argmax counts and calibration answer
different questions.

![Episode-weighted reliability curves](calibration.png)

Five fixed bins are [0,.2), [.2,.4), [.4,.6), [.6,.8), [.8,1]. The CSV retains
empty bins with undefined means and zero weight, and lists trigger/episode counts
for every bin. Weights are computed before binning and stay fixed. One episode
can span bins, so the bin episode counts are not additive. ECE is the weighted
absolute bin gap; it depends on the bins and is not a proper scoring rule or a
selection criterion. Sparse, dependent observations make these descriptive curves
uncertain. No statistically established miscalibration or superiority is claimed.

## 3. Validation sensitivity and saved coefficient stability

Only the original 32 completed validation triggers across ten episodes enter the
ranking sensitivity. Full-sample candidate losses and winners reproduce the
original selection. Below, each episode is omitted from SCORING in turn:

{frozen[['feature_set', 'C', 'n_omissions', 'n_ranked_first', 'best_rank', 'worst_rank', 'min_loss_minus_prior', 'max_loss_minus_prior']].to_markdown(index=False, floatfmt='.6f')}

All candidate winner counts, including alternatives to the frozen C values:

{counts[['feature_set', 'C', 'originally_selected', 'n_ranked_first']].to_markdown(index=False, floatfmt='.6f')}

This is not refitted cross-validation: an omitted validation episode may already
have entered training for later saved forecasts, and neighbouring label windows
can overlap. These are diagnostic rankings only. No C is changed, no grid is
expanded, and no evaluation outcome affects the ranks. Ten validation episodes
remain a very small design sample, even when a winner is stable to these omissions.
All three originally selected models retain lower validation loss than the prior
under every single-episode omission. Thus their validation advantage was not
entirely due to any one scored episode, but it still failed to carry over to the
primary evaluation. Price's unchanged C=10 is the upper edge of the original grid,
so its stable ranking within this grid does not establish a global optimum.

For the term-structure model, continuation-minus-recovery coefficient contrasts
over its saved 2019–2025 refits are:

{contrasts[['term', 'n_fits', 'first', 'last', 'min', 'max', 'n_positive', 'n_negative', 'nonzero_sign_changes']].to_markdown(index=False, floatfmt='.6f')}

These contrasts parameterize log(p_continuation / p_recovery). Positive slope
means a larger scaled feature increases that ratio, holding the others fixed.
The full CSV includes all fitted models and unresolved-minus-recovery contrasts.
Sign changes skip values within 1e-10 of zero. The observations are successive
fitted states, not independent coefficient estimates or causal effects. The
training scaler and clipping limits change between refits, so cross-date magnitude
comparisons are approximate. Validation coefficients were not saved; none were
refitted or fabricated for this report.
The term-structure contrast is positive in all 175 saved fits, while the VIX-level
contrast changes sign six times. A stable fitted direction for term structure
does not establish incremental predictive value; the unchanged primary comparison
with the VIX-level model still provides no established advantage.

## Handoff implications

Keep the original log-loss comparison as the primary result. These diagnostics
can explain model weaknesses; they cannot restore an untouched holdout or create
more independent decline episodes. A calibration plot or a favourable class AUC
does not establish profitable timing. Keep both prior and simple price baselines.

Member 4 should consume the original signal loaders, keep models frozen, define
allocation on design data, and explicitly handle non-decline dates, next-close
execution, initial trading costs, turnover and drifted weights. Probabilities are
not portfolio weights. Files in this diagnostics folder contain future outcomes
or retrospective summaries and must not be used as live trading inputs.

See `docs/recovery_diagnostics_protocol.md` for definitions and limitations,
`reports/recovery_evaluation/report.md` for the unchanged primary evaluation, and
`docs/model_extensions.md` for signal-loading examples. The manifest verifies
original input hashes, records analysis sources and hashes every output.
"""
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run_diagnostics()
    print(f"Frozen diagnostics saved to {REPORT_DIR}; no fitting or retuning.")
