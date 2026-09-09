"""Publication-readiness analyses that remain valid without retraining data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score


SEEDS = (7, 17, 27, 37, 47)
ANNOTATION_VERSION = "scam-benign-annotation-v1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def freeze_annotation_package(source: Path, output_dir: Path) -> dict[str, object]:
    frame = pd.read_csv(source, keep_default_na=False)
    forbidden_tokens = ("score", "probability", "prediction", "logit", "p_rag", "p_gnn")
    forbidden = [column for column in frame.columns if any(token in column.lower() for token in forbidden_tokens)]
    if forbidden:
        raise ValueError(f"annotation package contains model-answer hints: {forbidden}")
    required = {
        "sample_id", "campaign_id", "campaign_time", "campaign_title/text",
        "promoted_urls", "wallets", "domains", "source_links/identifiers",
        "CST_exact_hit", "CSDB_exact_hit", "annotation_1", "annotation_2",
        "final_label", "reason",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"annotation source missing columns: {missing}")
    frame.insert(0, "annotation_package_version", ANNOTATION_VERSION)
    frame["annotation_1"] = ""
    frame["annotation_2"] = ""
    frame["final_label"] = ""
    frame["reason"] = ""
    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "annotation_package_v1.csv"
    temporary = output_dir / "annotation_package_v1.tmp"
    frame.to_csv(temporary, index=False)
    temporary.replace(package_path)
    manifest = {
        "annotation_package_version": ANNOTATION_VERSION,
        "sample_count": int(len(frame)),
        "double_annotated_n": 0,
        "consensus_benign_n": 0,
        "contains_model_scores": False,
        "package_path": str(package_path),
        "package_sha256": sha256_file(package_path),
        "allowed_labels": ["BENIGN", "SCAM", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"],
        "admission_rule": "annotation_1=BENIGN and annotation_2=BENIGN and final_label=BENIGN",
        "status": "FROZEN_AWAITING_INDEPENDENT_HUMAN_ANNOTATION",
    }
    write_text(output_dir / "annotation_package_manifest.json", json.dumps(manifest, indent=2))
    return manifest


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            value += float(mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean()))
    return value


def adaptive_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    order = np.argsort(probabilities, kind="stable")
    value = 0.0
    for indices in np.array_split(order, bins):
        if len(indices):
            value += float(len(indices) / len(labels) * abs(labels[indices].mean() - probabilities[indices].mean()))
    return value


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return {
        "auc_pr": float(average_precision_score(labels, probabilities)),
        "auc_roc": float(roc_auc_score(labels, probabilities)) if np.unique(labels).size == 2 else float("nan"),
        "f1": float(f1_score(labels, probabilities >= 0.5, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "adaptive_ece": adaptive_ece(labels, probabilities),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def fit_temperature(validation_labels: np.ndarray, validation_probabilities: np.ndarray) -> float:
    """Fit one positive temperature using validation probabilities only."""
    labels = np.asarray(validation_labels, dtype=int)
    probabilities = np.clip(np.asarray(validation_probabilities, dtype=float), 1e-6, 1 - 1e-6)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("temperature scaling requires a non-empty two-class validation set")
    logits = np.log(probabilities / (1 - probabilities))

    def objective(log_temperature: float) -> float:
        temperature = np.exp(log_temperature)
        calibrated = 1 / (1 + np.exp(-np.clip(logits / temperature, -30, 30)))
        return float(log_loss(labels, calibrated, labels=[0, 1]))

    result = minimize_scalar(objective, bounds=(-4, 4), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return float(np.exp(result.x))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(probabilities / (1 - probabilities))
    return 1 / (1 + np.exp(-np.clip(logits / temperature, -30, 30)))


def load_prediction_panel(raw_dir: Path, passes: int) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    reference: pd.DataFrame | None = None
    scores: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        path = raw_dir / f"seed{seed}_T{passes}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).sort_values("event_id").reset_index(drop=True)
        identity = frame[["event_id", "timestamp", "label"]]
        if reference is None:
            reference = identity.copy()
        elif not identity.equals(reference):
            raise RuntimeError(f"prediction identity mismatch: {path}")
        scores[seed] = frame.p_mean.to_numpy(float)
    assert reference is not None
    return reference, scores


def build_calibration_results(raw_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity, deterministic = load_prediction_panel(raw_dir, 1)
    mc_identity, mc = load_prediction_panel(raw_dir, 10)
    if not identity.equals(mc_identity):
        raise RuntimeError("T=1 and T=10 test identities differ")
    labels = identity.label.to_numpy(int)
    rows: list[dict[str, object]] = []
    for method, panel in (("Deterministic GNN", deterministic), ("MC Dropout T=10", mc)):
        for seed, probability in panel.items():
            rows.append({
                "method": method, "seed": seed, "n_models": 1,
                "status": "COMPLETE_HELD_OUT_TEST", "fit_scope": "frozen_checkpoint",
                **classification_metrics(labels, probability),
            })
    deep = np.mean(np.stack(list(deterministic.values())), axis=0)
    mc_ensemble = np.mean(np.stack(list(mc.values())), axis=0)
    rows.append({
        "method": "Deep Ensemble", "seed": "ensemble", "n_models": len(deterministic),
        "status": "COMPLETE_HELD_OUT_TEST", "fit_scope": "five_independent_frozen_checkpoints",
        **classification_metrics(labels, deep),
    })
    rows.append({
        "method": "MC Dropout Ensemble T=10", "seed": "ensemble", "n_models": len(mc),
        "status": "COMPLETE_HELD_OUT_TEST", "fit_scope": "five_independent_frozen_checkpoints",
        **classification_metrics(labels, mc_ensemble),
    })
    rows.append({
        "method": "Temperature-Scaled GNN", "seed": "unavailable", "n_models": 5,
        "status": "BLOCKED_MISSING_VALIDATION_PREDICTIONS", "fit_scope": "validation_only_required",
        **{metric: np.nan for metric in ("auc_pr", "auc_roc", "f1", "brier", "ece", "adaptive_ece", "nll")},
    })
    calibration = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_dir / "calibration_baselines.csv", index=False)
    predictions = identity.copy()
    predictions["p_deep_ensemble"] = deep
    predictions["p_mc_dropout_ensemble_t10"] = mc_ensemble
    predictions.to_parquet(output_dir / "ensemble_predictions.parquet", index=False)
    return calibration, predictions


def _fast_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def paired_ap_comparison(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    n_randomization: int = 10_000,
    seed: int = 20260903,
) -> dict[str, float | int | bool]:
    labels = np.asarray(labels, dtype=int)
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if not (labels.shape == left.shape == right.shape):
        raise ValueError("paired arrays must have identical shape")
    rng = np.random.default_rng(seed)
    observed = _fast_ap(labels, left) - _fast_ap(labels, right)
    bootstrap = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        draw = rng.integers(0, len(labels), len(labels))
        bootstrap[index] = _fast_ap(labels[draw], left[draw]) - _fast_ap(labels[draw], right[draw])
    randomized = np.empty(n_randomization, dtype=float)
    for index in range(n_randomization):
        swap = rng.random(len(labels)) < 0.5
        perm_left = np.where(swap, right, left)
        perm_right = np.where(swap, left, right)
        randomized[index] = _fast_ap(labels, perm_left) - _fast_ap(labels, perm_right)
    return {
        "mean_auc_pr_difference": observed,
        "ci95_low": float(np.nanquantile(bootstrap, 0.025)),
        "ci95_high": float(np.nanquantile(bootstrap, 0.975)),
        "bootstrap_probability_gt_zero": float(np.nanmean(bootstrap > 0)),
        "randomization_p_value_two_sided": float((1 + np.sum(np.abs(randomized) >= abs(observed))) / (n_randomization + 1)),
        "n_bootstrap": n_bootstrap,
        "n_randomization": n_randomization,
        "n_events": len(labels),
        "n_positive": int(labels.sum()),
    }


def build_statistical_comparisons(
    raw_dir: Path,
    ensemble_predictions: pd.DataFrame,
    output_dir: Path,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    identity, deterministic = load_prediction_panel(raw_dir, 1)
    _, mc = load_prediction_panel(raw_dir, 10)
    labels = identity.label.to_numpy(int)
    # Five-seed mean metrics: event resampling is shared across all seed models.
    rng = np.random.default_rng(20260904)
    observed_seed_delta = np.mean([
        _fast_ap(labels, mc[seed]) - _fast_ap(labels, deterministic[seed]) for seed in SEEDS
    ])
    bootstrap = np.empty(n_resamples)
    for index in range(n_resamples):
        draw = rng.integers(0, len(labels), len(labels))
        bootstrap[index] = np.mean([
            _fast_ap(labels[draw], mc[seed][draw]) - _fast_ap(labels[draw], deterministic[seed][draw])
            for seed in SEEDS
        ])
    randomization = np.empty(n_resamples)
    for index in range(n_resamples):
        per_seed = []
        for seed in SEEDS:
            swap = rng.random(len(labels)) < 0.5
            left = np.where(swap, deterministic[seed], mc[seed])
            right = np.where(swap, mc[seed], deterministic[seed])
            per_seed.append(_fast_ap(labels, left) - _fast_ap(labels, right))
        randomization[index] = np.mean(per_seed)
    rows: list[dict[str, object]] = [{
        "comparison": "MC Dropout T=10 vs Deterministic GNN",
        "status": "COMPLETE",
        "mean_auc_pr_difference": observed_seed_delta,
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_probability_gt_zero": float(np.mean(bootstrap > 0)),
        "randomization_p_value_two_sided": float((1 + np.sum(np.abs(randomization) >= abs(observed_seed_delta))) / (n_resamples + 1)),
        "n_bootstrap": n_resamples,
        "n_randomization": n_resamples,
        "n_events": len(labels),
        "n_positive": int(labels.sum()),
        "aggregation": "event-resampled mean across five paired seeds",
    }]
    ensemble_stats = paired_ap_comparison(
        labels,
        ensemble_predictions.p_mc_dropout_ensemble_t10.to_numpy(float),
        ensemble_predictions.p_deep_ensemble.to_numpy(float),
        n_bootstrap=n_resamples,
        n_randomization=n_resamples,
        seed=20260905,
    )
    rows.append({
        "comparison": "MC Dropout Ensemble T=10 vs Deep Ensemble",
        "status": "COMPLETE", **ensemble_stats,
        "aggregation": "paired event-level ensemble probabilities",
    })
    for comparison, reason in (
        ("MC Dropout T=10 vs Temperature Scaling", "missing validation predictions"),
        ("Best Temporal Baseline vs Proposed GNN", "frozen training dataset unavailable"),
    ):
        rows.append({
            "comparison": comparison, "status": "BLOCKED", "mean_auc_pr_difference": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "bootstrap_probability_gt_zero": np.nan,
            "randomization_p_value_two_sided": np.nan, "n_bootstrap": 0,
            "n_randomization": 0, "n_events": 0, "n_positive": 0,
            "aggregation": reason,
        })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "statistical_comparisons.csv", index=False)
    return result


def build_temporal_shift_analysis(
    manifest: dict,
    predictions: pd.DataFrame,
    output_dir: Path,
    bins: int = 6,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, support in manifest["split"].items():
        rows.append({
            "level": "split", "slice": split, "method": "prevalence",
            "start_timestamp": support["start_time"], "end_timestamp": support["end_time"],
            "n": support["n_events"], "n_positive": support["n_positive"],
            "prevalence": support["fraud_ratio"],
            **{metric: np.nan for metric in ("auc_pr", "auc_roc", "f1", "brier", "ece", "adaptive_ece", "nll")},
            "status": "PREVALENCE_ONLY" if split != "test" else "HELD_OUT_TEST_AVAILABLE",
        })
    ordered = predictions.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
    for bin_index, indices in enumerate(np.array_split(np.arange(len(ordered)), bins), start=1):
        section = ordered.iloc[indices]
        labels = section.label.to_numpy(int)
        for method, column in (
            ("Deep Ensemble", "p_deep_ensemble"),
            ("MC Dropout Ensemble T=10", "p_mc_dropout_ensemble_t10"),
        ):
            rows.append({
                "level": "test_sequential_bin", "slice": f"test_bin_{bin_index}", "method": method,
                "start_timestamp": int(section.timestamp.min()), "end_timestamp": int(section.timestamp.max()),
                "n": len(section), "n_positive": int(labels.sum()), "prevalence": float(labels.mean()),
                **classification_metrics(labels, section[column].to_numpy(float)),
                "status": "COMPLETE" if np.unique(labels).size == 2 else "ONE_CLASS_METRIC_PARTIAL",
            })
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "temporal_shift_analysis.csv", index=False)
    return result


def reliability_bins(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0, 1, bins + 1)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        rows.append({
            "bin": index, "lower": low, "upper": high, "count": int(mask.sum()),
            "mean_confidence": float(probabilities[mask].mean()) if mask.any() else np.nan,
            "empirical_positive_rate": float(labels[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def generate_figures(
    manifest: dict,
    raw_dir: Path,
    predictions: pd.DataFrame,
    temporal: pd.DataFrame,
    mc_sensitivity_path: Path,
    figure_dir: Path,
    output_dir: Path,
) -> dict[str, bool]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 160, "font.size": 9})

    fig, axis = plt.subplots(figsize=(8.2, 2.8))
    axis.axis("off")
    boxes = [
        (0.02, "Recorded\ntransactions"), (0.22, "Causal snapshot\nedge time <= event time"),
        (0.48, "Frozen chronological\ntrain / validation / test"),
        (0.73, "Five-seed GNN\nand baselines"), (0.91, "Held-out ranking\nand calibration"),
    ]
    for x, label in boxes:
        axis.text(x, 0.55, label, ha="center", va="center", transform=axis.transAxes,
                  bbox={"boxstyle": "round,pad=0.45", "fc": "#e8f1fb", "ec": "#315d8a"})
    for left, right in zip(boxes[:-1], boxes[1:]):
        axis.annotate("", xy=(right[0] - 0.07, 0.55), xytext=(left[0] + 0.08, 0.55),
                      xycoords=axis.transAxes, arrowprops={"arrowstyle": "->", "lw": 1.4})
    axis.text(0.5, 0.12, "Selection uses validation only; the held-out test is evaluated once.",
              ha="center", transform=axis.transAxes, color="#444444")
    fig.tight_layout(); fig.savefig(figure_dir / "causal_pipeline.png", bbox_inches="tight"); plt.close(fig)

    split_names = ["train", "validation", "test"]
    prevalence = [manifest["split"][name]["fraud_ratio"] for name in split_names]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    axes[0].bar(split_names, prevalence, color=["#315d8a", "#6595c5", "#d2765e"])
    axes[0].set_ylabel("Fraud prevalence"); axes[0].set_ylim(0, max(prevalence) * 1.15)
    axes[0].set_title("Frozen chronological split")
    bins_frame = temporal[(temporal.level == "test_sequential_bin") & (temporal.method == "Deep Ensemble")]
    axes[1].plot(bins_frame.slice, bins_frame.prevalence, marker="o", color="#d2765e")
    axes[1].tick_params(axis="x", rotation=35); axes[1].set_ylabel("Fraud prevalence")
    axes[1].set_title("Held-out sequential slices")
    fig.tight_layout(); fig.savefig(figure_dir / "temporal_prevalence_shift.png", bbox_inches="tight"); plt.close(fig)

    labels = predictions.label.to_numpy(int)
    _, deterministic = load_prediction_panel(raw_dir, 1)
    reliability_methods = {
        "Deterministic (seed 7)": deterministic[7],
        "Deep Ensemble (5)": predictions.p_deep_ensemble.to_numpy(float),
        "MC Dropout Ensemble": predictions.p_mc_dropout_ensemble_t10.to_numpy(float),
    }
    reliability_rows = []
    fig, axis = plt.subplots(figsize=(4.6, 4.2))
    axis.plot([0, 1], [0, 1], "--", color="#777777", label="Perfect calibration")
    for method, probability in reliability_methods.items():
        frame = reliability_bins(labels, probability)
        frame["method"] = method; reliability_rows.append(frame)
        available = frame[frame["count"] > 0]
        axis.plot(available.mean_confidence, available.empirical_positive_rate, marker="o", label=method)
    axis.text(0.04, 0.94, "Temperature scaling unavailable:\nvalidation predictions missing",
              transform=axis.transAxes, va="top", fontsize=8,
              bbox={"fc": "white", "ec": "#aa5555", "alpha": 0.9})
    axis.set(xlabel="Mean predicted probability", ylabel="Empirical fraud rate", xlim=(0, 1), ylim=(0, 1))
    axis.legend(fontsize=7, loc="lower right"); fig.tight_layout()
    fig.savefig(figure_dir / "reliability_comparison.png", bbox_inches="tight"); plt.close(fig)
    pd.concat(reliability_rows, ignore_index=True).to_csv(output_dir / "reliability_bins.csv", index=False)

    sensitivity = pd.read_csv(mc_sensitivity_path)
    aggregate = sensitivity.groupby("T", as_index=False).agg(
        auc_pr=("auc_pr", "mean"), ece=("ece", "mean"), latency_ms=("latency_ms", "median")
    )
    fig, axis = plt.subplots(figsize=(5.4, 3.7)); latency_axis = axis.twinx()
    axis.plot(aggregate["T"], aggregate.auc_pr, marker="o", label="AUC-PR", color="#315d8a")
    axis.plot(aggregate["T"], aggregate.ece, marker="s", label="ECE", color="#d2765e")
    latency_axis.plot(aggregate["T"], aggregate.latency_ms, marker="^", label="Latency", color="#4f8a52")
    axis.set(xlabel="MC passes T", ylabel="Metric value"); latency_axis.set_ylabel("Median latency (ms)")
    lines = axis.lines + latency_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(figure_dir / "mc_tradeoff.png", bbox_inches="tight"); plt.close(fig)
    aggregate.to_csv(output_dir / "mc_tradeoff_summary.csv", index=False)
    return {
        "causal_pipeline": True,
        "temporal_prevalence_shift": True,
        "reliability_comparison_generated": True,
        "reliability_comparison_complete": False,
        "mc_tradeoff": True,
    }
