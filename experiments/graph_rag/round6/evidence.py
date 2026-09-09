"""Package the preserved panel and complete analyses that need no training data."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.round5.analysis import load_prediction_panel


SEEDS = (7, 17, 27, 37, 47)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fast_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="stable")
    ranked = labels[order]
    positives = int(ranked.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(np.sum(precision * ranked) / positives)


def package_preserved_artifacts(round4_dir: Path, output_dir: Path) -> dict[str, object]:
    """Copy only hash-bound T=1/T=10 panels and manifest five checkpoints."""
    raw_output = output_dir / "raw_predictions"
    raw_output.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in SEEDS:
        for passes in (1, 10):
            source = round4_dir / "raw_predictions" / f"seed{seed}_T{passes}.csv"
            target = raw_output / source.name
            shutil.copy2(source, target)
            entries.append({
                "seed": seed,
                "passes": passes,
                "source_path": str(source),
                "packaged_path": str(target),
                "sha256": sha256_file(target),
                "bytes": target.stat().st_size,
            })
    raw_manifest = {
        "scope": "preserved held-out test predictions; no validation rows",
        "n_files": len(entries),
        "entries": entries,
    }
    (raw_output / "manifest.json").write_text(
        json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8"
    )

    checkpoint_entries = []
    for seed in SEEDS:
        checkpoint = round4_dir / "real_checkpoints" / f"seed{seed}.pt"
        source_manifest = round4_dir / "checkpoint_manifests" / f"seed{seed}.json"
        metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
        checkpoint_entries.append({
            "seed": seed,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256_file(source_manifest),
            "dataset_sha256": metadata.get("dataset_sha256"),
            "split_type": metadata.get("split_type", "chronological_real"),
        })
    checkpoint_manifest = {
        "model_class": "CausalLocalGIN",
        "seed_count": len(checkpoint_entries),
        "test_access_policy": "preserved checkpoints; no Round 6 refit",
        "entries": checkpoint_entries,
    }
    (output_dir / "checkpoints_manifest.json").write_text(
        json.dumps(checkpoint_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {"raw_predictions": raw_manifest, "checkpoints": checkpoint_manifest}


def build_model_metrics(output_dir: Path) -> pd.DataFrame:
    calibration = pd.read_csv(output_dir / "calibration_baselines.csv")
    calibration.insert(0, "model_group", "calibration")
    calibration["reason"] = ""
    metric_columns = [
        "model_group", "method", "seed", "n_models", "status", "fit_scope",
        "auc_pr", "auc_roc", "f1", "brier", "ece", "adaptive_ece", "nll", "reason",
    ]
    rows = [calibration.reindex(columns=metric_columns)]
    for filename, group in (
        ("temporal_baselines.csv", "temporal_baseline"),
        ("fraud_specific_baseline.csv", "fraud_specific_baseline"),
    ):
        frame = pd.read_csv(output_dir / filename)
        frame.insert(0, "model_group", group)
        frame["seed"] = "blocked"
        frame["n_models"] = 0
        frame["fit_scope"] = "frozen_v1_required"
        frame["adaptive_ece"] = np.nan
        rows.append(frame.reindex(columns=metric_columns))
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_dir / "model_metrics.csv", index=False)
    return result


def build_temporal_slice_metrics(output_dir: Path) -> pd.DataFrame:
    source = pd.read_csv(output_dir / "temporal_shift_analysis.csv")
    complete = source[source["level"] == "test_sequential_bin"].copy()
    blocked = []
    for slice_name, section in complete.groupby("slice", sort=False):
        reference = section.iloc[0]
        for method, reason in (
            ("Best temporal baseline", "frozen v1 train/validation data unavailable"),
            ("Temperature-Scaled GNN", "validation predictions unavailable"),
        ):
            blocked.append({
                "level": "test_sequential_bin", "slice": slice_name, "method": method,
                "start_timestamp": reference.start_timestamp,
                "end_timestamp": reference.end_timestamp,
                "n": reference.n, "n_positive": reference.n_positive,
                "prevalence": reference.prevalence,
                "auc_pr": np.nan, "auc_roc": np.nan, "f1": np.nan,
                "brier": np.nan, "ece": np.nan, "adaptive_ece": np.nan, "nll": np.nan,
                "status": "BLOCKED", "reason": reason,
            })
    complete["reason"] = ""
    result = pd.concat((complete, pd.DataFrame(blocked)), ignore_index=True)
    result.to_csv(output_dir / "temporal_slice_metrics.csv", index=False)
    return result


def _stratified_draws(labels: np.ndarray, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    for _ in range(n_resamples):
        draw = np.concatenate((
            rng.choice(positive, size=len(positive), replace=True),
            rng.choice(negative, size=len(negative), replace=True),
        ))
        rng.shuffle(draw)
        yield draw


def build_positive_count_sensitivity(
    round4_dir: Path,
    output_dir: Path,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    identity, deterministic = load_prediction_panel(round4_dir / "raw_predictions", 1)
    _, mc = load_prediction_panel(round4_dir / "raw_predictions", 10)
    labels = identity.label.to_numpy(int)
    deep = np.mean(np.stack(list(deterministic.values())), axis=0)
    mc_ensemble = np.mean(np.stack(list(mc.values())), axis=0)

    definitions = (
        (
            "MC Dropout T=10 vs Deterministic GNN",
            lambda draw: float(np.mean([
                _fast_ap(labels[draw], mc[s][draw]) - _fast_ap(labels[draw], deterministic[s][draw])
                for s in SEEDS
            ])),
            float(np.mean([
                _fast_ap(labels, mc[s]) - _fast_ap(labels, deterministic[s]) for s in SEEDS
            ])),
        ),
        (
            "MC Dropout Ensemble T=10 vs Deep Ensemble",
            lambda draw: _fast_ap(labels[draw], mc_ensemble[draw]) - _fast_ap(labels[draw], deep[draw]),
            _fast_ap(labels, mc_ensemble) - _fast_ap(labels, deep),
        ),
    )
    rows = []
    for offset, (comparison, statistic, observed) in enumerate(definitions):
        samples = np.fromiter(
            (statistic(draw) for draw in _stratified_draws(labels, n_resamples, 20260906 + offset)),
            dtype=float, count=n_resamples,
        )
        rows.append({
            "comparison": comparison,
            "status": "COMPLETE",
            "bootstrap_scheme": "class-stratified event bootstrap",
            "mean_auc_pr_difference": observed,
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "bootstrap_probability_gt_zero": float(np.mean(samples > 0)),
            "n_bootstrap": n_resamples,
            "n_events": len(labels),
            "n_positive_fixed": int(labels.sum()),
        })
    sensitivity = pd.DataFrame(rows)
    sensitivity.to_csv(output_dir / "positive_count_sensitivity.csv", index=False)

    comparisons = pd.read_csv(output_dir / "statistical_comparisons.csv")
    if "bootstrap_scheme" in comparisons.columns:
        comparisons = comparisons[
            comparisons.bootstrap_scheme != "class-stratified event bootstrap"
        ].copy()
    comparisons["bootstrap_scheme"] = np.where(
        comparisons["status"] == "COMPLETE", "ordinary paired event bootstrap", "not run"
    )
    additions = sensitivity.copy()
    additions["randomization_p_value_two_sided"] = np.nan
    additions["n_randomization"] = 0
    additions["n_positive"] = additions["n_positive_fixed"]
    additions["aggregation"] = "positive/negative counts held fixed at 107/3541"
    additions = additions.drop(columns=["n_positive_fixed"])
    pd.concat((comparisons, additions), ignore_index=True, sort=False).to_csv(
        output_dir / "statistical_comparisons.csv", index=False
    )
    return sensitivity
