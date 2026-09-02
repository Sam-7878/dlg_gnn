"""Paired statistical closure for the SCI-v3 R2 production path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(cfg["output_root"])
    metrics = pd.read_csv(root / "cascade/calibrated_cascade_metrics.csv")
    seed_pairs, mcnemar = [], []
    sample_deltas: dict[int, np.ndarray] = {}
    for seed in map(int, cfg["seeds"]):
        prediction = pd.read_csv(root / f"cascade/predictions/ProductionLevel1GIN__seed{seed}.csv")
        labels = prediction.label.to_numpy(dtype=int)
        fast_prediction = prediction.fast_prediction.to_numpy(dtype=int)
        final_prediction = prediction.final_prediction.to_numpy(dtype=int)
        fast_correct = fast_prediction == labels
        final_correct = final_prediction == labels
        wrong_to_correct = int((~fast_correct & final_correct).sum())
        correct_to_wrong = int((fast_correct & ~final_correct).sum())
        discordant = wrong_to_correct + correct_to_wrong
        p_value = float(binomtest(min(wrong_to_correct, correct_to_wrong), discordant, 0.5).pvalue) if discordant else 1.0
        mcnemar.append({"seed": seed, "comparison": "calibrated_cascade_vs_calibrated_GIN",
                         "wrong_to_correct": wrong_to_correct, "correct_to_wrong": correct_to_wrong,
                         "discordant": discordant, "p_value_exact": p_value})
        fast_row = metrics[(metrics.seed == seed) & (metrics.model == "ProductionLevel1GIN")].iloc[0]
        final_row = metrics[(metrics.seed == seed) & (metrics.model.str.contains("GATv2"))].iloc[0]
        seed_pairs.append({"seed": seed, "comparison": "calibrated_cascade_vs_calibrated_GIN",
            **{f"fast_{metric}": fast_row[metric] for metric in ("roc_auc", "pr_auc", "f1", "fraud_recall", "precision", "mcc")},
            **{f"cascade_{metric}": final_row[metric] for metric in ("roc_auc", "pr_auc", "f1", "fraud_recall", "precision", "mcc")},
            **{f"delta_{metric}": final_row[metric]-fast_row[metric] for metric in ("roc_auc", "pr_auc", "f1", "fraud_recall", "precision", "mcc")}})
        sample_deltas[seed] = final_correct.astype(float) - fast_correct.astype(float)
    adjusted = holm_adjust([row["p_value_exact"] for row in mcnemar])
    for row, value in zip(mcnemar, adjusted):
        row["p_value_holm"] = value
        row["reject_h0_familywise_0_05"] = value < float(cfg["statistics"]["familywise_alpha"])

    rng = np.random.default_rng(int(cfg["statistics"]["bootstrap_seed"]))
    seeds = np.asarray(list(sample_deltas))
    f1_by_seed = np.asarray([row["delta_f1"] for row in seed_pairs])
    replicates = int(cfg["statistics"]["bootstrap_resamples"])
    hierarchical, seed_bootstrap = np.empty(replicates), np.empty(replicates)
    for index in range(replicates):
        sampled_seeds = rng.choice(seeds, len(seeds), replace=True)
        seed_bootstrap[index] = np.mean([f1_by_seed[np.where(seeds == seed)[0][0]] for seed in sampled_seeds])
        within = []
        for seed in sampled_seeds:
            values = sample_deltas[int(seed)]
            within.append(float(rng.choice(values, len(values), replace=True).mean()))
        hierarchical[index] = np.mean(within)
    alpha = 1.0 - float(cfg["statistics"]["confidence_level"])
    ci_rows = [
        {"estimand": "mean_seed_delta_f1", "estimate": float(f1_by_seed.mean()),
         "ci_low": float(np.quantile(seed_bootstrap, alpha/2)), "ci_high": float(np.quantile(seed_bootstrap, 1-alpha/2)),
         "resampling_unit": "seed", "resamples": replicates, "random_seed": int(cfg["statistics"]["bootstrap_seed"])},
        {"estimand": "mean_paired_accuracy_gain", "estimate": float(np.mean([v.mean() for v in sample_deltas.values()])),
         "ci_low": float(np.quantile(hierarchical, alpha/2)), "ci_high": float(np.quantile(hierarchical, 1-alpha/2)),
         "resampling_unit": "seed_then_paired_sample", "resamples": replicates, "random_seed": int(cfg["statistics"]["bootstrap_seed"])},
    ]
    seed_frame = pd.DataFrame(seed_pairs)
    significant_seeds = int(sum(row["reject_h0_familywise_0_05"] for row in mcnemar))
    f1_ci = ci_rows[0]
    status = "SUPPORTED" if f1_ci["ci_low"] > 0 and significant_seeds >= 3 else "DESCRIPTIVE_ONLY"
    reason = ("positive seed-bootstrap CI and Holm-adjusted paired tests support the improvement"
              if status == "SUPPORTED" else "multiplicity-controlled paired evidence is insufficient for a confirmatory improvement claim")
    claim = {"claim_id": "C-GIN-CASCADE-F1", "status": status, "reason": reason,
             "mean_delta_f1": float(seed_frame.delta_f1.mean()), "ci_low": f1_ci["ci_low"], "ci_high": f1_ci["ci_high"],
             "significant_seed_tests": significant_seeds, "total_seed_tests": len(mcnemar),
             "multiplicity": "Holm family over five fixed-seed exact McNemar tests"}
    atomic_csv(root / "statistics/seed_pairs.csv", seed_frame)
    atomic_csv(root / "statistics/mcnemar_holm.csv", pd.DataFrame(mcnemar))
    atomic_csv(root / "statistics/bootstrap_ci.csv", pd.DataFrame(ci_rows))
    atomic_csv(root / "statistics/claim_status.csv", pd.DataFrame([claim]))
    atomic_json(root / "statistics/claim_status.json", claim)
    print(json.dumps(claim, indent=2))


if __name__ == "__main__":
    main()
