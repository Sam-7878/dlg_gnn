"""Create strictly separated Round 4 main and Round 3 controlled results."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import DATASET_DIR, RAW_PREDICTION_DIR, RESULTS_DIR, ensure_dirs


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    result = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if mask.any():
            result += float(mask.mean() * abs(y[mask].mean() - p[mask].mean()))
    return result


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc_pr": float(average_precision_score(y, p)),
        "auc_roc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, p >= 0.5, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
        "ece": ece(y, p),
        "nll": float(log_loss(y, p, labels=[0, 1])),
    }


def ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args(); ensure_dirs()
    shutil.copy2(DATASET_DIR / "real_dataset_manifest.json", RESULTS_DIR / "real_dataset_manifest.json")
    methods = {1: "GNN Only", 10: "MC-GNN (T=10)"}
    run_rows: list[dict] = []
    calibration_rows: list[dict] = []
    for passes, method in methods.items():
        for path in sorted(RAW_PREDICTION_DIR.glob(f"seed*_T{passes}.csv")):
            frame = pd.read_csv(path)
            seed = int(path.stem.split("_")[0].replace("seed", ""))
            score = metrics(frame.label.to_numpy(int), frame.p_mean.to_numpy(float))
            run_rows.append({"method": method, "seed": seed, **score})
            calibration_rows.append({
                "track": "SCI Main Track", "method": method, "seed": seed,
                "brier": score["brier"], "ece": score["ece"], "nll": score["nll"],
                "n_test": len(frame), "gnn_source": "real_checkpoint",
                "split_type": "chronological_real",
            })
    runs = pd.DataFrame(run_rows)
    expected = {(method, seed) for method in methods.values() for seed in (7, 17, 27, 37, 47)}
    observed = set(zip(runs.get("method", []), runs.get("seed", [])))
    if observed != expected:
        raise RuntimeError(f"incomplete main runs: missing={sorted(expected - observed)}")

    summary_rows = []
    rng = np.random.default_rng(20260829)
    for method, group in runs.groupby("method", sort=False):
        row = {
            "track": "SCI Main Track", "method": method, "n_seeds": len(group),
            "gnn_source": "real_checkpoint", "split_type": "chronological_real",
            "timestamp_source": "recorded_transaction_timestamp",
            "context_used": False, "paper_eligible": True,
        }
        for metric in ("auc_pr", "auc_roc", "f1", "brier", "ece", "nll"):
            values = group[metric].to_numpy(float)
            low, high = ci(values, rng, args.bootstrap)
            row.update({f"mean_{metric}": values.mean(), f"std_{metric}": values.std(ddof=1),
                        f"ci95_low_{metric}": low, f"ci95_high_{metric}": high})
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "main_results.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(RESULTS_DIR / "calibration.csv", index=False)

    # Paired comparison operates on the same five seeds. Context fusion comparisons
    # are deliberately not promoted into the context-free main track.
    pivot = runs.pivot(index="seed", columns="method", values="auc_pr")
    differences = (pivot["MC-GNN (T=10)"] - pivot["GNN Only"]).to_numpy()
    low, high = ci(differences, rng, args.bootstrap)
    stats = [{
        "comparison": "MC-GNN (T=10) vs GNN Only", "metric": "auc_pr",
        "mean_paired_difference": differences.mean(), "ci95_low": low, "ci95_high": high,
        "n_seeds": len(differences), "n_bootstrap": args.bootstrap,
        "significant_95pct": bool(low > 0 or high < 0), "status": "evaluated",
    }]
    for comparison in (
        "Uncertainty Fusion vs GNN Only", "Uncertainty Fusion vs Fixed Fusion",
        "Uncertainty Fusion vs Learned Fusion",
    ):
        stats.append({
            "comparison": comparison, "metric": "auc_pr", "mean_paired_difference": np.nan,
            "ci95_low": np.nan, "ci95_high": np.nan, "n_seeds": 0,
            "n_bootstrap": args.bootstrap, "significant_95pct": False,
            "status": "not_applicable_context_excluded_from_main",
        })
    pd.DataFrame(stats).to_csv(RESULTS_DIR / "statistical_summary.csv", index=False)

    controlled_source = ROOT / "results" / "graphrag" / "round_3" / "real_main_results.csv"
    controlled = pd.read_csv(controlled_source)
    controlled.insert(0, "track", "Controlled Context-Augmentation Study")
    controlled["timestamp_source"] = "synthetic_context_timestamp"
    controlled["context_policy"] = "label-conditioned context"
    controlled["paper_eligible"] = False
    controlled["source_round"] = 3
    controlled.to_csv(RESULTS_DIR / "controlled_context_results.csv", index=False)
    print(json.dumps({"main_methods": list(methods.values()), "seed_count": 5,
                      "controlled_rows": len(controlled), "bootstrap": args.bootstrap}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
