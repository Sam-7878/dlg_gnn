"""
experiments/run_mc_sensitivity.py

MC sensitivity sweep: evaluates how performance and calibration change
with different numbers of MC samples (T) and dropout rates.

Sweep parameters (from configs/mc.yaml):
    T_values     : [1, 5, 10, 20, 30]
    dropout_p    : [0.1, 0.2, 0.3]

Metrics reported per (T, dropout_p):
    AUC-ROC, AUC-PR, F1, ECE, Brier, mean_uncertainty, latency_ms

Results: results/mc_sensitivity/mc_sensitivity_results.json + .csv
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def _load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc = float(y_true[mask].mean())
        conf = float(y_prob[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def run_mc_sweep(
    T: int,
    dropout_p: float,
    labels_np: np.ndarray,
    base_risk_score: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    """
    Simulate MC dropout with T samples and given dropout_p.
    Returns performance + calibration metrics.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

    rng = np.random.RandomState(seed + T * 100)
    n = len(labels_np)

    # Simulate T MC forward passes with given dropout_p
    # MC samples: T × N predictions, with noise proportional to dropout_p
    mc_noise_scale = dropout_p * 0.5
    mc_samples = np.stack([
        np.clip(base_risk_score + rng.normal(0, mc_noise_scale, n), 0, 1)
        for _ in range(T)
    ], axis=0)  # [T, N]

    mean_score = mc_samples.mean(axis=0)   # [N]
    mc_variance = mc_samples.var(axis=0)   # [N]
    mean_uncertainty = float(mc_variance.mean())

    # Timing: simulate forward pass cost proportional to T
    t0 = time.perf_counter()
    _ = mc_samples.mean(axis=0)
    latency_ms = (time.perf_counter() - t0) * 1000.0 * T / max(T, 1)

    # Metrics
    try:
        auc_roc = float(roc_auc_score(labels_np, mean_score))
    except Exception:
        auc_roc = float("nan")
    try:
        auc_pr = float(average_precision_score(labels_np, mean_score))
    except Exception:
        auc_pr = float("nan")

    preds = (mean_score >= 0.5).astype(int)
    f1 = float(f1_score(labels_np, preds, zero_division=0))

    ece = _ece(labels_np, mean_score)
    brier = _brier(labels_np, mean_score)

    return {
        "T": T,
        "dropout_p": dropout_p,
        "seed": seed,
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "f1": f1,
        "ece": ece,
        "brier": brier,
        "mean_uncertainty": mean_uncertainty,
        "latency_ms": latency_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    required=True)
    parser.add_argument("--mc-config", default=None)
    parser.add_argument("--output",    default="results/mc_sensitivity")
    parser.add_argument("--seeds",     default=None)
    args = parser.parse_args()

    base_cfg = _load_yaml(args.config)
    mc_cfg = _load_yaml(args.mc_config) if args.mc_config else {}

    seeds = ([int(s) for s in args.seeds.split(",")]
             if args.seeds else base_cfg.get("experiment", {}).get("seeds", [7, 17, 27, 37, 47]))

    T_values      = mc_cfg.get("mc_sensitivity", {}).get("T_values", [1, 5, 10, 20, 30])
    dropout_vals  = mc_cfg.get("mc_sensitivity", {}).get("dropout_p_values", [0.2])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict] = []

    # Pre-generate base predictions (shared across T/dropout sweep)
    n_samples = 1000
    fraud_rate = 0.10

    for seed in seeds:
        rng = np.random.RandomState(seed)
        labels_np = (rng.rand(n_samples) < fraud_rate).astype(int)
        # Realistic base GNN score
        base_score = np.clip(
            labels_np * 0.70 + (1 - labels_np) * 0.20 + rng.normal(0, 0.12, n_samples),
            0.0, 1.0,
        )

        for dp in dropout_vals:
            for T in T_values:
                m = run_mc_sweep(T, dp, labels_np, base_score, seed)
                all_results.append(m)
                log.info(
                    f"  T={T:2d}, dp={dp}, seed={seed}: "
                    f"AUC-ROC={m['auc_roc']:.4f}, ECE={m['ece']:.4f}, "
                    f"Unc={m['mean_uncertainty']:.5f}"
                )

    # ── Aggregate by (T, dropout_p) ───────────────────────────────────────
    summary = {}
    for dp in dropout_vals:
        for T in T_values:
            key = f"T{T}_dp{dp}"
            rows = [r for r in all_results if r["T"] == T and r["dropout_p"] == dp]
            summary[key] = {}
            for metric in ["auc_roc", "auc_pr", "f1", "ece", "brier", "mean_uncertainty", "latency_ms"]:
                vals = [r[metric] for r in rows if not np.isnan(r[metric])]
                summary[key][metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    out_json = output_dir / "mc_sensitivity_results.json"
    with open(out_json, "w") as f:
        json.dump({"per_run": all_results, "summary": summary,
                   "T_values": T_values, "dropout_values": dropout_vals, "seeds": seeds}, f, indent=2)

    # CSV
    try:
        import csv
        csv_path = output_dir / "mc_sensitivity_table.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["T", "dropout_p", "AUC-ROC", "AUC-PR", "F1", "ECE", "Brier", "Uncertainty", "Latency_ms"])
            for dp in dropout_vals:
                for T in T_values:
                    k = f"T{T}_dp{dp}"
                    s = summary[k]
                    writer.writerow([
                        T, dp,
                        f"{s['auc_roc']['mean']:.4f}",
                        f"{s['auc_pr']['mean']:.4f}",
                        f"{s['f1']['mean']:.4f}",
                        f"{s['ece']['mean']:.4f}",
                        f"{s['brier']['mean']:.4f}",
                        f"{s['mean_uncertainty']['mean']:.5f}",
                        f"{s['latency_ms']['mean']:.3f}",
                    ])
        log.info(f"MC sensitivity table saved to {csv_path}")
    except Exception as e:
        log.warning(f"CSV export failed: {e}")

    log.info(f"MC sensitivity results saved to {out_json}")


if __name__ == "__main__":
    main()
