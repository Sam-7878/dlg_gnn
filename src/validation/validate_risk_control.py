#!/usr/bin/env python3
"""Temporal validation-constrained routing audit for conditional direct-exit FNR."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta

from gog_fraud.pipelines.run_round4_experiments import SciV2Records, _dlg_scores, _fit_dlg, _normalize
from validation.sci_v3_final_common import SEEDS, atomic_csv, atomic_json, sha256_json


ALPHAS = (0.01, 0.025, 0.05, 0.10)


def binomial_upper(false_negatives: int, n_direct_fraud: int, delta: float) -> float:
    if n_direct_fraud == 0:
        return 1.0
    if false_negatives == n_direct_fraud:
        return 1.0
    return float(beta.ppf(1.0 - delta, false_negatives + 1, n_direct_fraud - false_negatives))


def routing_counts(
    labels: np.ndarray,
    scores: np.ndarray,
    uncertainty: np.ndarray,
    tau_b: float,
    tau_f: float,
    tau_u: float,
) -> dict[str, float | int | np.ndarray]:
    certain = uncertainty <= tau_u
    benign_direct = certain & (scores <= tau_b)
    fraud_direct = certain & (scores >= tau_f)
    direct = benign_direct | fraud_direct
    direct_fraud = direct & (labels == 1)
    false_negative = benign_direct & (labels == 1)
    n_direct_fraud = int(direct_fraud.sum())
    misses = int(false_negative.sum())
    return {
        "direct": direct,
        "n_direct": int(direct.sum()),
        "n_direct_fraud": n_direct_fraud,
        "false_negatives": misses,
        "risk": float(misses / n_direct_fraud) if n_direct_fraud else None,
        "coverage": float(direct.mean()),
        "deep_route_rate": float((~direct).mean()),
    }


def calibrate(
    labels: np.ndarray,
    scores: np.ndarray,
    uncertainty: np.ndarray,
    alpha: float,
    delta: float,
) -> dict[str, float | int | np.ndarray]:
    tau_u = float(np.quantile(uncertainty, 0.90))
    score_quantiles = np.unique(np.quantile(scores, np.linspace(0.02, 0.98, 49)))
    candidates: list[dict] = []
    for tau_b in score_quantiles:
        for tau_f in score_quantiles[score_quantiles >= tau_b]:
            counts = routing_counts(labels, scores, uncertainty, float(tau_b), float(tau_f), tau_u)
            upper = binomial_upper(int(counts["false_negatives"]), int(counts["n_direct_fraud"]), delta)
            if upper <= alpha:
                candidates.append({**counts, "tau_b": float(tau_b), "tau_f": float(tau_f), "tau_u": tau_u, "upper": upper})
    if not candidates:
        return {
            **routing_counts(labels, scores, uncertainty, -math.inf, math.inf, -math.inf),
            "tau_b": -math.inf,
            "tau_f": math.inf,
            "tau_u": -math.inf,
            "upper": 1.0,
        }
    return max(candidates, key=lambda row: (float(row["coverage"]), int(row["n_direct_fraud"])))


def run(dataset_root: Path, output_dir: Path, epochs: int, delta: float) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw_predictions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = SciV2Records(dataset_root)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []

    for chain in ("ethereum", "bsc", "polygon", "pooled"):
        train_ids, valid_ids, test_ids = (dataset.ids(chain, group) for group in ("train", "validation", "test"))
        train_x, train_y = dataset.arrays(train_ids)
        valid_x, valid_y = dataset.arrays(valid_ids)
        test_x, test_y = dataset.arrays(test_ids)
        train_x, valid_x, test_x = _normalize(train_x, valid_x, test_x)
        for seed in SEEDS:
            model, _ = _fit_dlg(train_x, train_y, variant="DLG-Full-Fusion", seed=seed, epochs=epochs, device=device)
            val_score, val_var, _ = _dlg_scores(model, train_x, valid_x, device, mc=8)
            test_score, test_var, _ = _dlg_scores(model, train_x, test_x, device, mc=8)
            raw_path = raw_dir / f"{chain}__seed{seed}.parquet"
            pd.concat(
                [
                    pd.DataFrame({"sample_id": valid_ids, "split": "validation", "label": valid_y, "score": val_score, "uncertainty": val_var}),
                    pd.DataFrame({"sample_id": test_ids, "split": "test", "label": test_y, "score": test_score, "uncertainty": test_var}),
                ],
                ignore_index=True,
            ).to_parquet(raw_path, index=False)
            for alpha in ALPHAS:
                calibrated = calibrate(valid_y, val_score, val_var, alpha, delta)
                tested = routing_counts(
                    test_y,
                    test_score,
                    test_var,
                    float(calibrated["tau_b"]),
                    float(calibrated["tau_f"]),
                    float(calibrated["tau_u"]),
                )
                test_upper = binomial_upper(int(tested["false_negatives"]), int(tested["n_direct_fraud"]), delta)
                threshold_payload = {
                    "chain": chain,
                    "seed": seed,
                    "alpha": alpha,
                    "delta": delta,
                    "tau_b": calibrated["tau_b"],
                    "tau_f": calibrated["tau_f"],
                    "tau_u": calibrated["tau_u"],
                    "selection_dataset": "validation",
                }
                rows.append(
                    {
                        "chain": chain,
                        "seed": seed,
                        "risk_definition": "P(predicted_benign | fraud, direct_exit)",
                        "target_alpha": alpha,
                        "delta": delta,
                        "selection_dataset": "validation",
                        "selection_objective": "max_direct_coverage_subject_to_CP_upper_le_alpha",
                        "threshold_artifact_hash": sha256_json(threshold_payload),
                        "tau_b": calibrated["tau_b"],
                        "tau_f": calibrated["tau_f"],
                        "tau_u": calibrated["tau_u"],
                        "validation_observed_risk": calibrated["risk"],
                        "validation_upper_bound": calibrated["upper"],
                        "observed_test_risk": tested["risk"],
                        "test_upper_bound_diagnostic": test_upper,
                        "coverage": tested["coverage"],
                        "direct_exit_rate": tested["coverage"],
                        "deep_route_rate": tested["deep_route_rate"],
                        "n_direct": tested["n_direct"],
                        "n_direct_fraud": tested["n_direct_fraud"],
                        "false_negatives": tested["false_negatives"],
                        "validation_to_test_prevalence_shift": float(test_y.mean() - valid_y.mean()),
                        "validation_to_test_score_mean_shift": float(test_score.mean() - val_score.mean()),
                        "validation_to_test_uncertainty_mean_shift": float(test_var.mean() - val_var.mean()),
                        "theorem_assumptions_verified": False,
                        "claim_class": "temporally_evaluated_validation_constrained_risk_control",
                        "raw_prediction_artifact": str(raw_path),
                    }
                )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    frame = pd.DataFrame(rows)
    atomic_csv(output_dir / "risk_control_alpha_sweep.csv", frame)
    atomic_json(
        output_dir / "risk_control_manifest.json",
        {
            "risk_random_variable": "indicator(predicted_benign) conditional on fraud and direct_exit",
            "calibration_population": "frozen temporal validation split",
            "test_population": "later frozen temporal test split",
            "finite_sample_bound": "one-sided Clopper-Pearson binomial upper bound",
            "assumptions": ["i.i.d./exchangeable calibration observations", "stationarity from calibration to deployment"],
            "assumptions_verified_for_temporal_test": False,
            "allowed_claim": "validation-constrained empirical risk control",
            "records": len(frame),
        },
    )
    return frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/d/_Work/_data/GoG_sci_v2")
    parser.add_argument("--output-dir", default="results/sci_v3_final/risk_control")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--delta", type=float, default=0.05)
    args = parser.parse_args()
    frame = run(Path(args.dataset_root), Path(args.output_dir), args.epochs, args.delta)
    print(json.dumps({"records": len(frame), "claim": "validation-constrained empirical risk control"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
