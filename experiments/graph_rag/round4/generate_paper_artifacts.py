"""Generate paper-facing artifacts only after the v2 gate passes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.round4.artifact_paths import FIGURES_DIR, RESULTS_DIR, TABLES_DIR, ensure_dirs
from experiments.round4.paper_ready_gate_v2 import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-ready", action="store_true", required=True)
    args = parser.parse_args(); ensure_dirs()
    gate = evaluate()
    if not gate["paper_ready"]:
        raise RuntimeError(f"paper artifacts refused; failed checks: {gate['failed_checks']}")
    main_results = pd.read_csv(RESULTS_DIR / "main_results.csv")
    latency = pd.read_csv(RESULTS_DIR / "e2e_latency.csv")
    main_results.to_latex(TABLES_DIR / "sci_main_results.tex", index=False, float_format="%.4f",
                          caption="Timestamp-grounded SCI main-track results (five seeds).")
    latency.to_latex(TABLES_DIR / "sci_main_latency.tex", index=False, float_format="%.3f",
                     caption="Real-checkpoint context-free end-to-end latency.")

    plt.figure(figsize=(6.4, 4.0))
    plt.bar(main_results.method, main_results.mean_auc_pr,
            yerr=main_results.std_auc_pr, capsize=5, color=["#4c78a8", "#f58518"])
    plt.ylabel("Test AUC-PR")
    plt.title("SCI Main Track (mean ± SD, five seeds)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sci_main_auc_pr.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(latency["T"], latency.mean_ms, marker="o", label="mean")
    plt.plot(latency["T"], latency.p95_ms, marker="s", label="p95")
    plt.xlabel("MC passes (T)"); plt.ylabel("Latency (ms/event)")
    plt.title("Real-checkpoint inference latency"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sci_main_latency.png", dpi=300)
    plt.close()

    plt.figure(figsize=(5.5, 5.0))
    for passes, label, color in ((1, "GNN Only", "#4c78a8"), (10, "MC-GNN (T=10)", "#f58518")):
        frames = [pd.read_csv(path) for path in sorted((RESULTS_DIR / "raw_predictions").glob(f"seed*_T{passes}.csv"))]
        labels = np.concatenate([frame.label.to_numpy(int) for frame in frames])
        probabilities = np.concatenate([frame.p_mean.to_numpy(float) for frame in frames])
        observed, predicted = calibration_curve(labels, probabilities, n_bins=10, strategy="quantile")
        plt.plot(predicted, observed, marker="o", label=label, color=color)
    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="ideal")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed fraud frequency")
    plt.title("SCI main-track calibration"); plt.legend(); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sci_main_calibration.png", dpi=300)
    plt.close()

    sensitivity = pd.read_csv(RESULTS_DIR / "mc_sensitivity.csv")
    grouped = sensitivity.groupby("T").auc_pr.agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(6.4, 4.0))
    plt.errorbar(grouped["T"], grouped["mean"], yerr=grouped["std"], marker="o", capsize=4)
    plt.xlabel("MC passes (T)"); plt.ylabel("Test AUC-PR")
    plt.title("MC-dropout sensitivity (mean ± SD, five seeds)"); plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sci_main_mc_sensitivity.png", dpi=300)
    plt.close()
    (TABLES_DIR / "artifact_manifest.json").write_text(json.dumps({
        "gate_version": gate["gate_version"], "paper_ready": True,
        "source_track": "SCI Main Track", "controlled_context_included": False,
        "tables": ["sci_main_results.tex", "sci_main_latency.tex"],
        "figures": ["sci_main_auc_pr.png", "sci_main_latency.png",
                    "sci_main_calibration.png", "sci_main_mc_sensitivity.png"],
    }, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
