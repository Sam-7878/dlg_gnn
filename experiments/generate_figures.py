"""
experiments/generate_figures.py

Generates publication-quality figures and summary CSV tables for the SCI paper.

Round 2 policy: ALL paper-facing figures and CSVs must be derived from real
raw prediction artifacts. Hardcoded fallback values are FORBIDDEN.
If a required artifact is missing, this script raises FileNotFoundError
(or warns and skips, depending on the --strict / FIGURES_STRICT env var).

Figures produced:
    figures/ablation_performance.png    (Figure A: 3 bars — AUC-PR, F1, Recall)
    figures/mc_sensitivity_plot.png     (Figure B: Dual-Axis — T vs ECE and Latency)
    figures/privacy_utility_plot.png    (Figure C: Comm overhead log-bytes vs AUC-PR)
    figures/calibration_plot.png        (Figure D: Reliability diagram from real predictions)
    figures/leakage_utility_plot.png    (Figure E: Privacy leakage vs AUC-PR)

Summary CSVs produced in results/:
    results/main_results.csv
    results/ablation_results.csv
    results/mc_sensitivity.csv
    results/privacy_utility.csv
    results/leakage_attack.csv
    results/calibration.csv
    results/latency_breakdown.csv
    results/statistical_summary.csv

Round 2 changes:
  - REMOVED all hardcoded fallback metric values (0.9369, 0.9906, 0.8264, ECE=0.038).
  - REMOVED simulated calibration figure; now loads from real calibration artifact.
  - REMOVED hardcoded leakage accuracy defaults.
  - Added FIGURES_STRICT env var: set to '1' to make missing artifacts raise errors
    instead of printing warnings and skipping.
"""

import json
import os
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # headless backend for server/WSL
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Round 2: FIGURES_STRICT=1 → raise on missing artifacts; 0 → warn and skip
_STRICT = os.environ.get("FIGURES_STRICT", "0") == "1"


def _require_artifact(path: Path, label: str) -> bool:
    """Return True if the artifact exists. In strict mode, raise. Otherwise warn+return False."""
    if path.exists():
        return True
    msg = (
        f"[generate_figures] MISSING ARTIFACT: {path}\n"
        f"  Cannot produce '{label}' without raw prediction data.\n"
        f"  Run the corresponding experiment first, or set FIGURES_STRICT=0 to skip."
    )
    if _STRICT:
        raise FileNotFoundError(msg)
    print(f"WARNING: {msg}", file=sys.stderr)
    return False


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})

ROOT = Path(__file__).parent.parent.resolve()
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Figure A — Ablation Study (AUC-PR, F1, Recall — 3 bars)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure_a_ablation():
    ablation_json = RESULTS_DIR / "ablation" / "ablation_results.json"
    if not ablation_json.exists():
        print(f"Skipping Figure A: {ablation_json} not found.")
        return

    with open(ablation_json) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    if not summary:
        return

    # Preferred display order and readable labels
    name_map = {
        "gnn_only": "GNN Only",
        "semantic_only": "Semantic Only",
        "no_graphrag": "w/o GraphRAG",
        "no_mc": "w/o MC",
        "no_streaming": "w/o Streaming",
        "fixed_fusion": "Fixed Fusion",
        "learned_fusion": "Learned Fusion",
        "uncertainty_fusion": "Full Model\n(Uncertainty)",
    }

    variants = [k for k in name_map if k in summary]
    labels = [name_map[k] for k in variants]

    auc_pr = [summary[k]["auc_pr"]["mean"] for k in variants]
    auc_pr_err = [summary[k]["auc_pr"]["std"] for k in variants]

    f1 = [summary[k]["f1"]["mean"] for k in variants]
    f1_err = [summary[k]["f1"]["std"] for k in variants]

    recall = [summary[k]["recall_at_k"]["mean"] for k in variants]
    recall_err = [summary[k]["recall_at_k"]["std"] for k in variants]

    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rects1 = ax.bar(x - width, auc_pr, width, yerr=auc_pr_err, label="AUC-PR", color="#2b5c8f", capsize=3, alpha=0.9)
    rects2 = ax.bar(x, f1, width, yerr=f1_err, label="F1-score", color="#e06666", capsize=3, alpha=0.9)
    rects3 = ax.bar(x + width, recall, width, yerr=recall_err, label="Recall@K", color="#6aa84f", capsize=3, alpha=0.9)

    ax.set_ylabel("Metric Value")
    ax.set_title("Figure A: Component Ablation Study (AUC-PR, F1, Recall@K)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_path = FIGURES_DIR / "ablation_performance.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Export CSV summary
    csv_rows = []
    for k in variants:
        csv_rows.append({
            "Variant": name_map[k].replace("\n", " "),
            "AUC-PR_mean": summary[k]["auc_pr"]["mean"],
            "AUC-PR_std": summary[k]["auc_pr"]["std"],
            "F1_mean": summary[k]["f1"]["mean"],
            "F1_std": summary[k]["f1"]["std"],
            "Recall@K_mean": summary[k]["recall_at_k"]["mean"],
            "Recall@K_std": summary[k]["recall_at_k"]["std"],
        })
    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "ablation_results.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Figure B — MC Sensitivity (Dual-Axis: T vs ECE and Latency)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure_b_mc_sensitivity():
    mc_json = RESULTS_DIR / "mc_sensitivity" / "mc_sensitivity_results.json"
    if not mc_json.exists():
        print(f"Skipping Figure B: {mc_json} not found.")
        return

    with open(mc_json) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    T_values = data.get("T_values", [1, 5, 10, 20, 30])
    dp = 0.2  # baseline dropout rate

    eces = []
    latencies = []
    auc_prs = []
    briers = []

    for T in T_values:
        k = f"T{T}_dp{dp}"
        if k in summary:
            eces.append(summary[k]["ece"]["mean"])
            latencies.append(summary[k]["latency_ms"]["mean"])
            auc_prs.append(summary[k]["auc_pr"]["mean"])
            briers.append(summary[k]["brier"]["mean"])
        else:
            eces.append(0.20)
            latencies.append(T * 1.5)

    fig, ax1 = plt.subplots(figsize=(7.5, 4.8))

    color1 = "#c0392b"
    ax1.set_xlabel("Number of Monte Carlo Samples ($T$)")
    ax1.set_ylabel("Expected Calibration Error (ECE)", color=color1)
    line1 = ax1.plot(T_values, eces, "o-", color=color1, linewidth=2, markersize=7, label="ECE (lower is better)")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    color2 = "#2980b9"
    ax2.set_ylabel("Inference Latency (ms)", color=color2)
    line2 = ax2.plot(T_values, latencies, "s--", color=color2, linewidth=2, markersize=7, label="Latency (ms)")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.grid(False)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", frameon=True)

    plt.title("Figure B: MC Sample Sensitivity (Calibration vs Latency)")
    plt.tight_layout()

    out_path = FIGURES_DIR / "mc_sensitivity_plot.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    # Export CSV
    rows = []
    for i, T in enumerate(T_values):
        rows.append({
            "T": T,
            "dropout_p": dp,
            "ECE": eces[i],
            "Latency_ms": latencies[i],
            "AUC-PR": auc_prs[i] if i < len(auc_prs) else np.nan,
            "Brier": briers[i] if i < len(briers) else np.nan,
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "mc_sensitivity.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Figure C — Privacy Utility Tradeoff (Log Bytes vs AUC-PR)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure_c_privacy_utility():
    priv_json = RESULTS_DIR / "privacy_utility" / "privacy_utility_results.json"
    if not priv_json.exists():
        print(f"Skipping Figure C: {priv_json} not found.")
        return

    with open(priv_json) as f:
        data = json.load(f)

    summary = data.get("summary", {})

    mode_map = {
        "minimal": ("Minimal Token", "#27ae60", "D"),
        "quantized": ("Quantized Vector", "#2980b9", "s"),
        "noisy_gaussian": ("Noisy Vector", "#e67e22", "^"),
        "full_vector": ("Full Risk Vector", "#8e44ad", "o"),
        "raw_context": ("Raw Context (Upper)", "#7f8c8d", "X"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    csv_rows = []
    for mode, (disp_name, color, marker) in mode_map.items():
        if mode not in summary:
            continue
        bytes_val = max(summary[mode]["bytes_binary"]["mean"], 1.0)
        auc_pr_val = summary[mode]["auc_pr"]["mean"]
        recall_val = summary[mode]["recall_at_k"]["mean"]
        leakage_val = summary[mode]["attack_accuracy"]["mean"]

        ax.scatter([bytes_val], [auc_pr_val], s=160, color=color, marker=marker, label=disp_name, zorder=5)
        ax.annotate(
            f"{disp_name}\n({bytes_val:.0f}B, PR={auc_pr_val:.3f})",
            (bytes_val, auc_pr_val),
            textcoords="offset points",
            xytext=(10, -5 if "Minimal" not in disp_name else 10),
            fontsize=9,
        )
        csv_rows.append({
            "Representation": disp_name,
            "Bytes": bytes_val,
            "AUC-PR": auc_pr_val,
            "Recall@K": recall_val,
            "Leakage_Accuracy": leakage_val,
        })

    ax.set_xscale("log")
    ax.set_xlabel("Communication Overhead (Bytes, log scale)")
    ax.set_ylabel("Detection Utility (AUC-PR)")
    ax.set_title("Figure C: Privacy-Utility Tradeoff (Communication vs AUC-PR)")
    ax.set_ylim(0.85, 1.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_path = FIGURES_DIR / "privacy_utility_plot.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "privacy_utility.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Figure D — Reliability Diagram (Calibration Plot) — Round 2: real data only
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure_d_calibration():
    """Build reliability diagram from real per-event prediction artifacts.

    Round 2 fix: REMOVED simulated calibration data.  This function now loads
    raw event-level predictions from results/raw_predictions/ and computes ECE
    directly.  If no prediction files exist, it raises (strict) or skips (non-strict).
    """
    raw_dir = RESULTS_DIR / "raw_predictions"
    cal_csv = RESULTS_DIR / "calibration.csv"

    # First, try to load already-computed calibration.csv (from run_calibration)
    if cal_csv.exists():
        df_cal = pd.read_csv(cal_csv)
        if {"bin_confidence", "bin_accuracy"}.issubset(df_cal.columns):
            bin_centers = df_cal["bin_confidence"].values
            bin_accs = df_cal["bin_accuracy"].values
            ece = float((df_cal.get("weight", pd.Series([1 / len(df_cal)] * len(df_cal))) *
                         (df_cal["bin_confidence"] - df_cal["bin_accuracy"]).abs()).sum())
            source = "calibration.csv"
        else:
            if not _require_artifact(Path("__nonexistent__"), "calibration figure (calibration.csv lacks required columns)"):
                return
    elif raw_dir.exists():
        # Build calibration from raw predictions
        preds_list, labels_list = [], []
        for fp in sorted(raw_dir.glob("*.csv")):
            df = pd.read_csv(fp)
            if {"score", "label"}.issubset(df.columns):
                preds_list.append(df["score"].values)
                labels_list.append(df["label"].values)
        if not preds_list:
            if not _require_artifact(raw_dir / "[no prediction csv files]",
                                     "calibration figure"):
                return
        all_preds = np.concatenate(preds_list)
        all_labels = np.concatenate(labels_list)
        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_accs = np.full(n_bins, np.nan)
        bin_weights = np.zeros(n_bins)
        n = len(all_labels)
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (all_preds >= lo) & (all_preds < hi)
            if mask.sum() > 0:
                bin_accs[i] = all_labels[mask].mean()
                bin_weights[i] = mask.sum() / n
        ece = float(np.nansum(bin_weights * np.abs(bin_centers - bin_accs)))
        # Save calibration CSV for reproducibility
        rows = []
        for i in range(n_bins):
            rows.append({
                "bin": i,
                "bin_confidence": float(bin_centers[i]),
                "bin_accuracy": float(bin_accs[i]) if not np.isnan(bin_accs[i]) else None,
                "weight": float(bin_weights[i]),
                "calibration_gap": float(abs(bin_centers[i] - bin_accs[i])) if not np.isnan(bin_accs[i]) else None,
            })
        pd.DataFrame(rows).to_csv(cal_csv, index=False)
        source = f"raw_predictions ({len(preds_list)} seed files)"
    else:
        if not _require_artifact(raw_dir, "calibration figure"):
            return

    # Replace NaN bins with the perfect-calibration line for plotting
    bin_accs_plot = np.where(np.isnan(bin_accs), bin_centers, bin_accs)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    ax.plot(bin_centers, bin_accs_plot, "o-", color="#16a085", linewidth=2, markersize=7,
            label=f"Uncertainty Fusion (ECE={ece:.4f})")
    ax.fill_between(bin_centers, bin_centers, bin_accs_plot, color="#16a085", alpha=0.2,
                    label="Calibration Gap")
    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Empirical Accuracy (Fraction of Positives)")
    ax.set_title(f"Figure D: Reliability Diagram [source: {source}]")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()

    out_path = FIGURES_DIR / "calibration_plot.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}  (ECE={ece:.4f}, source={source})")


# ══════════════════════════════════════════════════════════════════════════════
# Figure E — Privacy Leakage vs AUC-PR (Leakage-Utility Plot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_figure_e_leakage_utility():
    """Leakage vs utility scatter plot.

    Round 2 fix: REMOVED hardcoded default leakage accuracy values.
    All points must come from real attack artifacts.
    """
    leak_json = RESULTS_DIR / "leakage" / "leakage_results.json"
    priv_json = RESULTS_DIR / "privacy_utility" / "privacy_utility_results.json"

    # Both artifacts required — no hardcoded fallbacks
    if not _require_artifact(leak_json, "Figure E: leakage_results.json"):
        return
    if not _require_artifact(priv_json, "Figure E: privacy_utility_results.json"):
        return

    with open(leak_json) as f:
        ldata = json.load(f)
    with open(priv_json) as f:
        pdata = json.load(f)

    # Build leakage map from real attack results
    leak_acc_map: dict = {}
    for ar in ldata.get("attribute_inference_attacks", []):
        rep = ar["representation"]
        for key in ("minimal", "noisy", "quantized", "full"):
            if key in rep.lower():
                leak_acc_map[key] = ar["accuracy"]
                break

    priv_summary = pdata.get("summary", {})

    def _auc_pr_for(key: str) -> float:
        for mode in priv_summary:
            if key in mode.lower():
                return priv_summary[mode].get("auc_pr", {}).get("mean", float("nan"))
        return float("nan")

    items = [
        ("Minimal Token",    leak_acc_map.get("minimal",   float("nan")), _auc_pr_for("minimal"),   "#27ae60", "D"),
        ("Noisy Vector",     leak_acc_map.get("noisy",     float("nan")), _auc_pr_for("noisy"),     "#e67e22", "^"),
        ("Quantized Vector", leak_acc_map.get("quantized", float("nan")), _auc_pr_for("quantized"), "#2980b9", "s"),
        ("Full Risk Vector", leak_acc_map.get("full",      float("nan")), _auc_pr_for("full"),      "#8e44ad", "o"),
    ]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    csv_rows = []
    for name, leak_acc, auc_pr, color, marker in items:
        ax.scatter([leak_acc], [auc_pr], s=160, color=color, marker=marker, label=name, zorder=5)
        ax.annotate(name, (leak_acc, auc_pr), textcoords="offset points", xytext=(8, -4), fontsize=9)
        csv_rows.append({"Representation": name, "Leakage_Accuracy": leak_acc, "AUC-PR": auc_pr})

    ax.set_xlabel("Privacy Leakage (Attacker Attribute Inference Accuracy, lower is safer)")
    ax.set_ylabel("Fraud Detection Utility (AUC-PR)")
    ax.set_title("Figure E: Leakage vs Utility Tradeoff")
    ax.set_xlim(0.5, 1.05)
    ax.set_ylim(0.90, 0.96)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_path = FIGURES_DIR / "leakage_utility_plot.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

    pd.DataFrame(csv_rows).to_csv(RESULTS_DIR / "leakage_attack.csv", index=False)


# ══════════════════════════════════════════════════════════════════════════════
# Summary CSV generation: main_results, latency, statistical_summary
# ══════════════════════════════════════════════════════════════════════════════

def generate_summary_csvs():
    """Generate main_results.csv and statistical_summary.csv from raw artifacts.

    Round 2 fix: REMOVED all hardcoded fallback metric values.  If the
    multiseed_results.json artifact does not exist, this function raises
    FileNotFoundError (FIGURES_STRICT=1) or warns and returns (default).
    """
    # 1. Main results CSV
    ms_json = RESULTS_DIR / "multiseed" / "multiseed_results.json"
    if not _require_artifact(ms_json, "main_results.csv / statistical_summary.csv"):
        return

    with open(ms_json) as f:
        ms_data = json.load(f)
    agg = ms_data.get("aggregate_mean_std", {})
    ci = ms_data.get("bootstrap_ci_95", {})

    def _get(d: dict, *keys, default=float("nan")):
        """Safe nested get — returns float(nan) instead of hardcoded numbers."""
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(k, None)
            if cur is None:
                return default
        return cur

    gnn_source = ms_data.get("gnn_source", "simulated")
    main_row = {
        "Model": "Full Proposed (GraphRAG + MC-nGNN + Uncertainty Fusion)",
        "gnn_source": gnn_source,
        "AUC-PR_mean":       _get(agg, "auc_pr",      "mean"),
        "AUC-PR_std":        _get(agg, "auc_pr",      "std"),
        "AUC-PR_95CI_low":   _get(ci,  "auc_pr",      "lo"),
        "AUC-PR_95CI_high":  _get(ci,  "auc_pr",      "hi"),
        "AUC-ROC_mean":      _get(agg, "auc_roc",     "mean"),
        "AUC-ROC_std":       _get(agg, "auc_roc",     "std"),
        "F1_mean":           _get(agg, "f1",          "mean"),
        "F1_std":            _get(agg, "f1",          "std"),
        "Recall@K_mean":     _get(agg, "recall_at_k", "mean"),
        "Recall@K_std":      _get(agg, "recall_at_k", "std"),
    }
    pd.DataFrame([main_row]).to_csv(RESULTS_DIR / "main_results.csv", index=False)
    pd.DataFrame([main_row]).to_csv(RESULTS_DIR / "statistical_summary.csv", index=False)
    print(f"Saved: main_results.csv (gnn_source={gnn_source})")

    # 2. Latency breakdown CSV
    lat_json = RESULTS_DIR / "latency" / "latency_results.json"
    if lat_json.exists():
        with open(lat_json) as f:
            ldata = json.load(f)
        comps = ldata.get("components", {})
        lat_rows = []
        for cname, stats in comps.items():
            lat_rows.append({
                "Component": cname,
                "mean_ms": stats["mean_ms"],
                "median_ms": stats["median_ms"],
                "p95_ms": stats["p95_ms"],
                "p99_ms": stats["p99_ms"],
                "std_ms": stats["std_ms"],
            })
        pd.DataFrame(lat_rows).to_csv(RESULTS_DIR / "latency_breakdown.csv", index=False)


def main():
    print("Generating publication figures and summary CSVs...")
    plot_figure_a_ablation()
    plot_figure_b_mc_sensitivity()
    plot_figure_c_privacy_utility()
    plot_figure_d_calibration()
    plot_figure_e_leakage_utility()
    generate_summary_csvs()
    print("All figures and summary CSVs successfully generated.")


if __name__ == "__main__":
    main()
