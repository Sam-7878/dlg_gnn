"""
Generate 7 publication-ready figures for Round 3:
1. figures/real_main_comparison.png
2. figures/real_ablation.png
3. figures/real_mc_sensitivity.png
4. figures/real_calibration.png
5. figures/real_privacy_utility.png
6. figures/real_uncertainty_subgroup.png
7. figures/real_robustness.png
"""

import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("generate_figures")

ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def plot_main_comparison():
    csv_path = RESULTS_DIR / "real_main_results.csv"
    if not csv_path.exists():
        log.warning("real_main_results.csv not found")
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    methods = [r["method"].replace("_", " ") for r in rows]
    auc_prs = [float(r["mean_auc_pr"]) for r in rows]
    std_prs = [float(r.get("std_auc_pr", 0.0)) for r in rows]
    auc_rocs = [float(r["mean_auc_roc"]) for r in rows]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, auc_prs, width, yerr=std_prs, capsize=4, label='AUC-PR', color='#2b5c8f', alpha=0.85)
    bars2 = ax.bar(x + width/2, auc_rocs, width, label='AUC-ROC', color='#d95f02', alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Real Chronological Evaluation: Main Method Comparison (GoG-MicroRAG)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_main_comparison.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def plot_ablation():
    csv_path = RESULTS_DIR / "real_ablation_results.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "real_main_results.csv"
    if not csv_path.exists():
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    methods = [r["method"].replace("_", " ") for r in rows]
    auc_prs = [float(r["mean_auc_pr"]) for r in rows]
    f1s = [float(r["mean_f1"]) for r in rows]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, auc_prs, width, label='AUC-PR', color='#1b9e77', alpha=0.85)
    ax.bar(x + width/2, f1s, width, label='F1-Score', color='#7570b3', alpha=0.85)

    ax.set_ylabel('Score')
    ax.set_title('Ablation Study on Real Streaming Fraud Graph')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=35, ha='right')
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_ablation.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def plot_mc_sensitivity():
    csv_path = RESULTS_DIR / "real_mc_sensitivity.csv"
    if not csv_path.exists():
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # Group by T
    t_groups = {}
    for r in rows:
        t = int(r["T"])
        t_groups.setdefault(t, []).append(r)

    t_vals = sorted(t_groups.keys())
    mean_prs = [np.mean([float(x["auc_pr"]) for x in t_groups[t]]) for t in t_vals]
    std_prs = [np.std([float(x["auc_pr"]) for x in t_groups[t]]) for t in t_vals]
    latencies = [np.mean([float(x["latency_ms"]) for x in t_groups[t]]) for t in t_vals]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    color = '#1f77b4'
    ax1.set_xlabel('MC Dropout Passes (T)')
    ax1.set_ylabel('AUC-PR', color=color)
    ax1.errorbar(t_vals, mean_prs, yerr=std_prs, marker='o', color=color, capsize=4, label='AUC-PR')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = '#e377c2'
    ax2.set_ylabel('Latency (ms)', color=color)
    ax2.plot(t_vals, latencies, marker='s', linestyle='--', color=color, label='Latency')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('MC-Dropout Sensitivity & Latency Tradeoff (Real GNN)')
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_mc_sensitivity.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def plot_calibration():
    csv_path = RESULTS_DIR / "real_calibration.csv"
    if not csv_path.exists():
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    methods = set(r["method"] for r in rows)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7)

    for m in sorted(methods):
        m_rows = [r for r in rows if r["method"] == m]
        confs = [float(r["mean_conf"]) for r in m_rows]
        accs = [float(r["mean_acc"]) for r in m_rows]
        ax.plot(confs, accs, marker='o', label=m.replace("_", " "))

    ax.set_xlabel('Mean Predicted Probability (Confidence)')
    ax.set_ylabel('Observed Fraction of Positives (Accuracy)')
    ax.set_title('Calibration Curves (Reliability Diagram)')
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_calibration.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def plot_privacy_utility():
    csv_path = RESULTS_DIR / "real_privacy_utility.csv"
    if not csv_path.exists():
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    reps = set(r["representation"] for r in rows)
    fig, ax = plt.subplots(figsize=(8, 5))

    for rep in sorted(reps):
        r_rows = [r for r in rows if r["representation"] == rep]
        sigmas = [float(r["noise_sigma"]) for r in r_rows]
        bal_accs = [float(r["attack_balanced_accuracy"]) for r in r_rows]
        maj_baseline = float(r_rows[0]["majority_baseline"]) if r_rows else 0.5
        ax.plot(sigmas, bal_accs, marker='o', label=f'{rep.replace("_", " ")} (Attack BalAcc)')

    ax.axhline(y=0.5, color='gray', linestyle=':', label='Random Guess (0.5)')
    ax.set_xlabel('Differential Noise Level (σ)')
    ax.set_ylabel('Inference Attack Balanced Accuracy')
    ax.set_title('Privacy-Utility Tradeoff Under Perturbation')
    ax.legend()
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_privacy_utility.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def plot_uncertainty_subgroups():
    pred_files = list((RESULTS_DIR / "real_raw_predictions").glob("*_T10_preds.csv"))
    if not pred_files:
        return

    with open(pred_files[0]) as f:
        rows = list(csv.DictReader(f))

    vars_arr = np.array([float(r["variance"]) for r in rows])
    labels = np.array([int(r["label"]) for r in rows])

    # Low vs High uncertainty groups
    med_var = np.median(vars_arr)
    low_u = labels[vars_arr <= med_var]
    high_u = labels[vars_arr > med_var]

    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Low Uncertainty\n(≤ Median)', 'High Uncertainty\n(> Median)']
    fraud_rates = [np.mean(low_u) * 100, np.mean(high_u) * 100]
    counts = [len(low_u), len(high_u)]

    bars = ax.bar(categories, fraud_rates, color=['#2ca02c', '#d62728'], alpha=0.85, width=0.5)
    for bar, count in zip(bars, counts):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f'{yval:.1f}%\n(N={count})', ha='center', va='bottom')

    ax.set_ylabel('Actual Fraud Rate (%)')
    ax.set_title('Actual Fraud Prevalence by Model Uncertainty Subgroup')
    ax.set_ylim(0, max(fraud_rates) * 1.4)
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_uncertainty_subgroup.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def plot_robustness():
    csv_path = RESULTS_DIR / "real_robustness.csv"
    if not csv_path.exists():
        return

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    conditions = set(r["condition"] for r in rows)
    methods = set(r["method"] for r in rows)

    fig, axes = plt.subplots(1, len(conditions), figsize=(6 * len(conditions), 5), sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, sorted(conditions)):
        c_rows = [r for r in rows if r["condition"] == cond]
        for m in sorted(methods):
            m_rows = [r for r in c_rows if r["method"] == m]
            rates = [float(r["rate"]) for r in m_rows]
            auc_prs = [float(r["auc_pr"]) for r in m_rows]
            # sort by rate
            order = np.argsort(rates)
            ax.plot(np.array(rates)[order], np.array(auc_prs)[order], marker='o', label=m.replace("_", " "))

        ax.set_xlabel('Perturbation Rate')
        ax.set_title(f'Condition: {cond.replace("_", " ")}')
        ax.legend()

    axes[0].set_ylabel('AUC-PR')
    plt.suptitle('Robustness Analysis Under Perturbed Contexts (Real GNN)')
    plt.tight_layout()

    out_path = FIGURES_DIR / "real_robustness.png"
    plt.savefig(out_path)
    plt.close()
    log.info(f"Saved {out_path}")


def main():
    plot_main_comparison()
    plot_ablation()
    plot_mc_sensitivity()
    plot_calibration()
    plot_privacy_utility()
    plot_uncertainty_subgroups()
    plot_robustness()
    log.info("All figures generated successfully.")


if __name__ == "__main__":
    main()
