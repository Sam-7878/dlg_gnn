#!/usr/bin/env python3
"""
A20 Figure Generator:
Renders publication-faithful figures aligning with A20 Editorial and Evidence Gates:
  1. fig_dlg_architecture.pdf/.png (Figure 1: Architecture, removing N <= 4.6x10^3 and batching note)
  2. fig_support_matrix.pdf/.png (Figure 2: Support matrix, updating title to 'mixed environments')
  3. fig_conditional_delta.pdf/.png (Figure 3: Paired AP differences, removing 'Verified Canonical Ground Truth')
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Root discovery
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

TARGET_DIRS = [
    WORKSPACE_ROOT / "projects/benchmark/submission/preprints/figures",
    WORKSPACE_ROOT / "projects/benchmark/submission/applsci/figures",
    WORKSPACE_ROOT / "projects/benchmark/paper/a16/figures",
    WORKSPACE_ROOT / "projects/benchmark/paper/a16/mdpi/figures",
    WORKSPACE_ROOT / "projects/benchmark/publication/a20",
]

for d in TARGET_DIRS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 9.5
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9.5
plt.rcParams['figure.dpi'] = 300


def render_figure_1():
    print("Rendering code-faithful Figure 1: DLG Architecture (A20 Revision)...")
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 52)
    ax.axis('off')

    box_kw = dict(edgecolor='#1f2937', linewidth=1.4)

    # 1. Input Graph Box (Left)
    rect1 = patches.FancyBboxPatch((2, 10), 20, 32, boxstyle="round,pad=0.8", facecolor='#f3f4f6', **box_kw)
    ax.add_patch(rect1)
    ax.text(12, 38, "Input Graph $\\mathcal{G}$", ha='center', va='center', weight='bold', color='#111827', fontsize=11)
    ax.text(12, 30, "Adjacency Matrix $A$\nNode Attributes $X$\n($N$ nodes, $|\\mathcal{E}|$ edges)", ha='center', va='center', color='#374151', fontsize=9)
    # A20 correction: removed inaccurate 'N <= 4.6x10^3' and 'batching' notes
    ax.text(12, 18, "Cryptocurrency &\nBenchmark Networks\n(Transductive full-graph\nevaluation on all graphs)", ha='center', va='center', style='italic', color='#4b5563', fontsize=8)

    # Arrows from Input -> L1 Pre-training and L2 Concatenation
    ax.annotate("", xy=(26, 38), xytext=(22, 34), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))
    ax.annotate("", xy=(51, 19), xytext=(22, 20), arrowprops=dict(arrowstyle="->", lw=1.8, color="#4b5563", linestyle="--"))

    # 2. Stage 1: Level 1 Local Feature Pre-training (Top Middle)
    rect2 = patches.FancyBboxPatch((26, 26), 22, 22, boxstyle="round,pad=0.8", facecolor='#dbeafe', edgecolor='#1d4ed8', linewidth=1.4)
    ax.add_patch(rect2)
    ax.text(37, 44, "Level 1: Local Pre-training", ha='center', va='center', weight='bold', color='#1e3a8a', fontsize=10)
    ax.text(37, 36, "2-Layer Local GCN AE\nFinite 2-Hop Receptive Field\nObjective: $\\min \\|X - \\hat{X}_{\\mathrm{L1}}\\|_F^2$\n(No explicit ego-net extraction)", ha='center', va='center', color='#1e40af', fontsize=8)
    ax.text(37, 28.5, "[Detached & Frozen: $Z_{\\mathrm{local}}$]", ha='center', va='center', weight='bold', color='#1d4ed8', fontsize=8.5)

    # Arrow from L1 -> Feature Augmentation
    ax.annotate("", xy=(51, 25), xytext=(48, 27), arrowprops=dict(arrowstyle="->", lw=1.8, color="#2563eb"))

    # 3. Augmentation & Level 2 Global Encoder (Center Right)
    rect3 = patches.FancyBboxPatch((51, 8), 24, 38, boxstyle="round,pad=0.8", facecolor='#d1fae5', edgecolor='#047857', linewidth=1.4)
    ax.add_patch(rect3)
    ax.text(63, 42, "Level 2: Global Encoding", ha='center', va='center', weight='bold', color='#064e3b', fontsize=10.5)
    
    # Feature Augmentation sub-box
    rect3_sub = patches.FancyBboxPatch((53, 30), 20, 8, boxstyle="round,pad=0.4", facecolor='#a7f3d0', edgecolor='#059669', linewidth=1.0)
    ax.add_patch(rect3_sub)
    ax.text(63, 34, "Feature Augmentation\n$\\tilde{X} = [X \\parallel Z_{\\mathrm{local}}]$", ha='center', va='center', weight='bold', color='#065f46', fontsize=8.5)

    ax.annotate("", xy=(63, 27), xytext=(63, 30), arrowprops=dict(arrowstyle="->", lw=1.5, color="#059669"))

    ax.text(63, 21, "Global GCN Encoder\n$Z = \\mathrm{GCN}(\\tilde{X}, \\mathcal{E})$\nGraph-wide Relational Patterns", ha='center', va='center', color='#065f46', fontsize=8.5)
    ax.text(63, 12, "(DLG-Base: sequential L1$\\to$L2\nwith learnable fusion gate $\\alpha$)", ha='center', va='center', style='italic', color='#047857', fontsize=7.5)

    # Arrows to Decoders
    ax.annotate("", xy=(79, 36), xytext=(75, 30), arrowprops=dict(arrowstyle="->", lw=1.8, color="#b45309"))
    ax.annotate("", xy=(79, 18), xytext=(75, 18), arrowprops=dict(arrowstyle="->", lw=1.8, color="#b45309"))

    # 4. Decoders & Scoring (Right)
    rect4 = patches.FancyBboxPatch((79, 8), 24, 38, boxstyle="round,pad=0.8", facecolor='#fef3c7', edgecolor='#b45309', linewidth=1.4)
    ax.add_patch(rect4)
    ax.text(91, 42, "Dual Decoders & Scoring", ha='center', va='center', weight='bold', color='#78350f', fontsize=10.5)

    # Feature decoder
    rect4_f = patches.FancyBboxPatch((81, 29), 20, 9, boxstyle="round,pad=0.4", facecolor='#fde68a', edgecolor='#d97706', linewidth=1.0)
    ax.add_patch(rect4_f)
    ax.text(91, 33.5, "Attribute Decoder\nReconstructs Original $X$\n$\\hat{X} = \\mathrm{GCN}(Z, \\mathcal{E})$", ha='center', va='center', color='#92400e', fontsize=8)

    # Structure decoder
    rect4_s = patches.FancyBboxPatch((81, 16), 20, 9, boxstyle="round,pad=0.4", facecolor='#fde68a', edgecolor='#d97706', linewidth=1.0)
    ax.add_patch(rect4_s)
    ax.text(91, 20.5, "Structure Decoder\nUnsquashed Gram Matrix\n$\\hat{S} = Z Z^T$", ha='center', va='center', color='#92400e', fontsize=8)

    # Final scoring formula
    ax.text(91, 11, "Anomaly Score $\\mathcal{A}_i$:\n$w\\|X_i - \\hat{X}_i\\|_2 + (1-w)\\|S_i - \\hat{S}_i\\|_2$", ha='center', va='center', weight='bold', color='#78350f', fontsize=8)

    plt.tight_layout()
    for d in TARGET_DIRS:
        fig.savefig(d / "fig_dlg_architecture.pdf", bbox_inches='tight')
        fig.savefig(d / "fig_dlg_architecture.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig_dlg_architecture.pdf/.png")


def render_figure_2():
    print("Rendering Figure 2: Hardware Support Matrix (A20 Revision)...")
    reg_path = WORKSPACE_ROOT / 'projects/benchmark/publication/a16/canonical_selected_run_registry.csv'
    df = pd.read_csv(reg_path)

    datasets = [
        "Amazon-Syn", "BSC", "BitcoinOTC", "CiteSeer-Syn", "Cora-Syn",
        "DGraphFin", "Elliptic", "Ethereum", "Flickr-Syn", "Polygon",
        "PubMed-Syn", "Reddit-Syn", "Yelp-Syn", "LANL-RedTeam"
    ]
    models = ["DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]

    matrix = np.zeros((len(datasets), len(models)))
    for i, ds in enumerate(datasets):
        for j, m in enumerate(models):
            sub = df[(df['dataset'] == ds) & (df['model'] == m)]
            seeds = set(sub['seed'].tolist())
            if seeds == {42, 43, 44, 45, 46}:
                if ds == "LANL-RedTeam":
                    matrix[i, j] = 2
                else:
                    matrix[i, j] = 1
            else:
                matrix[i, j] = 0

    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    cmap = matplotlib.colors.ListedColormap(['#fee2e2', '#dcfce7', '#dbeafe'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    cax = ax.imshow(matrix, cmap=cmap, norm=norm, aspect='auto')

    ax.set_xticks(np.arange(len(models)))
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_xticklabels(models, rotation=35, ha='right', weight='bold')
    ax.set_yticklabels(datasets, weight='bold')

    for i in range(len(datasets)):
        for j in range(len(models)):
            val = matrix[i, j]
            if val == 1:
                text = "✓ 5/5"
                color = "#15803d"
            elif val == 2:
                text = "✓ Ext"
                color = "#1d4ed8"
            else:
                text = "✗ OOM" if models[j] == "AnomalyDAE" or (models[j] == "GADNR" and datasets[i] == "Flickr-Syn") else "✗ Fail"
                color = "#b91c1c"
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8.5, weight='bold')

    ax.set_xticks(np.arange(-.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(datasets), 1), minor=True)
    ax.grid(which='minor', color='#9ca3af', linestyle='-', linewidth=0.6)
    ax.tick_params(which='minor', size=0)

    # A20 correction: changed title from 'RTX 3090 24 GB' to 'mixed environments'
    ax.set_title("Hardware Support & Execution Contract Matrix (Mixed Environments)\n84/91 Primary Supported Pairs + 7 External LANL Pairs", pad=12, weight='bold')
    
    patch_sup = patches.Patch(facecolor='#dcfce7', edgecolor='#15803d', label='Supported Primary (5/5 seeds)')
    patch_ext = patches.Patch(facecolor='#dbeafe', edgecolor='#1d4ed8', label='External Cyber Network (5/5 seeds)')
    patch_uns = patches.Patch(facecolor='#fee2e2', edgecolor='#b91c1c', label='Unsupported / Memory Limit (0/5)')
    ax.legend(handles=[patch_sup, patch_ext, patch_uns], loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)

    plt.tight_layout()
    for d in TARGET_DIRS:
        fig.savefig(d / 'fig_support_matrix.pdf', bbox_inches='tight')
        fig.savefig(d / 'fig_support_matrix.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig_support_matrix.pdf/.png")


def render_figure_3():
    print("Rendering Figure 3: Conditional AP Delta Chart (A20 Revision)...")
    csv_path = WORKSPACE_ROOT / 'projects/benchmark/publication/a14/conditional_ap_14.csv'
    if not csv_path.exists():
        csv_path = WORKSPACE_ROOT / 'projects/benchmark/publication/a15/conditional_ap_14.csv'
    df = pd.read_csv(csv_path)

    df = df.sort_values(by='delta_mean', ascending=True).reset_index(drop=True)

    datasets = df['dataset'].tolist()
    deltas = df['delta_mean'].values
    ses = df['paired_se'].values

    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.12, top=0.92)

    y_pos = np.arange(len(datasets))
    colors = ['#1d4ed8' if d > 0 else '#b91c1c' for d in deltas]

    bars = ax.barh(y_pos, deltas, xerr=ses, align='center', color=colors, alpha=0.85, edgecolor='#111827', linewidth=0.8, capsize=3.5, error_kw=dict(lw=1.0, capthick=1.0, ecolor='#1f2937'))

    ax.axvline(0, color='#374151', linewidth=1.0, linestyle='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(datasets, fontsize=9.5, weight='bold')
    ax.set_xlabel(r"Seed-Paired Difference in Average Precision ($\Delta\mathrm{AP} = \mathrm{AP}_{\mathrm{Aug}} - \mathrm{AP}_{\mathrm{Base}}$)", fontsize=10, weight='bold', labelpad=8)
    # A20 correction: removed overclaiming 'Verified Canonical Ground Truth'
    ax.set_title(r"Conditional AP Gain / Degradation of DLG-Aug over DLG-Base Reference" "\n" r"(Observed Mean $\pm$ Paired SE across 5 Seeds)", fontsize=11, weight='bold', pad=12)

    min_x = np.min(deltas - ses)
    max_x = np.max(deltas + ses)
    pad_left = abs(min_x) * 0.45
    pad_right = abs(max_x) * 0.55
    ax.set_xlim(min_x - pad_left, max_x + pad_right)

    for i, (bar, d, se) in enumerate(zip(bars, deltas, ses)):
        sign = "+" if d > 0 else ""
        label_txt = f"{sign}{d:.4f} ± {se:.4f}"
        if d >= 0:
            tx = d + se + 0.002
            ha = 'left'
        else:
            tx = d - se - 0.002
            ha = 'right'
        ax.text(tx, i, label_txt, va='center', ha=ha, fontsize=8.2, weight='bold', color='#111827')

    ax.grid(axis='x', linestyle=':', alpha=0.6, color='#9ca3af')
    ax.set_axisbelow(True)

    for d in TARGET_DIRS:
        fig.savefig(d / 'fig_conditional_delta.pdf', bbox_inches='tight')
        fig.savefig(d / 'fig_conditional_delta.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  -> Saved fig_conditional_delta.pdf/.png")


def main():
    render_figure_1()
    render_figure_2()
    render_figure_3()
    print("All A20 figures generated and distributed successfully.")


if __name__ == '__main__':
    main()

