#!/usr/bin/env python3
"""
A18 Gate G3: Code-Faithful DLG Architecture Figure Generator
Renders fig_dlg_architecture.pdf and fig_dlg_architecture.png
depicting the verified two-stage GCN pipeline:
  - Level 1: 2-layer Local GCN feature autoencoder (finite receptive field, frozen Z_local)
  - Feature Augmentation: Concatenation [X || Z_local]
  - Level 2: Global GCN Encoder (graph-wide relational propagation)
  - Attribute & Structure Decoders (Gram Z Z^T) & Per-node Anomaly Scoring
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

WORKSPACE_ROOT = Path("/mnt/d/_Work/goat_bank/dlg_gnn")
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", Path(__file__).resolve().parent.parent.parent.parent))

TARGET_DIRS = [
    WORKSPACE_ROOT / "projects/benchmark/paper/a16/figures",
    WORKSPACE_ROOT / "projects/benchmark/paper/a16/mdpi/figures",
    WORKSPACE_ROOT / "projects/benchmark/submission/preprints/figures",
    WORKSPACE_ROOT / "projects/benchmark/submission/applsci/figures",
    WORKSPACE_ROOT / "projects/benchmark/publication/a18",
]

for d in TARGET_DIRS:
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 9.0
plt.rcParams['figure.dpi'] = 300


def render_fig_dlg_architecture():
    print("Rendering code-faithful Figure 1: DLG Architecture...")
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
    ax.text(12, 18, "Cryptocurrency &\nBenchmark Networks\n($N \\leq 4.6\\times 10^3$ in crypto;\n$N > 10^5$ with batching)", ha='center', va='center', style='italic', color='#4b5563', fontsize=8)

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
        print(f"Saved figure to {d}")

    plt.close(fig)
    print("Figure 1 generation completed successfully.")


if __name__ == "__main__":
    render_fig_dlg_architecture()
