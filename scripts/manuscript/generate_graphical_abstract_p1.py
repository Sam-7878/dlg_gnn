#!/usr/bin/env python3
"""
generate_graphical_abstract_p1.py

Generates a publication-grade Graphical Abstract for Preprints.org and MDPI:
Illustrates:
  10 Frozen Primary Graphs (Financial & Synthetic Anomaly Suite)
        ↓
  8 Detector Configurations (6 Baselines + DLG-Base & DLG-Aug)
        ↓
  Exact / Support-Aware Execution (Sparse Fused GCN + Exact Sparse Reconstruction)
        ↓
  Predictive Performance + Scalability + Sensitivity Controls (355 Primary + 45 Control Runs)
        ↓
  Core Finding: Competitive Fraud-Oriented Performance, but Local Augmentation is Dataset-Dependent
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "publication" / "benchmark" / "preprints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PNG = OUTPUT_DIR / "graphical_abstract.png"


def create_graphical_abstract():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Header / Title Bar
    title_box = patches.FancyBboxPatch(
        (0.5, 7.1), 13.0, 0.75, boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#1e3a8a", facecolor="#1e3a8a", linewidth=1.5
    )
    ax.add_patch(title_box)
    ax.text(
        7.0, 7.47, "A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection",
        color="white", fontsize=15, weight="bold", ha="center", va="center"
    )

    # Box 1: 10 Frozen Primary Datasets
    b1 = patches.FancyBboxPatch(
        (0.5, 4.3), 3.8, 2.4, boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#3b82f6", facecolor="#eff6ff", linewidth=1.5
    )
    ax.add_patch(b1)
    ax.text(2.4, 6.4, "1. Frozen Benchmark Datasets", color="#1e3a8a", fontsize=11, weight="bold", ha="center")
    # Base graphs: Yelp, Amazon, Flickr; Reddit, Cora, CiteSeer, PubMed
    d_text = (
        "• 10 Heterogeneous Primary Graphs\n"
        "  - Real Financial: Elliptic, DGraphFin, BitcoinOTC\n"
        "  - Synthetic Injected: Yelp-Syn, Amazon-Syn, Flickr-Syn,\n"
        "    Reddit-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn\n"
        "• External Validation: LANL-RedTeam\n"
        "• Real vs Controlled Synthetic Provenance\n"
        "• Suite maxima: 1.23M nodes and 114.9M edges"
    )
    ax.text(0.7, 5.25, d_text, color="#1e293b", fontsize=8.5, va="center", linespacing=1.4)

    # Box 2: 8 Detector Configurations
    b2 = patches.FancyBboxPatch(
        (5.1, 4.3), 3.8, 2.4, boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#6366f1", facecolor="#eef2ff", linewidth=1.5
    )
    ax.add_patch(b2)
    ax.text(7.0, 6.4, "2. Evaluated Detector Suite", color="#312e81", fontsize=11, weight="bold", ha="center")
    m_text = (
        "• 6 Established Graph Baselines:\n"
        "  - DOMINANT (Reconstruction)\n"
        "  - AnomalyDAE (Nonlinear Autoencoder)\n"
        "  - CoLA (Local Substructure Contrastive)\n"
        "  - CONAD (Augmented Contrastive)\n"
        "  - GADNR (Neighborhood Distribution)\n"
        "  - OCGNN (One-Class Objective)\n"
        "• 2 DLG Variants: DLG-Base & DLG-Aug"
    )
    ax.text(5.3, 5.25, m_text, color="#1e293b", fontsize=8.5, va="center", linespacing=1.4)

    # Box 3: Exact Semantics Execution
    b3 = patches.FancyBboxPatch(
        (9.7, 4.3), 3.8, 2.4, boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#059669", facecolor="#ecfdf5", linewidth=1.5
    )
    ax.add_patch(b3)
    ax.text(11.6, 6.4, "3. Exact Scalability Backends", color="#064e3b", fontsize=11, weight="bold", ha="center")
    e_text = (
        "• Mathematical Exactness Guarantee:\n"
        "  - Exact Sparse Linear Structure Decoder:\n"
        "    ||A_i - z_i Z^T||^2 (avoids O(N^2) dense storage)\n"
        "  - Fused Sparse Message Passing:\n"
        "    Zero COO E x H expansion\n"
        "• Strict Fail-Closed Policy:\n"
        "  - No sampling or partitioning substitutes\n"
        "  - Unsupported cells remain explicit results"
    )
    ax.text(9.9, 5.25, e_text, color="#1e293b", fontsize=8.5, va="center", linespacing=1.4)

    # Connecting Arrows (Top Row)
    ax.annotate("", xy=(5.05, 5.5), xytext=(4.35, 5.5),
                arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=2, mutation_scale=15))
    ax.annotate("", xy=(9.65, 5.5), xytext=(8.95, 5.5),
                arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=2, mutation_scale=15))

    # Downward Arrow to Evaluation Layer
    ax.annotate("", xy=(7.0, 3.85), xytext=(7.0, 4.25),
                arrowprops=dict(arrowstyle="-|>", color="#1e3a8a", lw=2.5, mutation_scale=18))

    # Middle Layer: Protocol & Controls
    p_box = patches.FancyBboxPatch(
        (0.5, 2.4), 13.0, 1.4, boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#d97706", facecolor="#fffbeb", linewidth=1.5
    )
    ax.add_patch(p_box)
    ax.text(7.0, 3.5, "4. Multi-Dimensional Experimental Protocol & Targeted Sensitivity Controls",
            color="#92400e", fontsize=11, weight="bold", ha="center")

    proto_text = (
        "• 5 Model Seeds (355 successful primary runs; 71/80 supported pairs) | Validation-Only Decision Threshold Selection (Zero Test Leakage)\n"
        "• Complete-Case Omnibus Friedman & Holm-Adjusted Pairwise Wilcoxon Tests | External Validation on 16.7k-node LANL-RedTeam Graph\n"
        "• 45 Targeted Sensitivity Controls: Zero-Information Augmentation (DLG-Aug-Zero), Feature Permutation (DLG-Aug-Permuted), Epoch Budget (DLG-Base-70)"
    )
    ax.text(7.0, 2.9, proto_text, color="#78350f", fontsize=8.8, ha="center", va="center", linespacing=1.4)

    # Downward Arrow to Findings Layer
    ax.annotate("", xy=(7.0, 1.95), xytext=(7.0, 2.35),
                arrowprops=dict(arrowstyle="-|>", color="#1e3a8a", lw=2.5, mutation_scale=18))

    # Bottom Layer: Core Findings
    f_box = patches.FancyBboxPatch(
        (0.5, 0.35), 13.0, 1.55, boxstyle="round,pad=0.1,rounding_size=0.15",
        edgecolor="#0284c7", facecolor="#f0f9ff", linewidth=1.8
    )
    ax.add_patch(f_box)
    ax.text(7.0, 1.62, "5. Authoritative Benchmark Findings: Scalable Fraud Performance vs. Dataset-Dependent Augmentation",
            color="#0369a1", fontsize=11.5, weight="bold", ha="center")

    res_l = (
        "✓ Best Average Rank in Fraud-Oriented Common Subset:\n"
        "  DLG-Aug ranks 1st in ROC-AUC (1.71), PR-AUC (1.71),\n"
        "  and Val.-F1 (1.86) across 7 fraud graphs (5 models).\n"
        "✓ Exact Protocol Execution Support:\n"
        "  DLG-Base & DLG-Aug support 10/10 primary datasets;\n"
        "  Baseline support spans 50% - 100%.\n"
        "  Overall, 71/80 model-dataset pairs are supported."
    )
    ax.text(1.2, 0.95, res_l, color="#0f172a", fontsize=9.0, va="center", linespacing=1.3)

    res_r = (
        "⚠ Local Augmentation is Conditional, Not Universal:\n"
        "  - Positive on Elliptic (+0.0350 PR-AUC) and DGraphFin (+0.0034 PR-AUC)\n"
        "  - Substantially negative on dense Reddit-Syn (-0.0656 PR-AUC)\n"
        "  - DLG-Base outperforms DLG-Aug on external LANL-RedTeam (+0.0253 PR-AUC)\n"
        "  → Results motivate adaptive, graph-dependent use of local information."
    )
    ax.text(7.0, 0.95, res_r, color="#0f172a", fontsize=9.0, va="center", linespacing=1.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Graphical abstract successfully generated: {OUTPUT_PNG}")


if __name__ == "__main__":
    create_graphical_abstract()
