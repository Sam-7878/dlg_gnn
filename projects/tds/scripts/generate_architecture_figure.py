#!/usr/bin/env python3
"""Generate publication-quality architecture and methodology overview figure for TDS manuscript.
Outputs:
  - fig_tds_architecture.pdf
  - fig_tds_architecture.png
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("Starting architecture figure generation...", flush=True)

fig, ax = plt.subplots(figsize=(11.0, 5.6), dpi=300)
ax.set_xlim(0, 105)
ax.set_ylim(0, 52)
ax.axis('off')

# Style definitions
box_data = dict(boxstyle='round,pad=0.45', facecolor='#eef2f7', edgecolor='#243b53', linewidth=1.4)
box_filter = dict(boxstyle='round,pad=0.45', facecolor='#e0f2fe', edgecolor='#0369a1', linewidth=1.4)
box_graph = dict(boxstyle='round,pad=0.45', facecolor='#ecfdf5', edgecolor='#047857', linewidth=1.4)
box_tab = dict(boxstyle='round,pad=0.45', facecolor='#fef3c7', edgecolor='#b45309', linewidth=1.4)
box_gnn = dict(boxstyle='round,pad=0.45', facecolor='#f3e8ff', edgecolor='#6b21a8', linewidth=1.4)
box_size = dict(boxstyle='round,pad=0.45', facecolor='#fae8ff', edgecolor='#a21caf', linewidth=1.4)
box_base = dict(boxstyle='round,pad=0.45', facecolor='#fff7ed', edgecolor='#c2410c', linewidth=1.4)
box_eval = dict(boxstyle='round,pad=0.45', facecolor='#f1f5f9', edgecolor='#334155', linewidth=1.4)
box_out = dict(boxstyle='round,pad=0.45', facecolor='#fee2e2', edgecolor='#b91c1c', linewidth=1.4)

# Section Headers (Top stage indicators)
ax.text(12, 50, "Phase 1: Ingestion & Observation", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1e3a8a')
ax.text(37, 50, "Phase 2: Dual Representations", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#065f46')
ax.text(65, 50, "Phase 3: Controlled Architectures", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#6b21a8')
ax.text(93, 50, "Phase 4: Audit & Calibration", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#991b1b')

# Dividing dashed lines between phases
for x in [24.5, 49.5, 80.5]:
    ax.axvline(x, ymin=0.03, ymax=0.92, color='#cbd5e1', linestyle='--', linewidth=1.2)

# ==========================================
# PHASE 1: Data Ingestion & Time Filtering
# ==========================================
p1_text = (
    "Multi-Chain Token Streams\n"
    "• Ethereum (14,464 contracts)\n"
    "• BSC (7,499 contracts)\n"
    "• Polygon (2,353 contracts)\n"
    "268,282,924 Source Transfers"
)
ax.text(12, 38.5, p1_text, ha='center', va='center', bbox=box_data, fontsize=8.0)

p1_split = (
    "Cutoff Filter: $t_e \\leq t_i$\n"
    "[Cohort A: Full-Archive]\n"
    "Cutoff $t_i = l_i$ (Last Transfer)\n"
    "Test: 2.93% Positive (n=3,648)\n"
    "-------------------------\n"
    "[Cohort B: 7-Day Horizon]\n"
    "Cutoff $t_i = f_i + 7d$\n"
    "Test: 56.16% Positive (n=3,627)"
)
ax.text(12, 17.5, p1_split, ha='center', va='center', bbox=box_filter, fontsize=7.8)

# ==========================================
# PHASE 2: Dual-Track Representation
# ==========================================
p2_graph = (
    "Graph Representation\n"
    "• Topology E: Last 128 transfers\n"
    "• Node Features: x_v in R^3\n"
    "  [log(1+d_in), log(1+d_out), log(1+d_tot)]\n"
    "• Preserves exact event order"
)
ax.text(37, 36.5, p2_graph, ha='center', va='center', bbox=box_graph, fontsize=7.8)

p2_tab = (
    "Tabular Degree Summary\n"
    "• 24 Summary Features:\n"
    "  - Size: |V|, |E|, Directed Density\n"
    "  - In/Out/Total Deg (mean, med, max)\n"
    "  - History In/Out/Total (mean, med, max)\n"
    "  - One-hot Chain Indicator"
)
ax.text(37, 16.5, p2_tab, ha='center', va='center', bbox=box_tab, fontsize=7.8)

# ==========================================
# PHASE 3: Controlled Architectures
# ==========================================
p3_gin0 = (
    "Standard GNN (GIN-0)\n"
    "• 2-Layer GIN (width 48)\n"
    "• Mean--Max Readout\n"
    "• Head: Chain one-hot only\n"
    "--> AP: 0.6418 (lacks count scale)"
)
ax.text(65, 41.0, p3_gin0, ha='center', va='center', bbox=box_gnn, fontsize=7.7)

p3_ginsize = (
    "Size-Augmented GNNs\n"
    "• GIN-size: Explicit Size in Head\n"
    "  [h_G || log(1+|V|) || log(1+|E|)]\n"
    "  --> AP: 0.9445 (+0.3027)\n"
    "• GIN-sum: Sum-pooling readout\n"
    "  --> AP: 0.9529 (+0.3111)"
)
ax.text(65, 27.5, p3_ginsize, ha='center', va='center', bbox=box_size, fontsize=7.7)

p3_base = (
    "Tabular & Linear Controls\n"
    "• Summary MLP (24 feats) --> AP: 0.9582\n"
    "• LightGBM (Val-selected) --> AP: 0.9632\n"
    "• Retained-Node LR --> AP: 0.8943\n"
    "• Isolation Forest --> AP: 0.6068"
)
ax.text(65, 12.0, p3_base, ha='center', va='center', bbox=box_base, fontsize=7.7)

# ==========================================
# PHASE 4: Audit, Calibration & Decision
# ==========================================
p4_cal = (
    "Probability Calibration\n"
    "• Validation-only scalar T\n"
    "  sigma(logit(p) / T)\n"
    "• Preserves rank ordering\n"
    "• Held-out loss can worsen\n"
    "  (LightGBM NLL 0.304 --> 0.305)"
)
ax.text(93, 39.5, p4_cal, ha='center', va='center', bbox=box_eval, fontsize=7.7)

p4_audit = (
    "Diagnostic & Audit Rigor\n"
    "• Observed span L_i = l_i - f_i\n"
    "  Full oracle: AP 0.9464 (future)\n"
    "• L >= 7d continuation subset:\n"
    "  19.65% positive, MLP AP 0.5834\n"
    "• Chain Alert Budgets (top 100):\n"
    "  Eth: 1,800 / BSC: 226 / Poly: 11"
)
ax.text(93, 23.5, p4_audit, ha='center', va='center', bbox=box_eval, fontsize=7.7)

p4_out = (
    "Auditable Output Gate\n"
    "Decision Checklist & Protocol\n"
    "Strict separation: architecture\n"
    "vs. explicit size representation"
)
ax.text(93, 8.5, p4_out, ha='center', va='center', bbox=box_out, fontsize=8.0, fontweight='bold')

# Connecting Arrows
arrow_kw = dict(arrowstyle='->', lw=1.3, color='#334155')

# P1 -> P2
ax.annotate('', xy=(27, 36.5), xytext=(21.5, 38.5), arrowprops=arrow_kw)
ax.annotate('', xy=(27, 16.5), xytext=(21.5, 17.5), arrowprops=arrow_kw)
ax.annotate('', xy=(12, 26.5), xytext=(12, 30.5), arrowprops=arrow_kw)

# P2 -> P3
ax.annotate('', xy=(52, 41.0), xytext=(47.5, 37.0), arrowprops=arrow_kw)
ax.annotate('', xy=(52, 28.5), xytext=(47.5, 35.0), arrowprops=arrow_kw)
ax.annotate('', xy=(52, 26.5), xytext=(47.5, 18.0), arrowprops=arrow_kw)
ax.annotate('', xy=(52, 13.0), xytext=(47.5, 16.0), arrowprops=arrow_kw)

# P3 -> P4
ax.annotate('', xy=(82.5, 39.5), xytext=(78.5, 39.5), arrowprops=arrow_kw)
ax.annotate('', xy=(82.5, 27.5), xytext=(78.5, 27.5), arrowprops=arrow_kw)
ax.annotate('', xy=(82.5, 15.0), xytext=(78.5, 14.0), arrowprops=arrow_kw)
ax.annotate('', xy=(93, 14.0), xytext=(93, 17.0), arrowprops=arrow_kw)

# Overall Title
ax.set_title("Overview of Temporal-Admissible Contract Screening, Controlled Architectures, and Audit Framework",
             pad=12, fontsize=11, fontweight='bold', color='#0f172a')

# Output paths
root = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
out_paths = [
    root / 'projects/tds/results/319/figures/fig_tds_architecture.pdf',
    root / 'projects/tds/results/319/figures/fig_tds_architecture.png',
    root / 'projects/tds/paper/submission_319/preprints/figures/fig_tds_architecture.pdf',
    root / 'projects/tds/paper/submission_319/preprints/figures/fig_tds_architecture.png',
    root / 'projects/tds/paper/submission_319/mdpi_applsci_anomaly/figures/fig_tds_architecture.pdf',
    root / 'projects/tds/paper/submission_319/mdpi_applsci_anomaly/figures/fig_tds_architecture.png',
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == '.pdf':
        fig.savefig(p, format='pdf', bbox_inches='tight')
    else:
        fig.savefig(p, format='png', bbox_inches='tight')
    print(f'Saved: {p}', flush=True)

plt.close(fig)
print('All architecture figures successfully generated.', flush=True)
