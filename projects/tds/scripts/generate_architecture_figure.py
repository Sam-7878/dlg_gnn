#!/usr/bin/env python3
"""Generate publication-quality architecture and methodology overview figure for TDS manuscript.
Clean, perfectly row/column-aligned grid layout with non-overlapping boxes and neat arrows.
Outputs:
  - fig_tds_architecture.pdf
  - fig_tds_architecture.png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("Starting clean grid-aligned architecture figure generation...", flush=True)

# Set figure size and resolution
fig, ax = plt.subplots(figsize=(11.8, 5.6), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

# Style definitions (Preserving user-favored pastel color palette)
box_styles = {
    'data':   dict(facecolor='#eef2f7', edgecolor='#243b53', linewidth=1.3),
    'filter': dict(facecolor='#e0f2fe', edgecolor='#0369a1', linewidth=1.3),
    'graph':  dict(facecolor='#ecfdf5', edgecolor='#047857', linewidth=1.3),
    'tab':    dict(facecolor='#fef3c7', edgecolor='#b45309', linewidth=1.3),
    'gnn0':   dict(facecolor='#f3e8ff', edgecolor='#6b21a8', linewidth=1.3),
    'size':   dict(facecolor='#fae8ff', edgecolor='#a21caf', linewidth=1.3),
    'base':   dict(facecolor='#fff7ed', edgecolor='#c2410c', linewidth=1.3),
    'cal':    dict(facecolor='#f1f5f9', edgecolor='#334155', linewidth=1.3),
    'audit':  dict(facecolor='#f1f5f9', edgecolor='#334155', linewidth=1.3),
    'out':    dict(facecolor='#fee2e2', edgecolor='#b91c1c', linewidth=1.4),
}

# -------------------------------------------------------------
# 1. Phase Headers: 2-line layout to prevent any overlapping
# -------------------------------------------------------------
# Col 1: cx = 12.5
ax.text(12.5, 46.2, "Phase 1", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#1e3a8a')
ax.text(12.5, 44.2, "Ingestion & Observation", ha='center', va='center', fontsize=8.2, color='#1e3a8a')

# Col 2: cx = 36.5
ax.text(36.5, 46.2, "Phase 2", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#065f46')
ax.text(36.5, 44.2, "Dual Representations", ha='center', va='center', fontsize=8.2, color='#065f46')

# Col 3: cx = 61.2
ax.text(61.2, 46.2, "Phase 3", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#6b21a8')
ax.text(61.2, 44.2, "Controlled Architectures", ha='center', va='center', fontsize=8.2, color='#6b21a8')

# Col 4: cx = 86.5
ax.text(86.5, 46.2, "Phase 4", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#991b1b')
ax.text(86.5, 44.2, "Calibration & Audit", ha='center', va='center', fontsize=8.2, color='#991b1b')

# Dividing dashed vertical lines between phases
for x in [24.5, 48.5, 74.0]:
    ax.axvline(x, ymin=0.04, ymax=0.91, color='#cbd5e1', linestyle='--', linewidth=1.1)

# Helper function to draw standardized boxes
def draw_card(cx, cy, w, h, style_key, title, bullets, title_bold=True, title_color='#0f172a'):
    rect = patches.FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor=box_styles[style_key]['facecolor'],
        edgecolor=box_styles[style_key]['edgecolor'],
        linewidth=box_styles[style_key]['linewidth']
    )
    ax.add_patch(rect)
    
    # Text inside card
    card_text = f"\\textbf{{{title}}}\n" if title_bold else f"{title}\n"
    # Format bullets
    full_text = title + "\n" + "\n".join(bullets)
    ax.text(cx, cy, full_text, ha='center', va='center', fontsize=7.8, color='#1e293b', linespacing=1.28)

# -------------------------------------------------------------
# 2. Card Placement on Fixed 3-Row Grid
# Row 1: cy = 34.0, h = 9.2
# Row 2: cy = 21.5, h = 9.2
# Row 3: cy =  9.0, h = 9.2
# -------------------------------------------------------------

# Phase 1
draw_card(12.5, 34.0, 20.0, 9.2, 'data',
          "Multi-Chain Ingestion",
          ["• Ethereum, BSC, Polygon", "• 268M+ transfer rows"])

draw_card(12.5, 21.5, 20.0, 9.2, 'filter',
          "Cutoff Windowing ($t_e \\leq t_i$)",
          ["• Full-Archive: $t_i = l_i$", "• 7-Day Horizon: $t_i = f_i + 7\\mathrm{d}$"])

# Phase 2
draw_card(36.5, 34.0, 20.0, 9.2, 'graph',
          "Graph Topology $\\mathcal{E}$",
          ["• 128 recent event edges", "• In, Out, Total degrees"])

draw_card(36.5, 9.0, 20.0, 9.2, 'tab',
          "24-Dim Tabular Summary",
          ["• Explicit size: $|V|, |E|$, density", "• Degree moments & chain"])

# Phase 3
draw_card(61.2, 34.0, 22.0, 9.2, 'gnn0',
          "Standard GNN (GIN-0)",
          ["• 2-layer GIN (Mean--Max)", "• No explicit size $\\rightarrow$ AP: 0.6418"])

draw_card(61.2, 21.5, 22.0, 9.2, 'size',
          "Size-Augmented GNNs",
          ["• GIN-size (Explicit size) $\\rightarrow$ AP: 0.9445", "• GIN-sum (Sum readout) $\\rightarrow$ AP: 0.9529"])

draw_card(61.2, 9.0, 22.0, 9.2, 'base',
          "Tabular & Linear Controls",
          ["• Summary MLP $\\rightarrow$ AP: 0.9582", "• LightGBM $\\rightarrow$ AP: 0.9632", "• Retained-Node LR $\\rightarrow$ AP: 0.8943"])

# Phase 4
draw_card(86.5, 34.0, 22.0, 9.2, 'cal',
          "Probability Calibration",
          ["• Validation-only scalar $T$", "• Preserves rank; checks NLL"])

draw_card(86.5, 21.5, 22.0, 9.2, 'audit',
          "Observed-Span Diagnostics",
          ["• Full oracle $L$: AP 0.9464", "• $L \\geq 7\\mathrm{d}$ subset: AP 0.5834"])

draw_card(86.5, 9.0, 22.0, 9.2, 'out',
          "Auditable Output Gate",
          ["• Validity Decision Protocol", "• Disentangles GNN vs. Size"])

# -------------------------------------------------------------
# 3. Neat, Aligned, Orthogonal & Orderly Arrows
# -------------------------------------------------------------
arrow_kw = dict(arrowstyle='->', lw=1.3, color='#334155')

# Phase 1: Ingestion -> Cutoff Filter (Straight vertical)
ax.annotate('', xy=(12.5, 26.5), xytext=(12.5, 29.0), arrowprops=arrow_kw)

# Phase 1 Filter -> Phase 2 Dual Representations (Clean diagonal divergence)
ax.annotate('', xy=(26.2, 33.0), xytext=(22.8, 23.5), arrowprops=arrow_kw)
ax.annotate('', xy=(26.2, 10.0), xytext=(22.8, 19.5), arrowprops=arrow_kw)

# Phase 2 -> Phase 3 Model Architectures
# Graph -> GIN-0 (Perfect horizontal line at y=34.0)
ax.annotate('', xy=(49.8, 34.0), xytext=(46.8, 34.0), arrowprops=arrow_kw)

# Graph & Tabular -> Size-Augmented GNNs (Clean diagonal convergence to y=21.5)
ax.annotate('', xy=(49.8, 23.0), xytext=(46.8, 31.5), arrowprops=arrow_kw)
ax.annotate('', xy=(49.8, 20.0), xytext=(46.8, 11.5), arrowprops=arrow_kw)

# Tabular -> Tabular Controls (Perfect horizontal line at y=9.0)
ax.annotate('', xy=(49.8, 9.0), xytext=(46.8, 9.0), arrowprops=arrow_kw)

# Phase 3 -> Phase 4 Calibration & Audit (All 3 Rows: Perfect horizontal lines)
# Row 1: GIN-0 -> Calibration (y=34.0)
ax.annotate('', xy=(75.1, 34.0), xytext=(72.5, 34.0), arrowprops=arrow_kw)

# Row 2: Size-GNN -> Diagnostics (y=21.5)
ax.annotate('', xy=(75.1, 21.5), xytext=(72.5, 21.5), arrowprops=arrow_kw)

# Row 3: Tabular Controls -> Output Gate (y=9.0)
ax.annotate('', xy=(75.1, 9.0), xytext=(72.5, 9.0), arrowprops=arrow_kw)

# Phase 4 Internal Vertical flow (Straight vertical down to Output Gate)
ax.annotate('', xy=(86.5, 26.5), xytext=(86.5, 29.0), arrowprops=arrow_kw)
ax.annotate('', xy=(86.5, 14.0), xytext=(86.5, 16.5), arrowprops=arrow_kw)

# Overall Title
ax.set_title("Overview of Temporal-Admissible Contract Screening, Controlled Architectures, and Audit Framework",
             pad=10, fontsize=10.5, fontweight='bold', color='#0f172a')

# Output paths
root = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
out_paths = [
    root / 'projects/tds/results/319/figures/fig_tds_architecture.pdf',
    root / 'projects/tds/results/319/figures/fig_tds_architecture.png',
    root / 'projects/tds/paper/submission_320/preprints/figures/fig_tds_architecture.pdf',
    root / 'projects/tds/paper/submission_320/preprints/figures/fig_tds_architecture.png',
    root / 'projects/tds/paper/submission_320/mdpi_applsci_anomaly/figures/fig_tds_architecture.pdf',
    root / 'projects/tds/paper/submission_320/mdpi_applsci_anomaly/figures/fig_tds_architecture.png',
]

for p in out_paths:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == '.pdf':
        fig.savefig(p, format='pdf', bbox_inches='tight')
    else:
        fig.savefig(p, format='png', bbox_inches='tight')
    print(f'Saved: {p}', flush=True)

plt.close(fig)
print('Grid-aligned architecture figure successfully generated.', flush=True)
