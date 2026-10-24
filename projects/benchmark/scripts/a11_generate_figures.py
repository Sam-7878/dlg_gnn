#!/usr/bin/env python3
"""Phase P0-D: Generate Publication Figures (A, B, C) for Manuscript.
Generates:
  1. fig_conditional_delta.pdf / .png (Figure A: 13+LANL Aug minus Base AP Delta)
  2. fig_support_matrix.pdf / .png (Figure B: 14x7 Support and Resource Evidence)
  3. fig_dlg_architecture.pdf / .png (Figure C: Decoupled Local-Global Exact Gram Architecture)
"""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Use clean publication style
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 13,
    'figure.autolayout': True,
})

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'
E = BENCH / 'evidence/a08_data_repair'
FIG_DIR = BENCH / 'paper/a11/figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Figure A: Conditional Delta (Aug AP - Base AP across 13+LANL)
# ---------------------------------------------------------
num_reg = json.loads((E / 'number_registry.json').read_text(encoding='utf-8'))
cells = {(c['dataset'], c['model'], c['metric']): c for c in num_reg['cells']}

datasets_ordered = [
    # Native Financial
    ('Elliptic', 'Native Financial'),
    ('DGraphFin', 'Native Financial'),
    ('BitcoinOTC', 'Native Financial'),
    ('Ethereum', 'Native Financial'),
    ('BSC', 'Native Financial'),
    ('Polygon', 'Native Financial'),
    # Synthetic Injected
    ('Yelp-Syn', 'Injected Synthetic'),
    ('Amazon-Syn', 'Injected Synthetic'),
    ('Reddit-Syn', 'Injected Synthetic'),
    ('Flickr-Syn', 'Injected Synthetic'),
    ('Cora-Syn', 'Injected Synthetic'),
    ('CiteSeer-Syn', 'Injected Synthetic'),
    ('PubMed-Syn', 'Injected Synthetic'),
    # External Cybersecurity
    ('LANL-RedTeam', 'External Cyber')
]

ds_names = []
categories = []
deltas = []
base_aps = []
aug_aps = []
error_bars = []

for d, cat in datasets_ordered:
    base = cells.get((d, 'DLG-Base', 'pr_auc'))
    aug = cells.get((d, 'DLG-Aug', 'pr_auc'))
    if base and aug:
        b_val, a_val = base['mean'], aug['mean']
        diff = a_val - b_val
        # Pooled SD for descriptive error bar
        pooled_sd = np.sqrt((base['sd']**2 + aug['sd']**2) / 2)
        ds_names.append(d)
        categories.append(cat)
        deltas.append(diff)
        base_aps.append(b_val)
        aug_aps.append(a_val)
        error_bars.append(pooled_sd)

fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

y_pos = np.arange(len(ds_names))
colors = []
for cat, diff in zip(categories, deltas):
    if cat == 'External Cyber':
        colors.append('#8c564b' if diff >= 0 else '#d62728') # Brown/Red
    elif diff >= 0:
        colors.append('#2ca02c') # Green for positive gain
    else:
        colors.append('#d62728') # Red for negative delta (Reddit-Syn)

bars = ax.barh(y_pos, deltas, xerr=error_bars, align='center', color=colors, alpha=0.85,
               edgecolor='black', linewidth=0.8, capsize=3)

ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(ds_names)
ax.invert_yaxis()  # Labels read top-to-bottom
ax.set_xlabel(r'Conditional AP Delta: $\Delta\mathrm{AP} = \mathrm{AP}_{\mathrm{Aug}} - \mathrm{AP}_{\mathrm{Base}}$')
ax.set_title('DLG-Aug vs. DLG-Base Conditional AP Performance (13 Primary + LANL)', pad=12, fontweight='bold')

# Annotations for key findings
for idx, (d, diff) in enumerate(zip(ds_names, deltas)):
    offset = 0.003 if diff >= 0 else -0.003
    ha = 'left' if diff >= 0 else 'right'
    ax.text(diff + offset, idx, f"{diff:+.4f}", va='center', ha=ha, fontsize=8, fontweight='semibold')

# Add legend for provenance categories
legend_elements = [
    patches.Patch(facecolor='#2ca02c', edgecolor='black', label='Positive AP Delta (Aug > Base)'),
    patches.Patch(facecolor='#d62728', edgecolor='black', label='Negative AP Delta (Base > Aug, e.g. Reddit-Syn)'),
    patches.Patch(facecolor='#8c564b', edgecolor='black', label='External Cyber (LANL-RedTeam, Descriptive)')
]
ax.legend(handles=legend_elements, loc='lower right', frameon=True, fontsize=8.5)

ax.grid(axis='x', linestyle=':', alpha=0.6)
ax.set_xlim(-0.08, 0.06)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_conditional_delta.pdf')
fig.savefig(FIG_DIR / 'fig_conditional_delta.png')
plt.close(fig)
print('Generated fig_conditional_delta')

# ---------------------------------------------------------
# Figure B: Support & Execution Matrix Heatmap (14x7)
# ---------------------------------------------------------
models = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']
all_ds = [d for d, _ in datasets_ordered]

support_rows = list(csv.DictReader((E / 'tables/support_matrix.csv').open(encoding='utf-8')))
support_map = {(r['dataset'], r['model']): r['support_status'] for r in support_rows}

matrix_status = np.zeros((len(all_ds), len(models)))
# 0: Fully supported (Green)
# 1: Unsupported / OOM (Red)
# 2: External (Blue/Cyan)

for i, d in enumerate(all_ds):
    for j, m in enumerate(models):
        status = support_map.get((d, m), 'NOT_FULLY_SUPPORTED')
        if d == 'LANL-RedTeam':
            matrix_status[i, j] = 2 if status == 'SUPPORTED_EXACT' else 1
        else:
            matrix_status[i, j] = 0 if status == 'SUPPORTED_EXACT' else 1

fig, ax = plt.subplots(figsize=(8.0, 5.5), dpi=300)

cmap = plt.matplotlib.colors.ListedColormap(['#4daf4a', '#e41a1c', '#377eb8']) # Green, Red, Blue
bounds = [-0.5, 0.5, 1.5, 2.5]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(matrix_status, cmap=cmap, norm=norm, aspect='auto')

ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(all_ds)))
ax.set_xticklabels(models, rotation=35, ha='right', fontweight='semibold')
ax.set_yticklabels(all_ds, fontweight='semibold')

# Overlay text on cells
for i in range(len(all_ds)):
    for j in range(len(models)):
        val = matrix_status[i, j]
        txt = "5/5" if val in (0, 2) else "OOM"
        c = "white" if val in (1, 2) else "black"
        ax.text(j, i, txt, ha='center', va='center', color=c, fontsize=8.5, fontweight='bold')

ax.set_title('Execution Support & Hardware Feasibility Matrix (84/91 Primary Supported)', pad=12, fontweight='bold')

# Custom legend
c_patches = [
    patches.Patch(facecolor='#4daf4a', edgecolor='black', label='SUPPORTED_EXACT (5/5 Seeds Succeeded)'),
    patches.Patch(facecolor='#e41a1c', edgecolor='black', label='UNSUPPORTED (OOM / Hardware Bound)'),
    patches.Patch(facecolor='#377eb8', edgecolor='black', label='EXTERNAL SUPPORT (LANL 5/5, Non-Omnibus)')
]
ax.legend(handles=c_patches, bbox_to_anchor=(0.5, -0.18), loc='upper center', ncol=3, frameon=True, fontsize=8)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_support_matrix.pdf')
fig.savefig(FIG_DIR / 'fig_support_matrix.png')
plt.close(fig)
print('Generated fig_support_matrix')

# ---------------------------------------------------------
# Figure C: Decoupled Local-to-Global Architecture Pipeline
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

# Box styles
box_kw = dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#334e68', linewidth=1.5)
l1_kw = dict(boxstyle='round,pad=0.5', facecolor='#e6fffa', edgecolor='#234e52', linewidth=1.5)
l2_kw = dict(boxstyle='round,pad=0.5', facecolor='#ebf8ff', edgecolor='#2b6cb0', linewidth=1.5)
dec_kw = dict(boxstyle='round,pad=0.5', facecolor='#fffaf0', edgecolor='#9c4221', linewidth=1.5)

# Pipeline blocks
ax.text(12, 38, "Input Graph\n$G = (V, E, X)$", ha='center', va='center', bbox=box_kw, fontsize=9.5, fontweight='bold')

# Branch 1: Level 1 Local Ego-net
ax.text(38, 40, "Level 1: Local Ego-Net Encoder\n$h_v^{(1)} = \\mathrm{GNN}_{\\mathrm{local}}(G[N_k(v)])$", ha='center', va='center', bbox=l1_kw, fontsize=8.5)
ax.text(38, 30, "[Frozen Local Representation]\n$\\mathbf{Z}_{\\mathrm{local}} \\in \\mathbb{R}^{N \\times d_1}$", ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#e2e8f0', edgecolor='#4a5568', linestyle='--'), fontsize=8)

# Branch 2: Level 2 Global Interaction
ax.text(38, 14, "Level 2: Global Relational Encoder\n$h_v^{(2)} = \\mathrm{GCN}_{\\mathrm{global}}(A, [X \\parallel Z_{\\mathrm{local}}])$", ha='center', va='center', bbox=l2_kw, fontsize=8.5)

# Fusion and Closed-Form Exact Gram Decoder
ax.text(72, 27, "Exact Sparse Gram Decoder\n$\\mathcal{L}_{\\mathrm{exact}} = \\sum_{(u,v) \\in E} (A_{uv} - z_u^T z_v)^2 + \\lambda \\|Z^T Z\\|_F^2$\n(No dense $N \\times N$ matrix materialization)", ha='center', va='center', bbox=dec_kw, fontsize=8.5)

# Output Anomaly Score
ax.text(92, 27, "Anomaly\nScore\n$s(v)$", ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.4', facecolor='#fed7d7', edgecolor='#9b2c2c', linewidth=1.5), fontsize=9, fontweight='bold')

# Connecting Arrows
arrow_kw = dict(arrowstyle='->', lw=1.5, color='#2d3748')
ax.annotate('', xy=(24, 40), xytext=(20, 38), arrowprops=arrow_kw)
ax.annotate('', xy=(38, 34), xytext=(38, 37), arrowprops=arrow_kw)
ax.annotate('', xy=(24, 14), xytext=(20, 38), arrowprops=arrow_kw)
ax.annotate('', xy=(38, 18), xytext=(38, 27), arrowprops=arrow_kw)
ax.annotate('', xy=(56, 30), xytext=(52, 30), arrowprops=arrow_kw)
ax.annotate('', xy=(56, 24), xytext=(52, 14), arrowprops=arrow_kw)
ax.annotate('', xy=(88, 27), xytext=(86, 27), arrowprops=arrow_kw)

ax.set_title("Decoupled Local-to-Global (DLG) Anomaly Detection & Exact Gram Decoder", pad=8, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_dlg_architecture.pdf')
fig.savefig(FIG_DIR / 'fig_dlg_architecture.png')
plt.close(fig)
print('Generated fig_dlg_architecture')
