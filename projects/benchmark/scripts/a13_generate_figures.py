#!/usr/bin/env python3
"""A13 Generate Publication Figures (Fail-Closed).
Fulfills:
- A13-G6: Fail-closed 14x7 support matrix and seed-paired delta graphics.
  - Figure 2: Verifies unique seeds {42, 43, 44, 45, 46} from approved_run_registry.csv.
              No green default!
  - Figure 3: Explicit inner merge on (dataset, seed), calculating paired delta and SE.
              Exports publication/a13/fig3_paired_delta_data.csv.
  - Figure 1: DLG architecture and Gram decoder diagram.
Outputs to:
  - projects/benchmark/paper/a13/figures/
  - projects/benchmark/paper/a13/mdpi/figures/
"""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Set clean publication styling
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 12.5,
    'figure.autolayout': True,
})

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'
RUN_REG = BENCH / 'evidence/astra_revision/approved_run_registry.csv'
UNSUPPORTED_CSV = BENCH / 'paper/data/astra_unsupported_cells.csv'

FIG_DIR = BENCH / 'paper/a13/figures'
MDPI_FIG_DIR = BENCH / 'paper/a13/mdpi/figures'
A13_PUB = BENCH / 'publication/a13'

FIG_DIR.mkdir(parents=True, exist_ok=True)
MDPI_FIG_DIR.mkdir(parents=True, exist_ok=True)
A13_PUB.mkdir(parents=True, exist_ok=True)

run_df = pd.read_csv(RUN_REG)
unsupported_df = pd.read_csv(UNSUPPORTED_CSV)

datasets_ordered = [
    # Native Financial
    ('Elliptic', 'Native Financial'),
    ('DGraphFin', 'Native Financial'),
    ('BitcoinOTC', 'Injected Synthetic'),
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

models = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']

# -----------------------------------------------------------------------------
# 1. Figure 3: Paired Conditional Delta (Aug AP - Base AP across 13 + LANL)
# -----------------------------------------------------------------------------
print("Generating Figure 3 (Paired Conditional Delta)...")
fig3_rows = []
ds_names = []
categories = []
paired_delta_means = []
paired_delta_ses = []

EXPECTED_SEEDS = {42, 43, 44, 45, 46}

for d, cat in datasets_ordered:
    base_sub = run_df[(run_df['dataset'] == d) & (run_df['model'] == 'DLG-Base')].copy()
    aug_sub = run_df[(run_df['dataset'] == d) & (run_df['model'] == 'DLG-Aug')].copy()
    
    # Assert exactly 5 unique expected seeds
    base_seeds = set(base_sub['seed'].tolist())
    aug_seeds = set(aug_sub['seed'].tolist())
    assert base_seeds == EXPECTED_SEEDS, f"Dataset {d} DLG-Base missing seeds: {EXPECTED_SEEDS - base_seeds}"
    assert aug_seeds == EXPECTED_SEEDS, f"Dataset {d} DLG-Aug missing seeds: {EXPECTED_SEEDS - aug_seeds}"
    
    # Explicit inner join on (dataset, seed)
    merged = pd.merge(base_sub, aug_sub, on=['dataset', 'seed'], suffixes=('_base', '_aug'))
    assert len(merged) == 5, f"Merged length for {d} is {len(merged)}, expected 5"
    
    merged['delta_ap'] = merged['pr_auc_aug'] - merged['pr_auc_base']
    
    for _, mr in merged.iterrows():
        fig3_rows.append({
            'dataset': d,
            'category': cat,
            'seed': mr['seed'],
            'base_ap': mr['pr_auc_base'],
            'aug_ap': mr['pr_auc_aug'],
            'delta_ap': mr['delta_ap']
        })
        
    mean_d = float(merged['delta_ap'].mean())
    se_d = float(merged['delta_ap'].std(ddof=1) / np.sqrt(5))
    
    ds_names.append(d)
    categories.append(cat)
    paired_delta_means.append(mean_d)
    paired_delta_ses.append(se_d)

# Export fig3_paired_delta_data.csv
fig3_csv_path = A13_PUB / 'fig3_paired_delta_data.csv'
with open(fig3_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['dataset', 'category', 'seed', 'base_ap', 'aug_ap', 'delta_ap'])
    writer.writeheader()
    writer.writerows(fig3_rows)
print(f"Exported {fig3_csv_path} with {len(fig3_rows)} seed-level paired records.")

fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
y_pos = np.arange(len(ds_names))
colors = []
for cat, diff in zip(categories, paired_delta_means):
    if cat == 'External Cyber':
        colors.append('#8c564b' if diff >= 0 else '#d62728')
    elif diff >= 0:
        colors.append('#2ca02c') # Green for positive gain
    else:
        colors.append('#d62728') # Red for negative delta

bars = ax.barh(y_pos, paired_delta_means, xerr=paired_delta_ses, align='center', color=colors,
               alpha=0.85, edgecolor='black', linewidth=0.8, capsize=3)

ax.axvline(0, color='black', linewidth=1.0, linestyle='--')
ax.set_yticks(y_pos)
ax.set_yticklabels(ds_names, fontweight='semibold')
ax.invert_yaxis()
ax.set_xlabel(r'Average Precision Delta ($\Delta\mathrm{AP} = \mathrm{AP}_{\mathrm{Aug}} - \mathrm{AP}_{\mathrm{Base}}$, mean $\pm$ seed-paired SE, $n=5$)', fontweight='bold')
ax.set_title(r'DLG Augmentation Utility: Seed-Paired $\Delta\mathrm{AP}$ across 13 Primary + 1 Cyber Graph', pad=12, fontweight='bold')

for i, (bar, mean_v, se_v) in enumerate(zip(bars, paired_delta_means, paired_delta_ses)):
    x_offset = mean_v + (se_v + 0.003 if mean_v >= 0 else -(se_v + 0.003))
    ha = 'left' if mean_v >= 0 else 'right'
    sign = '+' if mean_v >= 0 else ''
    ax.text(x_offset, bar.get_y() + bar.get_height()/2.0, f"{sign}{mean_v:.4f}",
            va='center', ha=ha, fontsize=8.5, fontweight='bold',
            color='#1a531b' if mean_v >= 0 else '#8b0000')

ax.set_xlim(-0.09, 0.055)
ax.grid(axis='x', linestyle=':', alpha=0.6)

c_patches = [
    patches.Patch(facecolor='#2ca02c', edgecolor='black', label=r'Positive Utility ($\Delta\mathrm{AP} > 0$)'),
    patches.Patch(facecolor='#d62728', edgecolor='black', label=r'Negative Degradation ($\Delta\mathrm{AP} < 0$)'),
    patches.Patch(facecolor='#8c564b', edgecolor='black', label=r'External Cyber Baseline (LANL)')
]
ax.legend(handles=c_patches, loc='lower right', frameon=True, fontsize=8.5)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_conditional_delta.pdf')
fig.savefig(FIG_DIR / 'fig_conditional_delta.png')
fig.savefig(MDPI_FIG_DIR / 'fig_conditional_delta.pdf')
fig.savefig(MDPI_FIG_DIR / 'fig_conditional_delta.png')
plt.close(fig)
print('Generated fig_conditional_delta.pdf')

# -----------------------------------------------------------------------------
# 2. Figure 2: Fail-Closed Support Matrix
# -----------------------------------------------------------------------------
print("\nGenerating Figure 2 (Fail-Closed Support Matrix)...")

# Parse unsupported CSV failure provenance
unsupported_map = {}
for _, r in unsupported_df.iterrows():
    ds_k = r['dataset']
    m_k = r['model']
    status = str(r['support_status'])
    reason = str(r['failure_reason']) if pd.notna(r['failure_reason']) else ''
    
    if 'OOM' in status or 'oom' in reason:
        unsupported_map[(ds_k, m_k)] = 'OOM'
    elif 'ERROR' in status or 'error' in reason:
        unsupported_map[(ds_k, m_k)] = 'ERR'
    else:
        unsupported_map[(ds_k, m_k)] = 'NA'

# Status coding:
# 0: Fully supported primary (Green, 5/5)
# 1: External supported LANL (Blue, 5/5)
# 2: Out of Memory (OOM, Red)
# 3: Execution Error (Err, Orange)
# 4: Prohibitive Dense Scaling / Unscored (N/A, Gray)

all_ds = [d for d, _ in datasets_ordered]
status_matrix = np.zeros((len(all_ds), len(models)))
text_matrix = []

primary_supported_count = 0
primary_unsupported_count = 0
external_supported_count = 0

for i, d in enumerate(all_ds):
    row_text = []
    for j, m in enumerate(models):
        sub = run_df[(run_df['dataset'] == d) & (run_df['model'] == m)]
        unique_seeds = set(sub['seed'].tolist())
        
        if unique_seeds == EXPECTED_SEEDS:
            # Verified 5/5 unique seeds exist in approved registry
            if d == 'LANL-RedTeam':
                status_matrix[i, j] = 1
                row_text.append('5/5')
                external_supported_count += 1
            else:
                status_matrix[i, j] = 0
                row_text.append('5/5')
                primary_supported_count += 1
        else:
            # Must be verified in unsupported_map; FAIL-CLOSED if not found!
            if (d, m) not in unsupported_map:
                raise ValueError(f"FAIL-CLOSED: Cell ({d}, {m}) has seeds {unique_seeds} and is NOT documented in {UNSUPPORTED_CSV}!")
            
            code = unsupported_map[(d, m)]
            primary_unsupported_count += 1
            if code == 'OOM':
                status_matrix[i, j] = 2
                row_text.append('OOM')
            elif code == 'ERR':
                status_matrix[i, j] = 3
                row_text.append('Err')
            else:
                status_matrix[i, j] = 4
                row_text.append('N/A')
                
    text_matrix.append(row_text)

# Assertions for 84/91 primary support and 7 LANL external support
assert primary_supported_count == 84, f"Expected 84 primary supported cells, got {primary_supported_count}"
assert primary_unsupported_count == 7, f"Expected 7 primary unsupported cells, got {primary_unsupported_count}"
assert external_supported_count == 7, f"Expected 7 external supported cells, got {external_supported_count}"
print(f"Verified fail-closed support matrix: {primary_supported_count}/91 primary supported, {primary_unsupported_count} unsupported, {external_supported_count} external.")

fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)

cmap = plt.matplotlib.colors.ListedColormap(['#4daf4a', '#377eb8', '#e41a1c', '#ff7f00', '#999999'])
bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

im = ax.imshow(status_matrix, cmap=cmap, norm=norm, aspect='auto')

ax.set_xticks(np.arange(len(models)))
ax.set_yticks(np.arange(len(all_ds)))
ax.set_xticklabels(models, rotation=35, ha='right', fontweight='semibold')
ax.set_yticklabels(all_ds, fontweight='semibold')

for i in range(len(all_ds)):
    for j in range(len(models)):
        val = status_matrix[i, j]
        txt = text_matrix[i][j]
        c = "white" if val in (1, 2, 4) else "black"
        ax.text(j, i, txt, ha='center', va='center', color=c, fontsize=8.5, fontweight='bold')

ax.set_title('Execution Support & Failure Provenance Matrix (84/91 Primary Supported)', pad=12, fontweight='bold')

c_patches = [
    patches.Patch(facecolor='#4daf4a', edgecolor='black', label='SUPPORTED_EXACT (5/5 Seeds)'),
    patches.Patch(facecolor='#377eb8', edgecolor='black', label='EXTERNAL SUPPORT (LANL, 5/5)'),
    patches.Patch(facecolor='#e41a1c', edgecolor='black', label='OOM (Hardware Exhaustion)'),
    patches.Patch(facecolor='#ff7f00', edgecolor='black', label='Execution Error (Numerical Determinant)'),
    patches.Patch(facecolor='#999999', edgecolor='black', label='N/A (Prohibitive Dense Memory Scaling)')
]
ax.legend(handles=c_patches, bbox_to_anchor=(0.5, -0.19), loc='upper center', ncol=3, frameon=True, fontsize=7.5)

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_support_matrix.pdf')
fig.savefig(FIG_DIR / 'fig_support_matrix.png')
fig.savefig(MDPI_FIG_DIR / 'fig_support_matrix.pdf')
fig.savefig(MDPI_FIG_DIR / 'fig_support_matrix.png')
plt.close(fig)
print('Generated fig_support_matrix.pdf')

# -----------------------------------------------------------------------------
# 3. Figure 1: Architecture Diagram
# -----------------------------------------------------------------------------
print("\nGenerating Figure 1 (DLG Architecture Diagram)...")
fig, ax = plt.subplots(figsize=(9.2, 4.4), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis('off')

box_kw = dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#334e68', linewidth=1.5)
l1_kw = dict(boxstyle='round,pad=0.5', facecolor='#e6fffa', edgecolor='#234e52', linewidth=1.5)
l2_kw = dict(boxstyle='round,pad=0.5', facecolor='#ebf8ff', edgecolor='#2b6cb0', linewidth=1.5)
dec_kw = dict(boxstyle='round,pad=0.5', facecolor='#fffaf0', edgecolor='#9c4221', linewidth=1.5)

# Pipeline blocks
ax.text(12, 38, "Input Graph\n$G = (V, E, X)$", ha='center', va='center', bbox=box_kw, fontsize=9.5, fontweight='bold')

# Branch 1: Level 1 Local Ego-net
ax.text(38, 40, "Level 1: Local Ego-Net Encoder\n$h_v^{(1)} = \\mathrm{GNN}_{\\mathrm{local}}(G[N_k(v)])$", ha='center', va='center', bbox=l1_kw, fontsize=8.5)
ax.text(38, 30, "[Frozen Pretrained Representation]\n$\\mathbf{Z}_{\\mathrm{local}} \\in \\mathbb{R}^{N \\times d_1}$", ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#e2e8f0', edgecolor='#4a5568', linestyle='--'), fontsize=8)

# Branch 2: Level 2 Global Interaction
ax.text(38, 14, "Level 2: Global Relational Encoder\n$h_v^{(2)} = \\mathrm{GCN}_{\\mathrm{global}}(A, [X \\parallel Z_{\\mathrm{local}}])$", ha='center', va='center', bbox=l2_kw, fontsize=8.5)

# Fusion and Closed-Form Exact Gram Decoder
ax.text(72, 27, "Exact Sparse Gram Decoder\n$\\mathcal{L}_{\\mathrm{struct}} = \\frac{1}{N^2}\\left[\\|A\\|_F^2 - 2\\mathrm{Tr}(Z^T A Z) + \\|Z^T Z\\|_F^2\\right]$\n(Memory: $O(Nd + |E|d + d^2)$ vs. dense $O(N^2)$)", ha='center', va='center', bbox=dec_kw, fontsize=8)

# Output Anomaly Score
ax.text(93, 27, "Anomaly\nScore\n$s(v)$", ha='center', va='center', bbox=dict(boxstyle='circle,pad=0.4', facecolor='#fed7d7', edgecolor='#9b2c2c', linewidth=1.5), fontsize=9, fontweight='bold')

# Connecting Arrows
arrow_kw = dict(arrowstyle='->', lw=1.5, color='#2d3748')
ax.annotate('', xy=(24, 40), xytext=(20, 38), arrowprops=arrow_kw)
ax.annotate('', xy=(38, 34), xytext=(38, 37), arrowprops=arrow_kw)
ax.annotate('', xy=(24, 14), xytext=(20, 38), arrowprops=arrow_kw)
ax.annotate('', xy=(38, 18), xytext=(38, 27), arrowprops=arrow_kw)
ax.annotate('', xy=(56, 30), xytext=(52, 30), arrowprops=arrow_kw)
ax.annotate('', xy=(56, 24), xytext=(52, 14), arrowprops=arrow_kw)
ax.annotate('', xy=(88, 27), xytext=(86, 27), arrowprops=arrow_kw)

ax.set_title("Decoupled Local-to-Global (DLG) Architecture & Exact Sparse Gram Decoder", pad=8, fontweight='bold')

plt.tight_layout()
fig.savefig(FIG_DIR / 'fig_dlg_architecture.pdf')
fig.savefig(FIG_DIR / 'fig_dlg_architecture.png')
fig.savefig(MDPI_FIG_DIR / 'fig_dlg_architecture.pdf')
fig.savefig(MDPI_FIG_DIR / 'fig_dlg_architecture.png')
plt.close(fig)
print('Generated fig_dlg_architecture.pdf')

print("\nSUCCESS: All A13 publication figures generated with fail-closed assertions!")
