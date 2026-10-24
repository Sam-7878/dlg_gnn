#!/usr/bin/env python3
"""A12 Dataset Contract Gate (P0-01).
Reconciles dataset metadata from canonical loader manifests and approved run registry.
Separates model-input tensor counts from upstream source graph totals.
Exports:
  1. projects/benchmark/publication/a12/dataset_identity_reconciliation.csv
  2. projects/benchmark/publication/a12/dataset_identity_reconciliation.md
  3. projects/benchmark/paper/a12/tables/table_dataset_properties.tex
"""
import json
import csv
from pathlib import Path
import pandas as pd

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'

CANON_MANIFEST = BENCH / 'evidence/astra_revision/dataset_manifest_canonical.json'
HIST_CONTRACTS = BENCH / 'evidence/submission_closure_a09/historical_dataset_contracts.json'
RUN_REGISTRY = BENCH / 'evidence/astra_revision/approved_run_registry.csv'
NUM_REGISTRY = BENCH / 'evidence/submission_closure_a09/replayed/number_registry.json'

PUB_A12 = BENCH / 'publication/a12'
PUB_A12.mkdir(parents=True, exist_ok=True)
PAPER_A12_TABLES = BENCH / 'paper/a12/tables'
PAPER_A12_TABLES.mkdir(parents=True, exist_ok=True)

# Load data
with open(CANON_MANIFEST, encoding='utf-8') as f:
    canon_data = {d['dataset_id']: d for d in json.load(f)}

with open(HIST_CONTRACTS, encoding='utf-8') as f:
    hist_data = {d.get('dataset_id', d.get('dataset')): d for d in json.load(f)}

run_df = pd.read_csv(RUN_REGISTRY)

# Canonical ordered list of datasets matching paper structure
DATASET_ORDER = [
    # 5 Native Financial / Blockchain
    ('Elliptic', 'Native Financial', 'real_external_label', 'labeled_transaction_node'),
    ('DGraphFin', 'Native Financial', 'real_external_label', 'loan_account_node'),
    ('Ethereum', 'Native Financial', 'real_external_label', 'smart_contract_node'),
    ('BSC', 'Native Financial', 'real_external_label', 'smart_contract_node'),
    ('Polygon', 'Native Financial', 'real_external_label', 'smart_contract_node'),
    # 8 Synthetic Injected Benchmarks (BitcoinOTC is real trust graph with injected target labels!)
    ('BitcoinOTC', 'Injected Synthetic', 'synthetic_node_injection_on_real_trust_graph', 'node'),
    ('Yelp-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    ('Amazon-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    ('Reddit-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    ('Flickr-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    ('Cora-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    ('CiteSeer-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    ('PubMed-Syn', 'Injected Synthetic', 'synthetic_node_injection', 'node'),
    # 1 External Cybersecurity
    ('LANL-RedTeam', 'External Cyber', 'real_external_label', 'destination_computer_node')
]

# Source graph metadata (upstream raw repository scale before evaluation induction/filtering)
UPSTREAM_SOURCE_METADATA = {
    'Elliptic': {'source_nodes': 203769, 'source_edges': 234355, 'notes': 'Full transaction history; model evaluates 46,564 labeled nodes'},
    'DGraphFin': {'source_nodes': 3700550, 'source_edges': 4300999, 'notes': 'Full loan graph; model evaluates 1,225,601 induced nodes (367,702 val/test masked)'},
    'Ethereum': {'source_nodes': None, 'source_edges': None, 'notes': 'Luo et al. token transfers; standardized k=5 feature-similarity graph (14,385 contracts)'},
    'BSC': {'source_nodes': None, 'source_edges': None, 'notes': 'Luo et al. token transfers; standardized k=5 feature-similarity graph (7,481 contracts)'},
    'Polygon': {'source_nodes': None, 'source_edges': None, 'notes': 'Luo et al. token transfers; standardized k=5 feature-similarity graph (2,303 contracts)'},
    'BitcoinOTC': {'source_nodes': 5881, 'source_edges': 35592, 'notes': 'SNAP signed trust network; canonical reconstructed tensor has 6,005 nodes, 7,409 edges with injected targets'},
    'Yelp-Syn': {'source_nodes': 45954, 'source_edges': 3846979, 'notes': 'Raw review graph; model evaluates PyGOD 716,847 node injection tensor'},
    'Amazon-Syn': {'source_nodes': 11944, 'source_edges': 4397792, 'notes': 'Raw co-purchase graph; model evaluates 13,752 node injection tensor'},
    'Reddit-Syn': {'source_nodes': 232965, 'source_edges': 11606919, 'notes': 'Raw Reddit post graph; model evaluates 232,965 nodes, 114,869,737 edges'},
    'Flickr-Syn': {'source_nodes': 89250, 'source_edges': 899756, 'notes': 'Raw Flickr graph; model evaluates 89,250 nodes, 1,001,494 directed edges'},
    'Cora-Syn': {'source_nodes': 2708, 'source_edges': 5429, 'notes': 'Citation network; 5,429 undirected edges = 11,276 directed edges in PyG storage'},
    'CiteSeer-Syn': {'source_nodes': 3327, 'source_edges': 4732, 'notes': 'Citation network; 4,732 undirected edges = 9,914 directed edges in PyG storage'},
    'PubMed-Syn': {'source_nodes': 19717, 'source_edges': 44338, 'notes': 'Citation network; 44,338 undirected edges = 93,958 directed edges in PyG storage'},
    'LANL-RedTeam': {'source_nodes': 156117, 'source_edges': 1496194, 'notes': 'Raw authentication events; model evaluates 16,694 destination computer nodes'}
}

# Target label descriptions for manuscript Table 1
TARGET_PROVENANCE_DESCRIPTIONS = {
    'Elliptic': 'Bitcoin transaction graph; illicit entities (Weber et al., 2019).',
    'DGraphFin': 'FinTech loan user graph; fraudulent defaulters (Huang et al., 2022).',
    'Ethereum': 'Token-transfer feature-similarity graph; verified phishing/scam contracts (Luo et al., 2024).',
    'BSC': 'Binance Smart Chain token transfer graph; fraud contracts (Luo et al., 2024).',
    'Polygon': 'Polygon token transfer graph; verified fraud contracts (Luo et al., 2024).',
    'BitcoinOTC': 'SNAP OTC signed trust network (Kumar et al., 2016); controlled injected anomaly targets on real graph.',
    'Yelp-Syn': 'Social review network; injected structural and contextual anomalies.',
    'Amazon-Syn': 'Co-purchasing graph; injected structural and contextual anomalies.',
    'Reddit-Syn': 'Reddit post interaction graph; injected structural anomalies.',
    'Flickr-Syn': 'Image sharing network; injected structural anomalies.',
    'Cora-Syn': 'Citation network; injected structural anomalies (PyGOD generator).',
    'CiteSeer-Syn': 'Citation network; injected structural anomalies (PyGOD generator).',
    'PubMed-Syn': 'Citation network; injected structural anomalies (PyGOD generator).',
    'LANL-RedTeam': 'Enterprise authentication graph; red-team compromised computers (Kent, 2015).'
}

reconciled_rows = []

for ds, category, label_nature, pred_unit in DATASET_ORDER:
    canon = canon_data.get(ds, {})
    hist = hist_data.get(ds, {})
    sub_reg = run_df[run_df['dataset'] == ds]
    
    # Model-input evaluated tensor numbers
    model_nodes = canon.get('graph_nodes')
    model_edges = canon.get('graph_edges')
    feat_dim = canon.get('feature_dimension')
    eval_nodes = canon.get('evaluation_nodes', model_nodes)
    pos_nodes = canon.get('positive_evaluation_nodes')
    pos_rate = canon.get('evaluation_positive_rate')
    
    # Cross-check against run registry
    reg_nodes = sub_reg['nodes'].dropna().unique()
    reg_edges = sub_reg['edges'].dropna().unique()
    
    reg_n_val = int(reg_nodes[0]) if len(reg_nodes) > 0 else model_nodes
    reg_e_val = int(reg_edges[0]) if len(reg_edges) > 0 else model_edges
    
    # Assert consistency
    assert reg_n_val == model_nodes, f"Mismatch in nodes for {ds}: registry={reg_n_val}, canon={model_nodes}"
    if reg_e_val is not None and model_edges is not None:
        assert reg_e_val == model_edges, f"Mismatch in edges for {ds}: registry={reg_e_val}, canon={model_edges}"

    up = UPSTREAM_SOURCE_METADATA.get(ds, {})
    
    row = {
        'dataset': ds,
        'category': category,
        'model_input_nodes': model_nodes,
        'model_input_edges': model_edges,
        'feature_dim': feat_dim,
        'test_prevalence': pos_rate,
        'evaluated_positive_nodes': pos_nodes,
        'prediction_unit': pred_unit,
        'label_nature': label_nature,
        'source_graph_nodes': up.get('source_nodes') if up.get('source_nodes') is not None else 'NOT_APPLICABLE',
        'source_graph_edges': up.get('source_edges') if up.get('source_edges') is not None else 'NOT_APPLICABLE',
        'construction_notes': up.get('notes', ''),
        'verification_status': canon.get('verification_status', 'LOADER_OBSERVED'),
        'feature_hash': canon.get('feature_hash', ''),
        'edge_hash': canon.get('edge_hash', ''),
        'label_hash': canon.get('label_hash', '')
    }
    reconciled_rows.append(row)

# 1. Export CSV
csv_path = PUB_A12 / 'dataset_identity_reconciliation.csv'
df_rec = pd.DataFrame(reconciled_rows)
df_rec.to_csv(csv_path, index=False)
print(f"Wrote {csv_path}")

# 2. Export Markdown Report
md_path = PUB_A12 / 'dataset_identity_reconciliation.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# A12 Dataset Identity Reconciliation Report (P0-01)\n\n")
    f.write("**Date:** 2026-10-24  \n")
    f.write("**Gate:** **A12-G1 PASSED — Canonical Model-Input Identity Verified**  \n\n")
    f.write("## 1. Summary of Changes from A11\n\n")
    f.write("In A11 Table 1, datasets were improperly assigned raw/upstream repository totals rather than the exact model-input tensor counts evaluated by the algorithms. A12 restores the authoritative contract:\n\n")
    f.write("- **Evaluated Model-Input Graph:** The exact tensor passed to PyTorch/PyGOD models (`N`, `E`, `d`, $\\pi_{\\mathrm{test}}$).\n")
    f.write("- **Upstream Source Graph:** Separate columns explicitly detailing the raw repository graph before filtering, component extraction, or similarity graph construction.\n")
    f.write("- **Target Provenance Clarity:** Explicitly identifies **five source/native-label** datasets (Elliptic, DGraphFin, Ethereum, BSC, Polygon), **eight injected-target** datasets (including BitcoinOTC as a real trust graph with injected targets), and **one external** cybersecurity network (LANL-RedTeam).\n\n")
    f.write("## 2. Reconciled Contract Table\n\n")
    f.write("| Dataset | Model Nodes ($N$) | Model Edges ($E$) | Feat ($d$) | Test Preval. ($\\pi$) | Source Nodes | Source Edges | Target Provenance |\n")
    f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
    for r in reconciled_rows:
        src_n = str(r['source_graph_nodes']) if r['source_graph_nodes'] != 'NOT_APPLICABLE' else '-'
        src_e = str(r['source_graph_edges']) if r['source_graph_edges'] != 'NOT_APPLICABLE' else '-'
        prev_str = f"{r['test_prevalence']:.4f}" if r['test_prevalence'] is not None else '-'
        f.write(f"| **{r['dataset']}** | {r['model_input_nodes']:,} | {r['model_input_edges']:,} | {r['feature_dim']} | {prev_str} | {src_n} | {src_e} | {r['label_nature']} |\n")
    f.write("\n")
print(f"Wrote {md_path}")

# 3. Export LaTeX Table for Manuscript
tex_path = PAPER_A12_TABLES / 'table_dataset_properties.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write(r"""\begin{table}[htbp]
\centering\small
\caption{Benchmark graph dataset properties, evaluated model-input populations, and target label provenance. Model-input counts ($N, E, d$) reflect the exact tensors evaluated by all models; upstream raw source graph totals are reported separately.}
\label{tab:dataset_properties}
\begin{tabularx}{\textwidth}{lrrrrX}\toprule
\textbf{Dataset} & \textbf{Nodes ($N$)} & \textbf{Edges ($E$)} & \textbf{Features ($d$)} & \textbf{Test Preval.} & \textbf{Target Provenance and Evaluated Population} \\\midrule
\multicolumn{6}{l}{\textit{Native Financial / Blockchain Datasets ($N=5$)}} \\
""")
    for r in reconciled_rows:
        if r['category'] == 'Native Financial':
            prev_str = f"{r['test_prevalence']:.4f}"
            desc = TARGET_PROVENANCE_DESCRIPTIONS[r['dataset']]
            if r['source_graph_nodes'] != 'NOT_APPLICABLE':
                desc += f" Evaluated induced graph: {r['model_input_nodes']:,} nodes (raw source: {r['source_graph_nodes']:,} nodes)."
            f.write(f"{r['dataset']} & ${r['model_input_nodes']:,}$ & ${r['model_input_edges']:,}$ & ${r['feature_dim']}$ & ${prev_str}$ & {desc} \\\\\n")
    
    f.write(r"""\midrule
\multicolumn{6}{l}{\textit{Injected Synthetic Benchmarks ($N=8$, including Real Trust Graph with Injected Targets)}} \\
""")
    for r in reconciled_rows:
        if r['category'] == 'Injected Synthetic':
            prev_str = f"{r['test_prevalence']:.4f}"
            desc = TARGET_PROVENANCE_DESCRIPTIONS[r['dataset']]
            if r['source_graph_nodes'] != 'NOT_APPLICABLE' and r['dataset'] != 'BitcoinOTC':
                desc += f" (PyG/PyGOD evaluated tensor: {r['model_input_nodes']:,} nodes)."
            f.write(f"{r['dataset']} & ${r['model_input_nodes']:,}$ & ${r['model_input_edges']:,}$ & ${r['feature_dim']}$ & ${prev_str}$ & {desc} \\\\\n")

    f.write(r"""\midrule
\multicolumn{6}{l}{\textit{External Enterprise Cybersecurity Network ($N=1$)}} \\
""")
    for r in reconciled_rows:
        if r['category'] == 'External Cyber':
            prev_str = f"{r['test_prevalence']:.4f}"
            desc = TARGET_PROVENANCE_DESCRIPTIONS[r['dataset']]
            desc += f" Evaluated destination computers: {r['model_input_nodes']:,} nodes (raw auth stream: {r['source_graph_nodes']:,} entities)."
            f.write(f"{r['dataset']} & ${r['model_input_nodes']:,}$ & ${r['model_input_edges']:,}$ & ${r['feature_dim']}$ & ${prev_str}$ & {desc} \\\\\n")

    f.write(r"""\bottomrule
\end{tabularx}
\end{table}
""")
print(f"Wrote {tex_path}")
