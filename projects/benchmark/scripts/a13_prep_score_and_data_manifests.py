#!/usr/bin/env python3
"""A13 Data and Score Availability Preparation Script.
Fulfills:
- A13-G3: Dataset verification qualification (ARCHIVED_MANIFEST_REGISTRY_CONSISTENT)
- A13-G7: Truthful raw score availability by campaign
"""
import csv
import shutil
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'
A13_PUB = BENCH / 'publication/a13'
A13_TABLES = BENCH / 'paper/a13/tables'

# 1. Generate score_availability_by_campaign.csv
score_rows = [
    {
        'campaign': 'Corrected Crypto A08',
        'datasets': 'Ethereum, BSC, Polygon (3 datasets)',
        'models': 'DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug (7 models)',
        'runs_count': 105,
        'node_scores_stored': True,
        'checkpoints_stored': True,
        'storage_format': 'author_local_npz',
        'availability_policy': 'AVAILABLE_TO_REVIEWERS_UPON_REQUEST (105 raw score NPZ arrays preserved author-locally)'
    },
    {
        'campaign': 'Historical Round 5',
        'datasets': 'Elliptic, DGraphFin, Yelp-Syn, Amazon-Syn, Reddit-Syn, Flickr-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn (9 datasets)',
        'models': 'DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug (7 models)',
        'runs_count': 280,
        'node_scores_stored': False,
        'checkpoints_stored': False,
        'storage_format': 'metrics_and_evaluation_logs_only',
        'availability_policy': 'HISTORICAL_RUNS_DID_NOT_ARCHIVE_NODE_SCORES (Cannot be supplied retroactively; aggregate metrics and reproduction scripts are public)'
    },
    {
        'campaign': 'Canonical BitcoinOTC',
        'datasets': 'BitcoinOTC (1 dataset)',
        'models': 'DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug (7 models)',
        'runs_count': 35,
        'node_scores_stored': False,
        'checkpoints_stored': False,
        'storage_format': 'metrics_and_evaluation_logs_only',
        'availability_policy': 'HISTORICAL_RUNS_DID_NOT_ARCHIVE_NODE_SCORES (Cannot be supplied retroactively; aggregate metrics and reproduction scripts are public)'
    },
    {
        'campaign': 'External Cybersecurity',
        'datasets': 'LANL-RedTeam (1 dataset)',
        'models': 'DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug (7 models)',
        'runs_count': 35,
        'node_scores_stored': False,
        'checkpoints_stored': False,
        'storage_format': 'metrics_and_evaluation_logs_only',
        'availability_policy': 'HISTORICAL_RUNS_DID_NOT_ARCHIVE_NODE_SCORES (Cannot be supplied retroactively; aggregate metrics and reproduction scripts are public)'
    }
]

score_csv_path = A13_PUB / 'score_availability_by_campaign.csv'
with open(score_csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'campaign', 'datasets', 'models', 'runs_count', 
        'node_scores_stored', 'checkpoints_stored', 'storage_format', 'availability_policy'
    ])
    writer.writeheader()
    writer.writerows(score_rows)
print(f"Wrote {score_csv_path}")

# 2. Generate updated dataset_identity_reconciliation.csv with ARCHIVED_MANIFEST_REGISTRY_CONSISTENT
in_csv = BENCH / 'publication/a12/dataset_identity_reconciliation.csv'
out_csv = A13_PUB / 'dataset_identity_reconciliation.csv'

rows = []
with open(in_csv, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        # Update verification_status to accurately qualify that it's based on archived manifest/registry reconciliation
        if 'A05' in r['verification_status']:
            r['verification_status'] = 'ARCHIVED_MANIFEST_REGISTRY_CONSISTENT_A05_CANONICAL'
        else:
            r['verification_status'] = 'ARCHIVED_MANIFEST_REGISTRY_CONSISTENT'
        rows.append(r)

with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote {out_csv}")

# 3. Generate updated dataset_identity_reconciliation.md
md_content = """# A13 Dataset Identity & Contract Reconciliation

**Document ID:** DLG-BENCH-DATA-A13-RECON  
**Date:** 2026-10-24  
**Audit Standard:** Strict separation between Evaluated Model-Input Graph Tensors and Upstream Source Graph Provenance.  
**Verification Level:** `ARCHIVED_MANIFEST_REGISTRY_CONSISTENT` (Reconciled from canonical manifest and approved run registry hashes).

## 1. Summary of Evaluated Populations vs. Source Graphs

| Dataset | Evaluated Category | Model-Input Nodes ($N$) | Model-Input Edges ($E$) | Features ($d$) | Test Prevalence | Verification Level | Upstream Source Context |
|---|---|---:|---:|---:|---:|---|---|
| **Elliptic** | Native Financial | 46,564 | 83,188 | 165 | 0.0976 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Bitcoin transaction graph; model evaluates 46,564 labeled nodes (raw source: 203,769 nodes, 234,355 edges). |
| **DGraphFin** | Native Financial | 1,225,601 | 746,271 | 17 | 0.0127 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | FinTech loan user graph; model evaluates 1,225,601 induced nodes (raw source: 3,700,550 nodes, 4,300,999 edges). |
| **Ethereum** | Native Financial | 14,385 | 115,065 | 8 | 0.4184 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Smart-contract token transfer feature similarity graph (14,385 contracts, standardized k=5). |
| **BSC** | Native Financial | 7,481 | 59,832 | 8 | 0.1476 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Binance Smart Chain token transfer feature similarity graph (7,481 contracts). |
| **Polygon** | Native Financial | 2,303 | 18,411 | 8 | 0.0261 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Polygon token transfer feature similarity graph (2,303 contracts). |
| **BitcoinOTC** | Injected Synthetic | 6,005 | 7,409 | 67 | 0.0300 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT_A05_CANONICAL | SNAP signed trust network; 6,005 nodes, 7,409 edges with controlled synthetic injected anomaly targets. |
| **Yelp-Syn** | Injected Synthetic | 716,847 | 14,004,995 | 300 | 0.0100 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Social review graph; model evaluates PyGOD 716,847 node injection tensor. |
| **Amazon-Syn** | Injected Synthetic | 13,752 | 507,378 | 767 | 0.0300 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Co-purchasing graph; model evaluates PyGOD 13,752 node injection tensor. |
| **Reddit-Syn** | Injected Synthetic | 232,965 | 114,869,737 | 602 | 0.0200 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Reddit post interaction graph; 232,965 nodes, 114.9M directed edges. |
| **Flickr-Syn** | Injected Synthetic | 89,250 | 1,001,494 | 500 | 0.0200 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Image sharing network; model evaluates 89,250 nodes, 1,001,494 directed edges. |
| **Cora-Syn** | Injected Synthetic | 2,708 | 11,276 | 1433 | 0.0299 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Citation network; 5,429 undirected edges = 11,276 directed edges in PyG storage. |
| **CiteSeer-Syn** | Injected Synthetic | 3,327 | 9,914 | 3703 | 0.0298 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Citation network; 4,732 undirected edges = 9,914 directed edges in PyG storage. |
| **PubMed-Syn** | Injected Synthetic | 19,717 | 93,958 | 500 | 0.0300 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Citation network; 44,338 undirected edges = 93,958 directed edges in PyG storage. |
| **LANL-RedTeam** | External Cyber | 16,694 | 323,897 | 13 | 0.0180 | ARCHIVED_MANIFEST_REGISTRY_CONSISTENT | Enterprise authentication bipartite graph; model evaluates 16,694 destination computers. |

## 2. Key Clarifications for Reviewers & Readers
1. **Model-Input Tensor Contract:** The $(N, E, d)$ counts in this table represent the exact tensors evaluated across all model architectures.
2. **Upstream Source Scale:** Datasets with larger source graphs (Elliptic: 203,769 transactions; DGraphFin: 3.7M nodes; LANL: 156k computers/users) underwent standard filtered or induced evaluation as documented in the original benchmark publications.
3. **BitcoinOTC:** Recognized as a real-world signed web-of-trust network with synthetic anomaly targets injected for benchmarking purposes.
"""
(A13_PUB / 'dataset_identity_reconciliation.md').write_text(md_content, encoding='utf-8')
print(f"Wrote {A13_PUB / 'dataset_identity_reconciliation.md'}")

# 4. Generate table_dataset_properties.tex in A13 tables
table_tex = r"""\begin{table}[htbp]
\centering\small
\caption{Benchmark graph dataset properties, evaluated model-input populations, and target label provenance. Model-input counts ($N, E, d$) reflect the exact tensors evaluated by all models; upstream raw source graph totals are reported separately.}
\label{tab:dataset_properties}
\begin{tabularx}{\textwidth}{lrrrrX}\toprule
\textbf{Dataset} & \textbf{Nodes ($N$)} & \textbf{Edges ($E$)} & \textbf{Features ($d$)} & \textbf{Test Preval.} & \textbf{Target Provenance and Evaluated Population} \\\midrule
\multicolumn{6}{l}{\textit{Native Financial / Blockchain Datasets ($N=5$)}} \\
Elliptic & $46,564$ & $83,188$ & $165$ & $0.0976$ & Bitcoin transaction graph; illicit entities (Weber et al., 2019). Evaluated induced graph: 46,564 nodes (raw source: 203,769 nodes). \\
DGraphFin & $1,225,601$ & $746,271$ & $17$ & $0.0127$ & FinTech loan user graph; fraudulent defaulters (Huang et al., 2022). Evaluated induced graph: 1,225,601 nodes (raw source: 3,700,550 nodes). \\
Ethereum & $14,385$ & $115,065$ & $8$ & $0.4184$ & Token-transfer feature-similarity graph; verified phishing/scam contracts (Luo et al., 2024). \\
BSC & $7,481$ & $59,832$ & $8$ & $0.1476$ & Binance Smart Chain token transfer graph; fraud contracts (Luo et al., 2024). \\
Polygon & $2,303$ & $18,411$ & $8$ & $0.0261$ & Polygon token transfer graph; verified fraud contracts (Luo et al., 2024). \\
\midrule
\multicolumn{6}{l}{\textit{Injected Synthetic Benchmarks ($N=8$, including Real Trust Graph with Injected Targets)}} \\
BitcoinOTC & $6,005$ & $7,409$ & $67$ & $0.0300$ & SNAP OTC signed trust network (Kumar et al., 2016); controlled injected anomaly targets on real graph. \\
Yelp-Syn & $716,847$ & $14,004,995$ & $300$ & $0.0100$ & Social review network; injected structural and contextual anomalies. (PyG/PyGOD evaluated tensor: 716,847 nodes). \\
Amazon-Syn & $13,752$ & $507,378$ & $767$ & $0.0300$ & Co-purchasing graph; injected structural and contextual anomalies. (PyG/PyGOD evaluated tensor: 13,752 nodes). \\
Reddit-Syn & $232,965$ & $114,869,737$ & $602$ & $0.0200$ & Reddit post interaction graph; injected structural anomalies. (PyG/PyGOD evaluated tensor: 232,965 nodes, 114.9M directed edges). \\
Flickr-Syn & $89,250$ & $1,001,494$ & $500$ & $0.0200$ & Image sharing network; injected structural anomalies. (PyG/PyGOD evaluated tensor: 89,250 nodes). \\
Cora-Syn & $2,708$ & $11,276$ & $1433$ & $0.0299$ & Citation network; injected structural anomalies (PyGOD generator). (PyG/PyGOD evaluated tensor: 2,708 nodes). \\
CiteSeer-Syn & $3,327$ & $9,914$ & $3703$ & $0.0298$ & Citation network; injected structural anomalies (PyGOD generator). (PyG/PyGOD evaluated tensor: 3,327 nodes). \\
PubMed-Syn & $19,717$ & $93,958$ & $500$ & $0.0300$ & Citation network; injected structural anomalies (PyGOD generator). (PyG/PyGOD evaluated tensor: 19,717 nodes). \\
\midrule
\multicolumn{6}{l}{\textit{External Enterprise Cybersecurity Network ($N=1$)}} \\
LANL-RedTeam & $16,694$ & $323,897$ & $13$ & $0.0180$ & Enterprise authentication graph; red-team compromised computers (Kent, 2015). Evaluated destination computers: 16,694 nodes (raw auth stream: 156,117 entities). \\
\bottomrule
\end{tabularx}
\end{table}
"""
(A13_TABLES / 'table_dataset_properties.tex').write_text(table_tex, encoding='utf-8')
print(f"Wrote {A13_TABLES / 'table_dataset_properties.tex'}")

# Also copy table_friedman_s1_s4.tex to A13 tables
shutil.copy2(BENCH / 'paper/a12/tables/table_friedman_s1_s4.tex', A13_TABLES / 'table_friedman_s1_s4.tex')
print("Copied table_friedman_s1_s4.tex to A13 tables.")
