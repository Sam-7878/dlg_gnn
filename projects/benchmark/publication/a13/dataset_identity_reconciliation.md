# A13 Dataset Identity & Contract Reconciliation

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
