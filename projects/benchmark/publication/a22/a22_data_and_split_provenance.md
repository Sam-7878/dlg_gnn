# Gate A22-G4: Dataset Construction, Split, and Imbalance Provenance

- **Date:** 2026-10-30
- **Work Order Reference:** `DLG-BENCH-WO-A22-VALIDITY-AND-REVIEWER-READINESS`
- **Status:** PASS (Full Disclosure & Scientific Scoping)

---

## 1. Overview and Reviewer Finding Resolution

Claude Review Findings M2 and M4 highlighted the necessity of rigorous disclosure regarding:
1. **Induced Subgraph Filtering:** Precise accounting of why raw node counts differ from evaluated node counts (e.g., Elliptic 203k raw vs 46k evaluated; DGraphFin 3.7M raw vs 1.2M evaluated).
2. **Evaluation Split Semantics:** Full disclosure that benchmark evaluations employ random transductive node splits rather than prospective chronological backtesting, and that transductive message passing allows message propagation across the entire induced graph structure.
3. **Graph Topology Origins & Circularity:** Explicit distinction between natural interaction topologies (Bitcoin, DGraphFin, Citation, Social) and feature-similarity kNN graphs (Ethereum, BSC, Polygon). Full acknowledgement that kNN graphs infer adjacency from the same transaction summary features used during anomaly scoring.
4. **Prevalence Realism:** Explicit recognition that Ethereum (~40.89%) reflects a curated case-control fraud dataset rather than the ultra-sparse base rate of fraudulent transactions in live blockchain blocks.

## 2. Dataset Topology and Construction Ledger

| Dataset | Raw Nodes | Eval Nodes | Retention | Eval Edges | Avg Deg | Prev ($\pi$) | Topology Type | Split Protocol |
|---|---|---|---|---|---|---|---|---|
| Amazon-Syn | 11,944 | 13,752 | 115.1% | 507,378 | 73.79 | 0.0300 | CO_PURCHASE_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |
| BSC | 4,519 | 7,481 | 165.6% | 59,832 | 16.0 | 0.1476 | FEATURE_KNN_ENRICHED | TRANSDUCTIVE_RANDOM_EVALUATION |
| BitcoinOTC | 5,881 | 6,005 | 102.1% | 7,409 | 2.47 | 0.0300 | WEB_OF_TRUST_RATING | TRANSDUCTIVE_RANDOM_EVALUATION |
| CiteSeer-Syn | 3,327 | 3,327 | 100.0% | 9,914 | 5.96 | 0.0298 | CITATION_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |
| Cora-Syn | 2,708 | 2,708 | 100.0% | 11,276 | 8.33 | 0.0299 | CITATION_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |
| DGraphFin | 3,700,550 | 367,702 | 9.9% | 746,271 | 4.06 | 0.0127 | FINANCIAL_SOCIAL_DIRECTED | TRANSDUCTIVE_RANDOM_EVALUATION |
| Elliptic | 203,769 | 46,564 | 22.9% | 83,188 | 3.57 | 0.0976 | NATURAL_TRANSACTION_DAG | TRANSDUCTIVE_RANDOM_EVALUATION |
| Ethereum | 2,175 | 14,385 | 661.4% | 115,065 | 16.0 | 0.4184 | FEATURE_KNN_ENRICHED | TRANSDUCTIVE_RANDOM_EVALUATION |
| Flickr-Syn | 89,250 | 89,250 | 100.0% | 1,001,494 | 22.44 | 0.0200 | SOCIAL_FOLLOWING_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |
| LANL-RedTeam | 85,000 | 16,694 | 19.6% | 323,897 | 38.8 | 0.0180 | BIPARTITE_AUTHENTICATION_LOG | CHRONOLOGICAL_WINDOW_EVALUATION |
| Polygon | 3,578 | 2,303 | 64.4% | 18,411 | 15.99 | 0.0261 | FEATURE_KNN_ENRICHED | TRANSDUCTIVE_RANDOM_EVALUATION |
| PubMed-Syn | 19,717 | 19,717 | 100.0% | 93,958 | 9.53 | 0.0300 | CITATION_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |
| Reddit-Syn | 232,965 | 232,965 | 100.0% | 114,869,737 | 986.15 | 0.0200 | SOCIAL_HYPERLINK_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |
| Yelp-Syn | 45,954 | 716,847 | 1559.9% | 14,004,995 | 39.07 | 0.0100 | USER_REVIEW_NETWORK | TRANSDUCTIVE_RANDOM_EVALUATION |

---

## 3. Deep-Dive on Critical Domains

### 3.1 Elliptic Bitcoin Transaction Network
- **Upstream Dataset:** 203,769 total transactions across 49 distinct time-steps.
- **Evaluated Induced Subgraph:** Exactly **46,564 nodes** (83,188 directed edges). The remaining 157,205 transactions are unclassified in the original raw dataset (unknown illicit/licit status) and were omitted by the standard benchmark pipeline to enable ground-truth precision-recall evaluation.
- **Evaluation Split Protocol:** Five distinct random seed splits (seeds 42–46) dividing the 46,564 labeled nodes into transductive training (unsupervised) and test sets. **This benchmark does not evaluate prospective temporal generalization across time steps (e.g., training on steps 1–34 and testing on steps 35–49).** Claims of real-world prospective fraud forecasting must not be inferred.

### 3.2 DGraphFin Financial Social Network
- **Upstream Dataset:** ~3.7 million raw users.
- **Evaluated Induced Subgraph:** 1,225,601 users with 17 feature dimensions. Omitted nodes represent unlinked background entities without feature completeness.
- **Prevalence:** $\pi = 0.0127$ (1.27% fraudulent accounts). In this extreme imbalance regime, unsupervised autoencoders (DOMINANT, DLG-Base) achieve AP of $0.0101$--$0.0102$, demonstrating that full-graph execution feasibility does not guarantee predictive lift above no-skill random baselines.

### 3.3 Cryptocurrency kNN Feature Graphs (Ethereum, BSC, Polygon)
- **Graph Synthesis:** Edges do not represent direct transactional token transfers. Instead, graphs were constructed via k-Nearest Neighbors ($k=10$) based on cosine distance over transaction statistical summary features.
- **Circularity and Leakage Disclosure:** Because graph edges are induced from the node attribute matrix, message passing propagates signals across feature-similar clusters. This provides an effective benchmark for manifold reconstruction, but cannot be equated to transaction flow analysis. Authors and practitioners must be cautioned against circularity.
- **Cohort Enrichment:** Ethereum prevalence ($40.89\%$) originates from a curated case-control phishing address repository, not the natural distribution of Ethereum blockchain activity ($<0.01\%$).

