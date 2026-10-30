# Gate A22-G5: Manuscript Claim Language & Taxonomy Audit

- **Date:** 2026-10-30
- **Work Order Reference:** `DLG-BENCH-WO-A22-VALIDITY-AND-REVIEWER-READINESS`
- **Status:** PASS (All Overstatements Scoped and Qualified)

---

## 1. Executive Summary

Claude Review Findings M1, M3, M6, and W1 identified several locations where manuscript claims exceeded empirical support or conflated literature references with experimental baselines. This audit formalizes the required textual corrections applied during Gate G8.

## 2. Specific Claim Audits and Required Modifications

### 2.1 Baselines Taxonomy & Related Work (PREM, ADA-GAD, CONAD, TAM)
- **Current Phrasing (Section 2.1):** 'Recent specialized architectures, including PREM, TAM, GADAM, and ADA-GAD, explore graph prompt learning...'
- **Audit Concern:** Reviewer noted that readers could confuse these survey citations with evaluated models in the benchmark table.
- **Resolution:** Explicitly clarify in Section 2.1 that PREM, TAM, and ADA-GAD are modern specialized architectures discussed as related literature, whereas our empirical execution matrix strictly evaluates the five canonical GAD architectures (DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN) alongside DLG-Base and DLG-Aug.

### 2.2 Global Superiority Claims for DLG-Aug
- **Current Phrasing:** Phrases implying that local-to-global decoupling universally improves anomaly detection across all graph classes.
- **Audit Concern:** S4 (synthetic graphs) shows DLG-Aug underperforming DLG-Base (mean rank 3.571 vs 2.143; Reddit $\Delta\mathrm{AP} = -0.0656$). S3 shows DLG-Aug leading (mean rank 2.167), but primarily in crypto/financial graphs.
- **Resolution:** Soften all claims to domain-conditional statements: 'Decoupled local-to-global augmentation yields distinct performance advantages on transaction and financial graphs characterized by high imbalance (e.g., Elliptic, Ethereum, BSC, Polygon), but demonstrates trade-offs or performance regression on dense synthetic citation graphs where global structural homogeneity dominates.'

### 2.3 Exact Full-Graph Semantics
- **Current Phrasing:** 'Exact full-graph evaluation without sampling.'
- **Audit Concern:** Reviewer noted potential confusion between raw upstream node counts (e.g., 203k Elliptic) and evaluated nodes (46k).
- **Resolution:** In Section 3.3 and Section 4.1, explicitly define 'exact full-graph evaluation' as processing the complete evaluated induced graph (all nodes in the component with ground-truth labels) in closed-form, in contrast to mini-batch neighborhood sampling.

### 2.4 Mechanistic Attribution of Local Augmentation (Elliptic Permutation Control)
- **Audit Concern:** Claude Review M3 observed that DLG-Aug-Permuted achieves $\mathrm{AP}=0.1087$ (higher than unpermuted $0.1056$).
- **Resolution:** Explicitly state in the Discussion: 'Diagnostic ablation controls (Section 5.3 / Appendix) reveal that permuting local feature vectors across nodes yields comparable precision ($\mathrm{AP}=0.1087$), indicating that the observed gain on Elliptic is driven by regularized representation expansion rather than strict node-wise local-global semantic alignment.'

