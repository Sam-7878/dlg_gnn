# Gate A22-G6: Feature-Only Tabular Anomaly Detection Scope Control

- **Date:** 2026-10-30
- **Work Order Reference:** `DLG-BENCH-WO-A22-VALIDITY-AND-REVIEWER-READINESS`
- **Gate:** A22-G6
- **Status:** SKIPPED_SCOPE_CONTROL (Bounded Supplementary Analysis)
- **Artifact:** `projects/benchmark/publication/a22/a22_feature_only_baseline_scope.md`

---

## 1. Context and Reviewer Query

In the Claude Review of `DLG-Benchmark_A16_Preprints.pdf` (Section 4.1), the reviewer inquired:
> "On financial and crypto datasets such as Elliptic (where node features contain 165 dense transaction statistics), does graph neural message passing significantly outperform classical tabular anomaly detectors (e.g., Isolation Forest, ECOD, or XGBoost-based unsupervised detectors) that ignore graph topology entirely?"

## 2. Benchmark Scope and Decision

1. **Core Scope of Current Benchmark:**
   - The primary objective of this benchmark is to evaluate **graph-relational anomaly detection (GAD) algorithms** (DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug) under standardized PyGOD semantics, focusing on structural versus contextual reconstruction trade-offs and scaling bottlenecks on large graphs.
   - The 455 primary runs across 14 graphs were conducted under a unified graph data interface.

2. **Orthogonal Role of Tabular Baselines:**
   - Previous literature (e.g., Weber et al., 2019; Tang et al., GADBench 2023) has shown that on Elliptic, simple tree ensembles or tabular models can achieve competitive AP when high-dimensional handcrafted transaction features are provided, whereas graph convolutions add value primarily when local homophily is strong or features are sparse.
   - Introducing tabular feature-only baselines across 14 datasets at this stage would require expanding the benchmark scope beyond graph architectures, violating the frozen submission baseline.

3. **Disposition for Manuscript Revision (Gate G8):**
   - In Section 6 (Discussion & Limitations), explicitly acknowledge this architectural scope boundary:
     > "Our benchmark specifically evaluates graph-relational anomaly detection architectures. On transaction datasets with rich feature sets (such as Elliptic), classical tabular detectors (e.g., Isolation Forest, ECOD) represent an orthogonal baseline class. While graph models exploit relational connectivity and message passing, future benchmark extensions should formally evaluate whether relational inductive biases offer consistent lift over tabular methods across diverse imbalance regimes."

