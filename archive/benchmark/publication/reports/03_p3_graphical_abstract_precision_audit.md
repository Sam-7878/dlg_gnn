# Round P3 Audit Report 03: Graphical Abstract Precision Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite  
**Scope:** Graphical Abstract terminology precision, synthetic dataset suffixing, support-rate decomposition, suite maxima phrasing, and rank scope boundaries.

---

## 1. Provenance Consistency: Synthetic Display Names

In accordance with manuscript-wide provenance standards, all seven synthetic-injection datasets in the Graphical Abstract display box have been explicitly suffixed with `-Syn`:
- `Yelp-Syn`
- `Amazon-Syn`
- `Flickr-Syn`
- `Reddit-Syn`
- `Cora-Syn`
- `CiteSeer-Syn`
- `PubMed-Syn`

This prevents reader confusion between public base graphs and the frozen contextual/structural anomaly-injected graphs generated for this study.  
**Automated Test:** `tests/benchmark/publication_p3/test_graphical_abstract_syn_suffixes.py` (PASS)

---

## 2. Decoupled Support-Rate & Suite Count Statements

Previous versions conflated baseline support rates with the total suite count in an ambiguous sentence:
> *"... baseline support spans 50%-100% (71/80 supported pairs)"*

In Round P3, the two statements were decoupled:
> *"Baseline support spans 50% - 100%."*  
> *"Overall, 71/80 model-dataset pairs are supported."*

This clearly distinguishes the baseline model operational range from the global 80-cell support matrix total.  
**Automated Test:** `tests/benchmark/publication_p3/test_graphical_abstract_support_statement_separated.py` (PASS)

---

## 3. Accurate Graph Suite Maxima Wording

Previous phrasing (`"Graph Size: up to 1.2M nodes, 114M edges"`) inadvertently suggested that a single graph reached both node and edge maxima simultaneously. In Round P3, this was corrected to:
> *"Suite maxima: 1.23M nodes and 114.9M edges"*

This reflects that DGraphFin provides the suite maximum for nodes ($N = 1,239,268$), while Reddit-Syn provides the suite maximum for edges ($E = 114,942,634$).  
**Automated Test:** `tests/benchmark/publication_p3/test_graphical_abstract_suite_maxima_wording.py` (PASS)

---

## 4. Explicit Fraud-Rank Scope

To avoid any implication of broad or universal superiority across all 10 graphs or all 8 models, the top-rank finding was strictly qualified:
> *"Best Average Rank in Fraud-Oriented Common Subset:"*  
> *"DLG-Aug ranks 1st in ROC-AUC (1.71), PR-AUC (1.71), and Val.-F1 (1.86) across 7 fraud graphs (5 models)."*

This aligns exactly with Section 5 of the manuscript and Table 3.  
**Automated Test:** `tests/benchmark/publication_p3/test_graphical_abstract_rank_scope.py` (PASS)

---

## 5. Visual Asset Verification

The graphical abstract generator script (`scripts/manuscript/generate_graphical_abstract_p1.py`) was executed to produce `publication/benchmark/preprints/graphical_abstract.png`:
- **Dimensions:** $4,200 \times 2,400$ pixels
- **Resolution:** 300 DPI
- **File Size:** 431.1 KB
- **Clipping / Overlap:** None observed across all 5 structured card layers.

---

## 6. Audit Verdict
**STATUS: PASS (`READY_FOR_PREPRINTS_ORG_DEPOSIT`)**  
The Graphical Abstract is mathematically precise, provenance-compliant, and visually verified.
