# Preprints.org Gate Verification Report (A13)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A13  
**Date:** 2026-10-24  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Status:** **ALL GATES PASSED (GREEN) — A13 PREPRINT READY**

---

## 1. A13 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A13-G1** | Input Freeze & Inventory | **PASS** | 38 inputs from A12 frozen and inventoried with exact disk SHA-256 in `a12_input_sha256_manifest.csv`. |
| **A13-G2** | Literal Manuscript Numeric Contract | **PASS** | Literal `.tex` verified directly against registries (0.4044/0.3388, $\Delta -0.0656$; Elliptic 0.0687/0.1037, $\Delta +0.0350$; Ethereum, BSC, Polygon comparators; S1--S4 statistics). Negative control verified. |
| **A13-G3** | Truthful Score Availability | **PASS** | `score_availability_by_campaign.csv` published disclosing 105 crypto runs available vs 350 historical runs without raw score arrays. |
| **A13-G4** | Causal Language Elimination | **PASS** | Unproved causal hub claims eliminated across Title, Abstract, Results, and Discussion; authentic empirical $<0.50$ ROC-AUC preserved without sign flipping. |
| **A13-G5** | Memory & Platform Accuracy | **PASS** | Dense float32 memory arithmetic verified ($4N^2$ single, $8N^2$ two; 10.0 GB/160.0 GB and 20.0 GB/320.0 GB). Closed-form Gram identity strictly qualified to unsquashed dot products. Hardware accurately stated as RTX 3090 24GB eGPU via Thunderbolt 3. |
| **A13-G6** | Fail-Closed Figures & Seed Pairing | **PASS** | Figure 2 support matrix verified fail-closed (requires 5 unique seeds {42..46}). Figure 3 computed with strict seed-paired difference ($n=5$) and exported to `fig3_paired_delta_data.csv`. |
| **A13-G7** | Dataset Identity Reconciliation | **PASS** | 14 canonical datasets reconciled with status `ARCHIVED_MANIFEST_REGISTRY_CONSISTENT`. Table 1 synchronized. |
| **A13-G8** | Self-Contained Source Bundle | **PASS** | Complete `source_bundle.zip` created and verified with clean unpack compilation. |
| **A13-G9** | Immutable Git Release Tag | **PASS** | Release tag `benchmark-a13-submission-2026-10-24` integrated throughout manuscript and metadata. |
| **A13-G10** | Clean Unpack Build Verification | **PASS** | Unpacked into clean temporary environment and compiled with pdflatex + bibtex: 0 undefined references, 0 undefined citations. |
| **A13-G11** | Dual Submission Sign-Off | **PASS** | **A13 PREPRINT READY** sign-off confirmed. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A13_Preprints.tex`  
  - SHA-256: `c6a5b94168188474dbc0ae7ae5f2503231f5f993ddfdcc81af49d60a3a0bfcee`
- **Rendered PDF (18 pages):** `DLG-Benchmark_A13_Preprints.pdf`  
  - SHA-256: `4491f4c035b1caa2f63783f59dce3a48a10881cf01ff31f768f2014fbf7d16c6`  
  - Size: `510,977` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `2dcfe3e83b83a1b1e34e595640aec92c85b0f94b29f6b6e7fde75cba3b83fefa`  
  - Size: `99,960` bytes
  - Contents: `DLG-Benchmark_A13_Preprints.tex`, `references.bib`, `figures/*.pdf`
- **Preprint DOI Policy:** To be registered as a distinct, standalone benchmark preprint upon upload.
