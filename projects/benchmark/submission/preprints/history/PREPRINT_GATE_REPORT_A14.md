# Preprints.org Gate Verification Report (A14)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A14  
**Date:** 2026-10-24  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Status:** **ALL GATES PASSED (GREEN) — A14 PREPRINT READY**

---

## 1. A14 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A14-G1** | Canonical Approved-Run Selection | **PASS** | Pinned `canonical_selected_run_registry.csv` (455 runs); resolved crypto campaign conflict with `benchmark_a08_crypto_clean_v1`. |
| **A14-G2** | Complete 14-Dataset Delta Table | **PASS** | Source-derived Table 4 (`table_conditional_delta.tex`) and paired CSVs (`paired_ap_by_seed.csv`, `conditional_ap_14.csv`) generated and synchronized. |
| **A14-G3** | Figure 3 Layout & PDF Rendering | **PASS** | Generated un-clipped horizontal bar chart with dynamic bounds (`[-0.112, 0.065]`); verified BSC bar is $+0.0021$ (gain); zero collisions. |
| **A14-G4** | Whole-Document Numerical Validation | **PASS** | Literal `.tex` parsed for all 14 rows; cross-checked with canonical CSV; 6/6 negative control injection tests rejected. |
| **A14-G5** | Reproducible Figure/Source Commands | **PASS** | Repository-relative paths implemented across all scripts; fast verification entrypoint provided. |
| **A14-G6** | Public Remote Git Identity | **PASS** | Tag `benchmark-a14-submission-2026-10-24` created and verified. |
| **A14-G7** | Submission Package Hashes & Clean Unpack | **PASS** | Clean directory compilation verified: 18 pages, 0 undefined references, 0 undefined citations. |
| **A14-G8** | Preprint / Journal Release Sign-off | **PASS** | **A14 PREPRINT READY** confirmed. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A14_Preprints.tex`  
  - SHA-256: `21424cc8e323f929b8eea1f7aaeebf90bf3aec78f5a8906c32bf8413c48be8ee`
- **Rendered PDF (18 pages):** `DLG-Benchmark_A14_Preprints.pdf`  
  - SHA-256: `671b0c80944de257595409b4cd59c1f88aa86da6aab3b43bf5dd59b936d0fe1c`  
  - Size: `540,944` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `5ca731a678fc89f57d5ea194bdbdf7107a492c6adef746fd6f2ec1a1bfc4f5c3`  
  - Size: `117,721` bytes
  - Contents: `DLG-Benchmark_A14_Preprints.tex`, `references.bib`, `figures/*.pdf`
