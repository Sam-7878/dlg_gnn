# Preprints.org Gate Verification Report (A15)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A15  
**Date:** 2026-10-25  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Status:** **ALL GATES PASSED (GREEN) — A15 PREPRINT READY**

---

## 1. A15 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A15-G1** | Evaluated Population & Test Prevalence Ledger | **PASS** | Generated `evaluation_population_by_seed.csv` (70 seed evaluations); arithmetic identity verified; test prevalences reconciled (Ethereum 0.4089, BSC 0.1473, Polygon 0.0259). |
| **A15-G2** | Complete Support & Main AP Claims Reconciled | **PASS** | Verified 84/91 primary support; corrected model support description to 5 GADNR OOM + 2 AnomalyDAE Err = 7 unsupported pairs; zero contradictions. |
| **A15-G3** | Operational Claims & Report Harmonization | **PASS** | Replaced causal interpretations in Table 4 and text with observational wording; removed non-contract datasets from execution report. |
| **A15-G4** | Verified Remote Git Identity | **PASS** | Tag `benchmark-a15-submission-2026-10-25` prepared and verified against remote origin. |
| **A15-G5** | SHA-256 Disk Verification & Clean Unpack | **PASS** | Hashes calculated directly from final disk bytes; clean directory build succeeds (19 pages, 0 undefined references, 0 undefined citations). |
| **A15-G6** | Extended Validator & Mutation Testing | **PASS** | Multi-table verification and negative control injection suite passed. |
| **A15-G7** | Final Sign-Off & Submission Package Ready | **PASS** | **A15 PREPRINT READY** confirmed. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A15_Preprints.tex`  
  - SHA-256: `ddb051606aaa3bad27f9489238d19d46420b8b67002f9b2899a84297220e53c2`
- **Rendered PDF (19 pages):** `DLG-Benchmark_A15_Preprints.pdf`  
  - SHA-256: `7330dee7b87af443b1d5ebec422f52319f735393d2d56ccca30a24d790d38a8e`  
  - Size: `534,386` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `bae44d3967e34278a0feca83b429449a55a3b1b1945166b700ab4bcdb72710f3`  
  - Size: `106,478` bytes
  - Contents: `DLG-Benchmark_A15_Preprints.tex`, `references.bib`, `figures/*.pdf`
