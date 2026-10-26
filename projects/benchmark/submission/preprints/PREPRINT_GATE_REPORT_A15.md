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
| **A15-G5** | SHA-256 Disk Verification & Clean Unpack | **PASS** | Hashes calculated directly from final disk bytes; clean directory build succeeds (18 pages, 0 undefined references, 0 undefined citations). |
| **A15-G6** | Extended Validator & Mutation Testing | **PASS** | Multi-table verification and negative control injection suite passed. |
| **A15-G7** | Final Sign-Off & Submission Package Ready | **PASS** | **A15 PREPRINT READY** confirmed. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A15_Preprints.tex`  
  - SHA-256: `db69f640e759d414bddbcc242cc0c666a4a020895405506c0a251ba78d3da7d7`
- **Rendered PDF (18 pages):** `DLG-Benchmark_A15_Preprints.pdf`  
  - SHA-256: `243af5be49cfc219141b57ce0b51b8b41573ffc76968e4192805acc17fa415ec`  
  - Size: `564,060` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `85b47f0c51534980979322540252147b381940ddad8f709922a6be975a9167ce`  
  - Size: `106,590` bytes
  - Contents: `DLG-Benchmark_A15_Preprints.tex`, `references.bib`, `figures/*.pdf`
