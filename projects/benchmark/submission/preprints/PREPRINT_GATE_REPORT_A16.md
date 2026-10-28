# Preprints.org Gate Verification Report (A16 Closure)

**Document ID:** DLG-BENCH-GATE-PREPRINT-A16  
**Date:** 2026-10-28  
**Target Platform:** Preprints.org  
**Preprint Policy:** Distinct Benchmark Preprint DOI (independent from preceding DLG-GNN architecture preprint)  
**Benchmark Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Status:** **ALL GATES PASSED (GREEN) — A16 PREPRINT READY**

---

## 1. A16 Gate Verification Summary

| Gate ID | Area | Status | Verification Summary |
|---|---|---|---|
| **A16-G1** | Programmatic Denominator Audit | **PASS** | Auto-generated `evidence_denominator_audit.md` and `denominator_crosscheck_a16.csv` from `evaluation_population_by_seed.csv` and `dataset_identity_reconciliation.csv`; 14/14 dataset rows match Table 1 & Table 2 identically. |
| **A16-G2** | Resource Failure Provenance | **PASS** | Verified 84/91 primary support; 7 unsupported pairs classified into 3 exact categories from `support_14x7.csv` (4 GADNR `NO_APPROVED_RAW_RUN`, 1 GADNR `OBSERVED_HISTORICAL_CUDA_OOM` on 8GB legacy GPU, 2 AnomalyDAE `OBSERVED_CUDA_OOM` in full encoder forward on 24GB RTX 3090). |
| **A16-G3** | DGraphFin Split Hash Identity | **PASS** | Cryptographically verified `SAME_MASK_DIFFERENT_METADATA`: all 6 supported models on DGraphFin evaluate the identical 183,840 test nodes and 2,326 positives. |
| **A16-G4** | Concise 172-Word Abstract | **PASS** | Updated Abstract to single-paragraph 172-word text matching all frozen tables (`+0.0350` Elliptic, `-0.0656` Reddit-Syn, 84/91 primary pairs, 420 primary + 35 external runs). |
| **A16-G5** | Checksums, Page Counts & Git Tag | **PASS** | Hashes computed from final disk bytes; clean unpack build succeeds (18 pages, 0 undefined references, 0 undefined citations). |
| **A16-G6** | Final Submission Gate & Negative Controls | **PASS** | 12/12 negative control injection tests caught; zero cross-artifact discrepancies. |

---

## 2. Submission Deliverables & Cryptographic Signatures

- **Neutral LaTeX Manuscript:** `DLG-Benchmark_A15_Preprints.tex` / `DLG-Benchmark_A16_Preprints.tex`  
  - SHA-256: `ad9b1a9a30091f460f0915dab18ab321d32a94443217e7deb025ed4f0030e567`
- **Rendered PDF (18 pages):** `DLG-Benchmark_A15_Preprints.pdf` / `DLG-Benchmark_A16_Preprints.pdf`  
  - SHA-256: `e1c2e3f622a8a772b01de34bdeba7364c4844b362e71907c31d776c0d7d21de5`  
  - Size: `589,769` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `1c5b6e5624586684b2edd467258c599368527ee9b69d1a98a961e9d14769102f`  
  - Size: `106,914` bytes
  - Contents: `DLG-Benchmark_A15_Preprints.tex`, `references.bib`, `figures/*.pdf`
