# Applied Sciences Special Issue Submission Gate Report (A16 Closure)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A16  
**Date:** 2026-10-28  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Benchmark Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — A16 APPLIED SCIENCES PACKAGE READY**

---

## 1. Comprehensive Gate Checklist (A16 Closure Work Order)

| Gate ID | Area & Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **A16-G1** | **Programmatic Denominator Audit:** Auto-generated from canonical CSVs | `evidence_denominator_audit.md` and `denominator_crosscheck_a16.csv` auto-generated from `evaluation_population_by_seed.csv` and `dataset_identity_reconciliation.csv`; 14/14 rows match Table 1 & Table 2 identically. | **PASS** |
| **A16-G2** | **Resource Failure Provenance:** 3 exact categories from `support_14x7.csv` | 84/91 primary support; 7 unsupported pairs classified into 4 GADNR `NO_APPROVED_RAW_RUN`, 1 GADNR `OBSERVED_HISTORICAL_CUDA_OOM` (8GB), and 2 AnomalyDAE `OBSERVED_CUDA_OOM` (24GB full encoder forward). | **PASS** |
| **A16-G3** | **DGraphFin Split Hash Identity:** Cryptographic mask verification | Verified `SAME_MASK_DIFFERENT_METADATA`: all 6 supported models on DGraphFin evaluate the identical 183,840 test nodes and 2,326 positives. | **PASS** |
| **A16-G4** | **Concise 172-Word Abstract:** Single-paragraph approved abstract | Replaced abstract in both manuscripts with approved 172-word single-paragraph text matching all tables. | **PASS** |
| **A16-G5** | **Submission Package Hashes & Clean Unpack:** Independent build test | Hashes calculated directly from final disk bytes; clean directory build succeeds (16 pages, 0 undefined references, 0 undefined citations). | **PASS** |
| **A16-G6** | **Extended Validator & Mutation Testing:** Full document validation | 12/12 negative control injection tests caught; zero cross-artifact discrepancies. | **PASS** |

---

## 2. Submission Package Deliverables & Cryptographic Signatures

- **Official LaTeX Source:** `DLG-Benchmark_A15_AppliedSciences.tex` / `DLG-Benchmark_A16_AppliedSciences.tex`  
  - SHA-256: `ea3e3af5ca23345da4f35432d596270d36f8d2efdfeb7877389ba6d425427f0e`
- **Rendered PDF (16 pages):** `DLG-Benchmark_A15_AppliedSciences.pdf` / `DLG-Benchmark_A16_AppliedSciences.pdf`  
  - SHA-256: `cc0187331ea7ad0a6dda3b2b07d15f2a7b133beea2bd45e766cc6d90239c0cb3`  
  - Size: `351,949` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `e5484e7e68dccf3be02f83b3d2541c7d0aa76c091bb790c1e261246540656cb8`  
  - Size: `1,142,897` bytes
  - Contents: `DLG-Benchmark_A15_AppliedSciences.tex`, `references.bib`, `soul.sty`, `soul-ori.sty`, `Definitions/`, `figures/*.pdf`
- **Submission Documents:**
  - `COVER_LETTER.md` (Addressed to Guest Editors Dr. Ayoub and Dr. Mouakher)
  - `DATA_AVAILABILITY.md` (Compliant with MDPI Research Data Policies)
  - `SUBMISSION_GATE_REPORT_A16.md` (This document)
