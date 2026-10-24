# Applied Sciences Special Issue Submission Gate Report (A15)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A15  
**Date:** 2026-10-25  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — A15 APPLIED SCIENCES PACKAGE READY**

---

## 1. Comprehensive Gate Checklist (A15 Work Order)

| Gate ID | Area & Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **A15-G1** | **Evaluation Population & Test Prevalence Ledger:** Pinned test denominator ledger | Generated `evaluation_population_by_seed.csv` (70 seed evaluations); arithmetic identity verified; test prevalences reconciled (Ethereum 0.4089, BSC 0.1473, Polygon 0.0259). | **PASS** |
| **A15-G2** | **Support Matrix & Model Counts:** Reconciled AP and support counts | Verified 84/91 primary support; corrected model support description to 5 GADNR OOM + 2 AnomalyDAE Err = 7 unsupported pairs; zero contradictions. | **PASS** |
| **A15-G3** | **Operational Claims & Report Harmonization:** Observational wording & clean 13+1 report | Replaced causal interpretations in Table 4 and text with observational wording; removed non-contract datasets from execution report. | **PASS** |
| **A15-G4** | **Public Remote Git Identity:** Verified remote git reference | Tag `benchmark-a15-submission-2026-10-25` prepared and verified against remote origin. | **PASS** |
| **A15-G5** | **Submission Package Hashes & Clean Unpack:** Independent build test | Hashes calculated directly from final disk bytes; clean directory build succeeds (16 pages, 0 undefined references, 0 undefined citations). | **PASS** |
| **A15-G6** | **Extended Validator & Mutation Testing:** Full document validation | Multi-table verification and negative control injection suite passed. | **PASS** |
| **A15-G7** | **Dual Submission Sign-Off:** Readiness confirmed | **A15 APPLIED SCIENCES PACKAGE READY** confirmed. | **PASS** |

---

## 2. Submission Package Deliverables & Cryptographic Signatures

- **Official LaTeX Source:** `DLG-Benchmark_A15_AppliedSciences.tex`  
  - SHA-256: `b72d2bbc6550875da17cf2e3174cb7156b2e1859fe723f88efc45fe8a25248b3`
- **Rendered PDF (16 pages):** `DLG-Benchmark_A15_AppliedSciences.pdf`  
  - SHA-256: `a6bfabead9c962c5403ab787467c94db8a988f37a1593126deb1199b653956d8`  
  - Size: `346,552` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `6fb4753d05057e1a23c520cf2358a4fc131a0c836336d73139c74ceaa585437d`  
  - Size: `1,142,459` bytes
  - Contents: `DLG-Benchmark_A15_AppliedSciences.tex`, `references.bib`, `soul.sty`, `soul-ori.sty`, `Definitions/`, `figures/*.pdf`
- **Submission Documents:**
  - `COVER_LETTER.md` (Addressed to Guest Editors Dr. Ayoub and Dr. Mouakher)
  - `DATA_AVAILABILITY.md` (Compliant with MDPI Research Data Policies)
  - `SUBMISSION_GATE_REPORT_A15.md` (This document)
