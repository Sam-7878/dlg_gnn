# Applied Sciences Special Issue Submission Gate Report (A14)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A14  
**Date:** 2026-10-24  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — A14 APPLIED SCIENCES PACKAGE READY**

---

## 1. Comprehensive Gate Checklist (A14 Work Order)

| Gate ID | Area & Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **A14-G1** | **Canonical Approved-Run Selection:** Single coherent execution registry | Pinned `canonical_selected_run_registry.csv` (455 runs); resolved crypto campaign conflict with `benchmark_a08_crypto_clean_v1`. | **PASS** |
| **A14-G2** | **Source-Derived Delta Table:** All deltas computed directly from canonical CSV | Generated `table_conditional_delta.tex` and paired CSVs (`paired_ap_by_seed.csv`, `conditional_ap_14.csv`); verified arithmetic exactness. | **PASS** |
| **A14-G3** | **Correct Figure 3 & Unclipped Layout:** Clear bounds without bar clipping | Generated unclipped horizontal bar chart with dynamic bounds (`[-0.112, 0.065]`); verified BSC bar is $+0.0021$ (gain); zero collisions. | **PASS** |
| **A14-G4** | **Whole-Document Numerical Validation:** 14-row table and negative controls | Literal `.tex` parsed for all 14 rows; cross-checked with canonical CSV; 6/6 negative control injection tests rejected. | **PASS** |
| **A14-G5** | **Reproducible Figure/Source Commands:** Repository-relative discovery | Repository-relative paths implemented across all scripts; fast verification entrypoint provided. | **PASS** |
| **A14-G6** | **Public Remote Git Identity:** Verified remote git reference | Tag `benchmark-a14-submission-2026-10-24` created and pushed to GitHub remote repository. | **PASS** |
| **A14-G7** | **Submission Package Hashes & Clean Unpack:** Independent build test | Clean directory compilation verified: 16 pages, 0 undefined references, 0 undefined citations. | **PASS** |
| **A14-G8** | **Dual Submission Sign-Off:** Readiness confirmed | **A14 APPLIED SCIENCES PACKAGE READY** confirmed. | **PASS** |

---

## 2. Submission Package Deliverables & Cryptographic Signatures

- **Official LaTeX Source:** `DLG-Benchmark_A14_AppliedSciences.tex`  
  - SHA-256: `fd750d53c28ef717b5f70dc523f954be119f018adb1a869f56e906a977ae8f80`
- **Rendered PDF (16 pages):** `DLG-Benchmark_A14_AppliedSciences.pdf`  
  - SHA-256: `e43270c3c8a7d2e513923c107674121506c7d18b83871cf7f71a8a94a44efb38`  
  - Size: `355,261` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `1122e1a982f700b4b75f7473a3ad7e17e3f0338aa573696522d3674bc41a990a`  
  - Size: `1,153,703` bytes
  - Contents: `DLG-Benchmark_A14_AppliedSciences.tex`, `references.bib`, `soul.sty`, `soul-ori.sty`, `Definitions/`, `figures/*.pdf`
- **Submission Documents:**
  - `COVER_LETTER.md` (Addressed to Guest Editors Dr. Ayoub and Dr. Mouakher)
  - `DATA_AVAILABILITY.md` (Compliant with MDPI Research Data Policies)
  - `SUBMISSION_GATE_REPORT_A14.md` (This document)
