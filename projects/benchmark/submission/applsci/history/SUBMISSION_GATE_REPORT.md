# Applied Sciences Special Issue Submission Gate Report (A13)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A13  
**Date:** 2026-10-24  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — A13 APPLIED SCIENCES PACKAGE READY**

---

## 1. Comprehensive Gate Checklist (A13 Work Order)

| Gate ID | Area & Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **A13-G1** | **Input Freeze & Inventory:** 38 inputs from A12 frozen and inventoried | Computed disk-byte SHA-256 for all inputs; recorded in `a12_input_sha256_manifest.csv`. | **PASS** |
| **A13-G2** | **Literal Manuscript Numeric Contract:** Literal `.tex` verified directly against registries | Verified literal `.tex` files match all numbers (Reddit AP 0.4044/0.3388, $\Delta -0.0656$; Elliptic AP 0.0687/0.1037, $\Delta +0.0350$; 3 blockchain comparators; Friedman S1--S4). Negative control passed. | **PASS** |
| **A13-G3** | **Truthful Score Availability:** Transparent disclosure of archived raw scores | Published `score_availability_by_campaign.csv` disclosing 105 crypto runs available vs 350 historical runs without node scores. | **PASS** |
| **A13-G4** | **Causal Language Elimination:** No unverified causal claims | All unproved hub claims softened to topological hypotheses; authentic below-0.50 ROC-AUC preserved with zero post-hoc sign flipping. | **PASS** |
| **A13-G5** | **Memory & Platform Accuracy:** Correct float32 math and hardware platform | Single matrix $4N^2$ (10.0 GB/160.0 GB) and two matrices $8N^2$ (20.0 GB/320.0 GB). Gram identity qualified to unsquashed dot-products. Hardware: RTX 3090 24GB eGPU via Thunderbolt 3. | **PASS** |
| **A13-G6** | **Fail-Closed Figures & Seed Pairing:** Fig 2 fail-closed, Fig 3 seed-paired SE | Figure 2 requires 5 unique seeds {42..46}; Figure 3 calculates seed-paired differences ($n=5$) with exported data `fig3_paired_delta_data.csv`. | **PASS** |
| **A13-G7** | **Dataset Identity Reconciliation:** Canonical $N, E, d$ separated from raw totals | Reconciled 14 datasets with status `ARCHIVED_MANIFEST_REGISTRY_CONSISTENT`. Table 1 synchronized. | **PASS** |
| **A13-G8** | **Self-Contained Source Bundle:** Complete ZIP packages | Created `source_bundle.zip` with MDPI style files, Definitions, and vector figures. | **PASS** |
| **A13-G9** | **Immutable Git Release Tag:** `benchmark-a13-submission-2026-10-24` | Tag integrated across all manuscripts, cover letters, and manifests. | **PASS** |
| **A13-G10** | **Clean Unpack Build Verification:** Independent clean-directory compile | Unpacked into clean temporary environment and compiled with pdflatex + bibtex: 16 pages, 0 undefined references, 0 undefined citations. | **PASS** |
| **A13-G11** | **Dual Submission Sign-Off:** Package complete and verified | **A13 APPLIED SCIENCES PACKAGE READY** confirmed. | **PASS** |

---

## 2. Submission Package Deliverables & Cryptographic Signatures

- **Official LaTeX Source:** `DLG-Benchmark_A13_AppliedSciences.tex`  
  - SHA-256: `b64e25f19cb0b7a2af464c6e555ac6dba1110b18ebf67dd97a26b0004f243078`
- **Rendered PDF (16 pages):** `DLG-Benchmark_A13_AppliedSciences.pdf`  
  - SHA-256: `7e7a587e3dccd2d7b425960fc32ecae767c7b46561e29c7436c705563379e09a`  
  - Size: `334,958` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `f5e2cab36a4ec25dd3c67e7b1767d53aec71cadf944b9c8db9d3fcf21bcdb260`  
  - Size: `1,135,946` bytes
  - Contents: `DLG-Benchmark_A13_AppliedSciences.tex`, `references.bib`, `soul.sty`, `soul-ori.sty`, `Definitions/`, `figures/*.pdf`
- **Submission Documents:**
  - `COVER_LETTER.md` (Addressed to Guest Editors Dr. Ayoub and Dr. Mouakher)
  - `DATA_AVAILABILITY.md` (Compliant with MDPI Research Data Policies)
  - `SUBMISSION_GATE_REPORT.md` (This document)
