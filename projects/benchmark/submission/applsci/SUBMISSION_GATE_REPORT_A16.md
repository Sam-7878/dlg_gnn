# Applied Sciences Special Issue Submission Gate Report (A16 Closure & A17 Gate Check)

**Document ID:** DLG-BENCH-GATE-APPLSCI-A16  
**Date:** 2026-10-28  
**Target:** *Applied Sciences* (MDPI), Special Issue **"Machine Learning and Its Application for Anomaly Detection"** (Special Issue ID: `C67H3U9QGT`)  
**Official Deadline:** 2026-12-31  
**Benchmark Git Tag:** `benchmark-a16-submission-2026-10-28` (superseding `benchmark-a15-submission-2026-10-25`)  
**Submission Sequence:** `Preprints.org FIRST (Distinct Benchmark DOI) -> Applied Sciences SI`  
**Overall Status:** **ALL GATES PASSED (GREEN) — APPLIED SCIENCES PACKAGE READY**

---

## 1. Comprehensive Gate Checklist

| Gate ID | Area & Required Condition | Audit Outcome | Status |
|---|---|---|---|
| **Gate R1** | Remote Git Tag Independent Verification | Tag `benchmark-a16-submission-2026-10-28` dereferencing to commit `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f` verified on GitHub via `git ls-remote` and GitHub API HTTP 200. | **PASS** |
| **Gate R2** | Private Review-Process Language Removal | Replaced "Claude's review" with neutral statistical narrative explaining within-block permutation vs asymptotic $\chi^2$ differences. | **PASS** |
| **Gate R3** | Source ZIP Internal Filename Normalization | Rebuilt `source_bundle.zip` with `DLG-Benchmark_A16_AppliedSciences.tex` as internal main file; clean unpack succeeds (16 pages, 0 undefined references/citations). | **PASS** |
| **Gate R4** | Memory Claims Reframed as Profiles | Table 6 reframed from "measured VRAM" to illustrative peak memory materialization profiles on RTX 3090, correctly noting AnomalyDAE preflight OOM on Reddit-Syn. | **PASS** |
| **Gate R5** | Decision Matrix Bounded to Observations | Table 7 replaces unverified causal phrases with concrete empirical AP observations. | **PASS** |
| **Gate R6** | Extended Validator & Cryptographic Digests | All 15 negative control tests pass; SHA-256 digests verified on actual disk bytes; 0 undefined references/citations. | **PASS** |

---

## 2. Submission Package Deliverables & Cryptographic Signatures

- **Official LaTeX Source:** `DLG-Benchmark_A16_AppliedSciences.tex`  
  - SHA-256: `61a93c4ee5f1b9d7448befb495b9908857d3e04499f1901568feb000ebd37947`
- **Rendered PDF (16 pages):** `DLG-Benchmark_A16_AppliedSciences.pdf`  
  - SHA-256: `36c0f66a7a50e8b2eb32d5a626ae92b891dea4d30172f2f1972073822604eef2`  
  - Size: `352,185` bytes
- **Source Bundle Archive:** `source_bundle.zip`  
  - SHA-256: `344f5d30e6f53d7da2242d194825444e172c36952f12c4aa30d97c66bd3fae6b`  
  - Size: `1,142,987` bytes
  - Contents: `DLG-Benchmark_A16_AppliedSciences.tex`, `references.bib`, `soul.sty`, `soul-ori.sty`, `Definitions/`, `figures/*.pdf`
- **Submission Documents:**
  - `COVER_LETTER.md` (Addressed to Guest Editors Dr. Ayoub and Dr. Mouakher)
  - `DATA_AVAILABILITY.md` (Compliant with MDPI Research Data Policies)
  - `SUBMISSION_GATE_REPORT_A16.md` (This document)
