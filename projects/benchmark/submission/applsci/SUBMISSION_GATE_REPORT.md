# Gate A20: Submission Gate Report and Canonical Checksum Ledger

**Document ID:** `DLG-BENCH-GATE-A20-RELEASE`  
**Date:** 2026-10-29  
**Status:** PASS (Canonical Ledger Synchronized with Disk Bytes)  
**Target Git Tag:** `benchmark-a20-preprint-2026-10-29` (superseding `benchmark-a19-submission-2026-10-28`)  

---

## 1. Unified Canonical SHA-256 Digest Ledger

All cryptographic hashes below are computed directly from the final disk bytes:

| Publication Target | Artifact File | Size (Bytes) | SHA-256 Digest | Status |
|---|---|---|---|---|
| **Preprints.org** | `DLG-Benchmark_A16_Preprints.pdf` | 602,671 | `a4bc3dd65dba431dd23f7d0cbd74bfd744506c192e79e7d64cbd0bcb1805c96f` | Verified |
| **Preprints.org** | `DLG-Benchmark_A16_Preprints.tex` | 66,043 | `f202bdf160fea13e9abfddf769890aaa3e0d6fc7e65afe17533b0c53809ba6d3` | Verified |
| **Preprints.org** | `source_bundle.zip` | 816,138 | `1b685586ec2e34fad47615db952cc7977cb0280b07874bce6666418c8293ffc7` | Verified |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.pdf` | 352,898 | `c44aa4c6cda155346b7e7eb5cfd13154952b198a634ae3afe841c77b80957af9` | Verified |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.tex` | 65,671 | `257bc901b097709e891880a7d3cbde6bc77426fc7436125c48a7bff0f6d8a03a` | Verified |
| **Applied Sciences** | `source_bundle.zip` | 1,852,125 | `2cf89c7e4376cded7daab1d3290542bcb78b2350cf931badeb5f5a28a56955fb` | Verified |

---

## 2. Compilation and Formatting Checks

- **Preprints.org Manuscript:**
  - Page Count: 20 pages (target satisfied)
  - Undefined References: 0 (`undef refs: []`)
  - Undefined Citations: 0 (`undef cits: []`)
  - Source Bundle Clean-Unpack Verification: Complete TeX, BibTeX, figures matching disk bytes 1:1
- **Applied Sciences Manuscript:**
  - Page Count: 19 pages (target satisfied)
  - Undefined References: 0 (`undef refs: []`)
  - Undefined Citations: 0 (`undef cits: []`)
  - Source Bundle Clean-Unpack Verification: Complete TeX, BibTeX, Definitions, style files, figures matching disk bytes 1:1

---

## 3. Evidence and Editorial Integrity Verification

- **BSC Source Map Alignment:** Verified test positive count vector $[215, 240, 232, 215, 203]$ across CSV, JSON, and prose (exact mean $221.0$).
- **Score-Level Audit Disclosure:** High-resolution per-node anomaly score vectors preserved locally for 105 cryptocurrency evaluations; historical baseline 350 runs preserved as summary metrics without raw score arrays.
- **Practitioner Decision Matrix (Table 7):** Feasibility strictly decoupled from detection accuracy; non-extrapolated coverage and prevalence thresholds calibrated to empirical evidence.
- **Split Protocol Semantics:** Cryptocurrency splits audited as seeded random permutation (non-stratified 60/20/20); synthetic benchmark splits audited as stratified.
- **Hardware Memory Semantics:** Reddit $4N^2 \approx 217$ GB dense lower bound and DGraphFin 6.0 TB clearly differentiated from measured VRAM and illustrative profiles.
- **Figures:** All figures re-rendered from authentic empirical facts; zero unsupported architectural claims.
