# Gate A23: Submission Gate Report and Canonical Checksum Ledger

**Document ID:** `DLG-BENCH-GATE-A23-RELEASE`  
**Date:** 2026-10-30  
**Status:** PASS (Canonical Ledger Synchronized with Disk Bytes in Single Post-Build Pass)  
**Target Git Tag:** `benchmark-a23-preprint-2026-10-30` (superseding `benchmark-a22-submission-2026-10-30` and `benchmark-a20-preprint-2026-10-29`)  

---

## 1. Unified Canonical SHA-256 Digest Ledger

All cryptographic hashes below are computed directly from the final frozen disk bytes after compilation and isolated bundle verification:

| Publication Target | Artifact File | Size (Bytes) | SHA-256 Digest | Status |
|---|---|---:|---|---|
| **Preprints.org** | `DLG-Benchmark_A16_Preprints.pdf` | 615,068 | `2ba2757a5372ccb05659ea664d4827956479a33acafa525eac397d94e341b719` | Verified |
| **Preprints.org** | `DLG-Benchmark_A16_Preprints.tex` | 72,260 | `f7d88d32810dbc545068b48310b5ca1f438769e9fd1f8b6775a4105ffe358b86` | Verified |
| **Preprints.org** | `references.bib` | 19,402 | `b42f92f2f6143b4cf79eaee3ed55f7e01deba2a6fc4d77164529c4ef3b19e42c` | Verified |
| **Preprints.org** | `source_bundle.zip` | 818,200 | `aec6d84e5f02dbc30710ffc7d019070f6d2de137ec3cbace7933514e8e187f62` | Verified |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.pdf` | 366,225 | `6cd073eda041b9a9767f994e32cdc4f38efd2a0303939e246c759a69e0ac8026` | Verified |
| **Applied Sciences** | `DLG-Benchmark_A16_AppliedSciences.tex` | 71,888 | `55c268a3f2f0776f1ff1662e3f66f2331d1d25ff3717bdd814cba216a83b7022` | Verified |
| **Applied Sciences** | `references.bib` | 19,402 | `b42f92f2f6143b4cf79eaee3ed55f7e01deba2a6fc4d77164529c4ef3b19e42c` | Verified |
| **Applied Sciences** | `source_bundle.zip` | 1,854,183 | `15a1d5cae582ce8673b5dcbdc07a12650b8d78008bb5b9419c4d41e38393cbbc` | Verified |

---

## 2. Compilation and Isolated Clean-Unpack Checks

- **Preprints.org Manuscript:**
  - Page Count: 23 pages
  - Undefined References: 0 (`undef_refs: []`)
  - Undefined Citations: 0 (`undef_cits: []`)
  - Isolated `/tmp` Clean-Unpack Compile Check: PASS (23 pages, 0 LaTeX errors)
- **Applied Sciences Manuscript:**
  - Page Count: 20 pages
  - Undefined References: 0 (`undef_refs: []`)
  - Undefined Citations: 0 (`undef_cits: []`)
  - Isolated `/tmp` Clean-Unpack Compile Check: PASS (20 pages, 0 LaTeX errors)
