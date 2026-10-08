# Round P4 Audit Report 01: Public Release Content & Clean Unpack Audit

**Date:** September 23, 2026  
**Auditor:** Automated Benchmark Publication Verification Suite (Round P4)  
**Target Release Asset:** `outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_v1.0.0_preprint.zip`  
**File Size:** 517,801 bytes (0.49 MB)  
**SHA-256 Hash:** `94513B8786F92FDFC0FEEBFECFC02DE9F0E6F6613215C5341EB6DE3149337BE9`

---

## 1. Executive Summary

Round P4 eliminates the historical release archive `DLG_GNN_Benchmark_M5_Release.zip` and builds a completely clean, self-contained public release package `DLG_GNN_Benchmark_v1.0.0_preprint.zip`. The release package contains all source code, models, execution backends, scripts, frozen artifact tables, evaluation metrics, and canonical manifests required for 100% independent replication under Mode 1 (frozen-artifact verification) and Mode 2 (full re-run pipeline).

---

## 2. Stale Release Archive Elimination

- **Decommissioned File:** `outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_M5_Release.zip`
- **Reason for Removal:** Contained obsolete `goat-bank` URLs, "Under Review" phrasing, and internal round terminology.
- **Verification:** The stale file has been physically removed from disk. Only the canonical `DLG_GNN_Benchmark_v1.0.0_preprint.zip` is distributed.

---

## 3. Mandatory Content Structure Audit

The release archive contains 173 files organized across standard open-source top-level directories:

| Component | Path in Archive | Status | Description |
|---|---|:---:|---|
| **Root Docs** | `README.md` | **PASS** | Authoritative Benchmark Reproduction Landing Page |
| | `CITATION.cff` | **PASS** | Machine-readable citation with v1.0.0-preprint and author ORCIDs |
| | `LICENSE` | **PASS** | Permissive MIT License (SeongSu Park & Ki-Hyung Kim) |
| | `INSTALL.md` | **PASS** | Installation guide and environment setup instructions |
| | `pyproject.toml` | **PASS** | Standard PEP 518/621 packaging metadata |
| | `environment.yml` | **PASS** | Full conda environment specification |
| | `release_metadata.json` | **PASS** | Truthful release status (`prepared_for_public_release`) |
| **Source Code** | `src/gog_fraud/` | **PASS** | Complete model definitions, layers, policies, pipelines |
| | `src/analysis/` | **PASS** | Statistical analysis and Friedman/Wilcoxon routines |
| **Experiments** | `experiments/benchmark/` | **PASS** | Primary runner (`run_sci_round5_final.py`), controls (`run_capacity_controls_m3.py`) |
| **Scripts** | `scripts/` | **PASS** | Reproduction script (`reproduce_frozen_artifacts.py`), environment verifier |
| **Tests** | `tests/` | **PASS** | Reproducibility and unit verification test suite |
| **Artifacts** | `artifacts/primary/` | **PASS** | `benchmark_raw.csv`, `model_dataset_support_matrix.csv`, rankings, Wilcoxon |
| | `artifacts/controls/` | **PASS** | Capacity control and permutation sensitivity summaries |
| | `artifacts/lanl/` | **PASS** | LANL final validation and calibration tables |
| | `artifacts/manifests/` | **PASS** | Canonical graph manifests (node/edge counts, feature dimensions) |
| **Provenance** | `provenance/` | **PASS** | `frozen_execution_environment.json`, `current_reproduction_environment.json` |
| **Documentation** | `docs/math/` | **PASS** | `exact_sparse_reconstruction.md` (single-source sparse theorem) |

---

## 4. Frozen Artifact Hash Verification

All frozen empirical results inside the archive match the immutable cryptographic hashes with zero deviations:

| Frozen Artifact | Relative Path in Archive | Status | SHA-256 Hash |
|---|---|:---:|---|
| Primary Benchmark Data | `artifacts/primary/benchmark_raw.csv` | **PASS** | `39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c` |
| Model-Dataset Support | `artifacts/primary/model_dataset_support_matrix.csv` | **PASS** | `c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914` |
| Seed Aggregated Data | `artifacts/primary/seed_aggregated_performance.csv` | **PASS** | Verified identical to manuscript Table 4 |
| Primary Rankings | `artifacts/primary/rankings.csv` | **PASS** | Verified identical to manuscript Section 4 |
| Friedman Tests | `artifacts/primary/friedman_tests.csv` | **PASS** | Verified identical to manuscript Table 5 |
| Wilcoxon-Holm Tests | `artifacts/primary/wilcoxon_holm.csv` | **PASS** | Verified identical to manuscript Table 6 |

---

## 5. String Sanitation & Privacy Audit

Automated string scan of all 173 files inside the archive confirmed 0 occurrences of prohibited terms:

- Banned string `goat-bank`: **0 occurrences** (PASS)
- Banned string `under review`: **0 occurrences** (PASS)
- Banned path `d:\_work`: **0 occurrences** (PASS)
- Banned path `/mnt/d/_work/`: **0 occurrences** (PASS)
- Preceding Paper DOI: Correctly separated as `10.20944/preprints202609.0848.v1`
- Current Paper DOI: Correctly declared `null` pending Preprints.org issuance

---

## 6. Clean Unpack Mode 1 Execution Audit

An automated clean-unpack test unpacked `DLG_GNN_Benchmark_v1.0.0_preprint.zip` into an isolated scratch directory and executed:
```bash
python scripts/reproduce_frozen_artifacts.py --artifacts-dir artifacts --output-dir scratch/reproduced_tables
```

**Result:**
- Elapsed time: 0.40 seconds
- Return code: 0
- Tables verified: All 8 LaTeX tables and statistical test outputs reproduced with 0 errors.

---

## 7. Audit Verdict

**STATUS: PASS**  
The release asset `DLG_GNN_Benchmark_v1.0.0_preprint.zip` is completely clean, self-contained, sanitized, and ready for publication release on GitHub.
