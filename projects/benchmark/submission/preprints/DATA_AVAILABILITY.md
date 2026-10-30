# Data and Reproduction Availability Statement (A20 Preprints.org)

**Document ID:** DLG-BENCH-DATA-PREPRINT-A20  
**Date:** 2026-10-29  
**Target Platform:** Preprints.org  
**Benchmark Git Tag:** `benchmark-a22-submission-2026-10-30` (superseding `benchmark-a20-preprint-2026-10-29`)  
**Repository URL:** https://github.com/Sam-7878/dlg_gnn  

---

## 1. Summary of Publicly Available Artifacts

All benchmark evaluation scripts, aggregation pipelines, mathematical validation routines, and figure generation scripts are openly archived under immutable Git tag `benchmark-a22-submission-2026-10-30` at https://github.com/Sam-7878/dlg_gnn.

The public repository tag contains:
1. **Source Code & Pipelines:** Clean, modular Python scripts to reproduce all Friedman tests, Monte Carlo permutations, denominator audits, and figures.
2. **Canonical Run Registry:** Pinned registry (`canonical_selected_run_registry.csv`) containing 455 evaluated executions across seeds 42--46.
3. **Evaluation Population Ledger:** Seed-level ledger (`evaluation_population_by_seed.csv`) and programmatic audit (`evidence_denominator_audit.md`) verifying the exact model-input tensors, test masks, label counts, and test positive rates (prevalences) for all 14 datasets.
4. **Paired Delta Data:** Seed-paired delta CSV (`paired_ap_by_seed.csv` and `conditional_ap_14.csv`) reporting exact paired performance across 70 evaluations.
5. **Support Matrix Ledger:** Verified hardware support ledger (`support_14x7.csv`) documenting the 84 primary supported pairs, 7 external LANL pairs, and 7 unsupported primary pairs across three exact failure categories (`4 NO_APPROVED_RAW_RUN` for GADNR, `1 OBSERVED_HISTORICAL_CUDA_OOM` on 8-GB legacy environment for GADNR, and `2 OBSERVED_CUDA_OOM` in full encoder forward on 24-GB RTX 3090 for AnomalyDAE).

## 2. Truthful Node-Level Score Availability Disclosure

In accordance with our submission integrity audit:
- **Available Upon Reasonable Request:** High-resolution per-node anomaly score vectors ($y_{\text{score}}$) and binary label arrays ($y_{\text{true}}$) are author-locally preserved in compressed NumPy `.npz` format for all 105 cryptocurrency evaluations (Ethereum, BSC, and Polygon across 7 models and 5 seeds). Qualified academic researchers and peer reviewers may obtain these arrays by contacting the corresponding author (`kkim86@ajou.ac.kr`).
- **Historical Runs Score Limitation:** 350 historical baseline runs (280 primary, 35 BitcoinOTC, 35 LANL runs) originally stored summary metric scalars rather than raw node-level score arrays. These arrays were not preserved at runtime and cannot be retroactively manufactured without invalidating the frozen benchmark protocol.

