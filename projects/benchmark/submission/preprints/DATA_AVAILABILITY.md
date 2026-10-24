# Data and Reproduction Availability Statement (A14 Preprints.org)

**Document ID:** DLG-BENCH-DATA-PREPRINT-A14  
**Date:** 2026-10-24  
**Target Platform:** Preprints.org  
**Benchmark Release Tag:** `benchmark-a14-submission-2026-10-24`  
**Repository URL:** https://github.com/Sam-7878/dlg_gnn  

---

## 1. Summary of Publicly Available Artifacts

All benchmark evaluation scripts, aggregation pipelines, mathematical validation notebooks, and figure generation routines are openly archived under immutable GitHub release tag `benchmark-a14-submission-2026-10-24`.

The public release contains:
1. **Source Code & Pipelines:** Clean, modular Python scripts to reproduce all Friedman tests, Monte Carlo permutations, and figures.
2. **Canonical Run Registry:** Pinned registry (`canonical_selected_run_registry.csv`) containing 455 evaluated executions across seeds 42--46.
3. **Paired Delta Data:** Seed-paired delta CSV (`paired_ap_by_seed.csv` and `conditional_ap_14.csv`) reporting exact paired performance and standard error bars across 70 evaluations.
4. **Dataset Reconciliation Manifests:** Full reconciliation table separating evaluated model-input graph tensors ($N, E, d$) from upstream raw repository totals.

## 2. Truthful Node-Level Score Availability Disclosure

In accordance with our submission integrity audit:
- **Available Upon Reasonable Request:** High-resolution per-node anomaly score vectors ($y_{\text{score}}$) and binary label arrays ($y_{\text{true}}$) are author-locally preserved in compressed NumPy `.npz` format for all 105 cryptocurrency evaluations (Ethereum, BSC, and Polygon across 7 models and 5 seeds). Qualified academic researchers and peer reviewers may obtain these arrays by contacting the corresponding author (`kkim86@ajou.ac.kr`).
- **Historical Runs Score Limitation:** 350 historical baseline runs (280 primary, 35 BitcoinOTC, 35 LANL runs) originally stored summary metric scalars rather than raw node-level score arrays. These arrays were not preserved at runtime and cannot be retroactively manufactured without invalidating the frozen benchmark protocol.
