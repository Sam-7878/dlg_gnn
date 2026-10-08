# Benchmark v2 reproduction

This project is the A05/A06 frozen, A07 manuscript-facing graph anomaly benchmark. It contains **13 primary datasets**, seven functioning inferential models, **80/91** supported primary pairs and **435** approved primary/external successes. CONAD remains diagnostic only. BitcoinOTC uses controlled node anomaly injection on a real trust graph.

## Three levels

1. `./projects/benchmark/reproduce.sh verify` checks all 645 files in the bundled frozen A05/A06 evidence ZIP, the 13×7 support and 435-run counts, archived exact sparse/fused/AnomalyDAE equivalence, CONAD policy, and a tiny independent Gram identity. No third-party raw graph is downloaded.
2. `./projects/benchmark/reproduce.sh paper` performs Level 1 and regenerates the A07 manuscript input, references, PR/ROC/F1 source tables, pairwise statistics, measured-memory figure copy, claims map, alert-budget CSV, and a separate F1 supplementary TeX file from frozen approved evidence. Requires SciPy for Wilcoxon and LaTeX for optional PDF compilation. Output: `projects/benchmark/paper/current/`.
3. `./projects/benchmark/reproduce.sh full` is an explicit guarded entry point. Full training remains manual campaign orchestration and needs licensed source data, GPUs, and the historical/current environment specified for each campaign. The guard deliberately exits without training.

Use `.venv_cuda` for local scientific regeneration: `/mnt/d/_work/goat_bank/.venv_cuda/bin/python scripts/reproduce_project.py --project benchmark --mode paper`. CI integrity checks use standard Python and do not claim to rerun neural models. The archived equivalence reports were produced in the recorded scientific environment. Source-code compatibility can be checked separately with `pytest tests/benchmark/pygod_integration/` in that environment.

The bundled ZIP is the immutable frozen evidence. Its manifest lists every payload hash. Data acquisition and graph construction hashes are in [DATASETS.md](DATASETS.md); environment provenance is in [ENVIRONMENT.md](ENVIRONMENT.md). The repository's historical `v1.0.0-preprint` release describes an earlier ten-dataset version and must not be used as the current benchmark identity.

Reviewer navigation: [A07 paper](paper/README.md), [final A06/A07 reports](reports/a07/A07_Execution_Report.md), [local binary storage policy](STORAGE.md), and [fresh checkout test](FRESH_CLONE_VERIFICATION.md). The A07-only manuscript and consistency scripts live in `scripts/`; frozen A05/A06 source snapshots and older campaign runners retain their original paths.
