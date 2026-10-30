# Benchmark A24 evidence and reproduction

Current manuscript: A24; preserved A23 baseline tag `benchmark-a23-preprint-2026-10-30`, commit `94b0abbb4c89405da3f3c4fab7121f62bbe0b827`. **455 frozen successful records (420 primary + 35 external), 84/91 primary pairs, 13 primary + 1 external graphs, 7 configurations**. Public A24 supporting evidence: [benchmark-a24-supplementary-2026-10-30-r1](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1); use the resolved commit and asset hashes in [A24_REPRODUCIBILITY.md](A24_REPRODUCIBILITY.md).

A24 corrects S1 n/Q/W, uses sample SD (ddof=1), removes the unsupported historical diagnostic F1 column, narrows recommendations and corrects stale pre-A08 graph metadata. Three raw reconstructions match the actual A08 features/edges and 105 score bindings. A24 adds 60 feature-only evaluations, 25 matched Elliptic controls, 4 full resource cases, full-Cora numerical checks and 30 separate Base/Aug alert evaluations (10 reused controls, 20 fresh fits). These never enter primary ranks.

```bash
python scripts/reproduce_project.py --project benchmark --mode verify
python scripts/reproduce_project.py --project benchmark --mode tables
python scripts/reproduce_project.py --project benchmark --mode paper
```

Public verification uses standard Python. Numeric replay checks means/sample SD against public per-seed metrics; neural retraining is not implied. Public `paper` fails explicitly because the unpublished A24 manuscript writer/inputs are private. Author-local `paper` builds both PDFs with LaTeX; actual source ZIP recompile and page reviews are recorded separately. `full` is manual and guarded. [Data construction](DATASETS.md), [environment](ENVIRONMENT.md), [data availability](DATA_AVAILABILITY.md).

A05–A07's 80/91 and 435 records, the A08 replacement campaign, and A09–A23 reports remain historical evidence at their original paths. Select `--revision legacy-a07` or `--revision a08` to run those explicit historical facades. No old PDF is silently returned as A24. Earlier author approval/platform status does not constitute submission of the newly revised paper; no platform submission was executed by A24.
