# DLG-GNN Research Repository

Current benchmark paper: **A24 manuscript**; preserved source baseline: A23. The frozen benchmark comprises **13 primary graphs + 1 external graph, 7 primary configurations, 84/91 supported primary pairs, 420 primary + 35 external = 455 successful runs**. CONAD remains diagnostic only. New A24 experiments are separate supplementary campaigns and do not change these counts or primary ranks.

Scientific supporting evidence is identified by annotated tag [`benchmark-a24-supplementary-2026-10-30-r1`](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1); the resolved commit, asset hashes and fresh-clone checks are recorded in [A24 reproducibility](projects/benchmark/A24_REPRODUCIBILITY.md). A Git tag is a reference; pin the recorded commit and checksums. No benchmark DOI or platform submission is claimed.

The shared implementations live in `src/`; project-specific evidence and commands live in `projects/`. Raw datasets, new per-node scores/checkpoints, supplementary wheels and unpublished A24 manuscript sources/PDFs/writer stay in ignored author-local storage.

| Project | Purpose | Verification |
|---|---|---|
| [DLG-GNN](projects/dlg_gnn/README.md) | Preceding architecture [preprint](https://doi.org/10.20944/preprints202609.0848.v1) | `python scripts/reproduce_project.py --project dlg_gnn --mode verify` |
| [Benchmark](projects/benchmark/README.md) | Graph anomaly predictive quality and exact execution | `python scripts/reproduce_project.py --project benchmark --mode verify` |
| [StreamMC](projects/stream_mc/README.md) | Streaming AML, separate paper validation | `python scripts/reproduce_project.py --project stream_mc --mode verify` |
| [TDS](projects/tds/README.md) | Transaction decomposition and micro-RAG | `python scripts/reproduce_project.py --project tds --mode verify` |

## Benchmark reproduction

```bash
python scripts/reproduce_project.py --project benchmark --mode verify
python scripts/reproduce_project.py --project benchmark --mode tables
```

`verify` checks public hashes, frozen support and run identities, sample-SD aggregation and supplementary contracts without raw data or GPU. `tables` independently recomputes current scalar tables from public per-seed metrics; it does not rerun training. `paper` requires author-local A24 inputs and explicitly refuses a historical fallback. `full` starts no training automatically. See [data](projects/benchmark/DATASETS.md), [environment](projects/benchmark/ENVIRONMENT.md) and [installation](INSTALL.md).

The corrected blockchain inputs were rebuilt from source histories: eight label-independent features and binary symmetrized Euclidean k=5 similarity, with **no transfer-edge union**. A24 corrects A23's stale graph dimensions without replacing its metric values. See [actual input/score binding](projects/benchmark/publication/a24/a24_repo_manuscript_consistency.json).

## Historical releases

A05–A07 had 80/91 supported pairs and 435 successful records; those are historical, not current counts. A08 repaired the three blockchain inputs. A13–A16 added exact supported executions; A23 is the preserved 455-run source baseline. Earlier releases and reports remain at their original paths. The previous hybrid tensors are excluded from corrected inference, and their old construction limitations do not describe the A08 score-bound inputs. Code is MIT licensed; third-party datasets retain provider terms.
