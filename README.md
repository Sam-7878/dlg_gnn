# DLG-GNN Research Repository

Shared research code and four paper-oriented project facades. The model/data/statistics implementations remain in `src/`; `projects/` provides reviewer-facing reproduction commands without duplicating scientific code.

Current papers, curated evidence and final reports are indexed by project in
[the artifact layout](projects/ARTIFACT_LAYOUT.md). Large raw runs, checkpoints,
offline wheels and backups stay in Git-ignored local storage. Historical frozen
benchmark sources retain their original paths.

| Project | Purpose | Status | Fast verification |
|---|---|---|---|
| [`dlg_gnn`](projects/dlg_gnn/README.md) | DLG model and exact execution | Preceding architecture [preprint](https://doi.org/10.20944/preprints202609.0848.v1) | `python scripts/reproduce_project.py --project dlg_gnn --mode verify` |
| [`benchmark`](projects/benchmark/README.md) | Support-aware graph anomaly benchmark | A07 manuscript candidate; public release pending | `python scripts/reproduce_project.py --project benchmark --mode verify` |
| [`stream_mc`](projects/stream_mc/README.md) | Streaming AML | Development; paper validation separate | `python scripts/reproduce_project.py --project stream_mc --mode verify` |
| [`tds`](projects/tds/README.md) | Transaction decomposition and micro-RAG | Development; paper validation separate | `python scripts/reproduce_project.py --project tds --mode verify` |

## Current benchmark

The frozen A05/A06 evidence covers **13 primary graphs plus LANL-RedTeam external validation**, **seven functioning primary inferential models**, **80/91 supported primary pairs**, and **435 approved primary/external successful records**. CONAD is diagnostic only following its loss-path audit. The S3 financial/blockchain view has five source/native-label graphs; BitcoinOTC has controlled node anomaly injection on a real trust graph. Historical `v1.0.0-preprint` is a ten-dataset release and remains separate.

```bash
python scripts/reproduce_project.py --project benchmark --mode verify
python scripts/reproduce_project.py --project benchmark --mode paper
```

`verify` checks the bundled frozen evidence, model-support counts and archived numerical-equivalence reports without third-party raw data or GPU. `paper` regenerates manuscript inputs and needs SciPy. `full` is guarded and requires manual campaign orchestration, source data and GPUs. See [benchmark data acquisition](projects/benchmark/DATASETS.md), [environment provenance](projects/benchmark/ENVIRONMENT.md), and [installation](INSTALL.md).

The [A07 benchmark manuscript](projects/benchmark/paper/README.md),
[preceding DLG paper](projects/dlg_gnn/paper/README.md),
[StreamMC manuscript](projects/stream_mc/paper/README.md), and
[TDS manuscript](projects/tds/paper/README.md) each have a single project
landing page. The benchmark candidate PDFs are not yet submission artifacts.

Code is MIT licensed ([LICENSE](LICENSE)). Third-party data licenses and distribution conditions remain with their providers. No A07 journal submission or preprint DOI is claimed here.
