# Project Artifact Layout

`dlg_gnn` repository organizes research assets into a **Two-Tier System**:
1. **Tier 1 (`projects/<project>/`)**: Active, reviewer-facing portal containing only the latest paper, canonical reports, verified results/evidence, and reproduction facades.
2. **Tier 2 (`archive/<project>/`)**: Historical archive containing developmental rounds, previous evaluation data, legacy reports, and earlier manuscript drafts.
3. **Core Library (`src/`)**: Reusable shared scientific packages (`gog_fraud/`, `analysis/`) without circular project dependencies.

| Previous location | Owner and role | Reorganized location / disposition |
|---|---|---|
| `docs/work_reports/benchmark/229_claude_review/submission_a07/` | Benchmark A07 candidate paper | `projects/benchmark/paper/current/` |
| `docs/work_reports/benchmark/228_A06/`, `229_claude_review/` | Benchmark work orders, reviews, rollback and A06 frozen submission | Final A06/A07 reports: `projects/benchmark/reports/`; frozen submission zips & rollback: `archive/benchmark/submissions/` |
| `docs/papers/_42_01_Benchmark_PrePrints/`, `_42_Benchmark/` | Benchmark A05 and older historical sources | `archive/benchmark/papers/` |
| `docs/papers/_40_DLG_GNN/` | Published preceding DLG paper, frozen original | Original review copy: `projects/dlg_gnn/paper/published_v1/`; historical draft: `archive/dlg_gnn/papers/_40_DLG_GNN/` |
| `manuscript/_41_01_DLG_StreamMC/`, `docs/papers/_41_01_Stream/` | StreamMC working and historical papers | Active manuscript: `projects/stream_mc/paper/current/`; historical drafts: `projects/stream_mc/paper/history/` |
| `docs/papers/_43_01_TDS/` | TDS working manuscript and venue wrappers | Active manuscript: `projects/tds/paper/current/` |
| `reports/dlg_gnn/`, `results/dlg_gnn/`, `figures/dlg_gnn/` | Preceding DLG reports, tables, metrics, figures | Active: `projects/dlg_gnn/reports/`, `results/`, `figures/`; old runs: `archive/dlg_gnn/outputs/` |
| `reports/stream_mc/round_4/`, `results/stream_mc/sci_v3_submission_r4/` | StreamMC canonical R4 evidence & verification | Active: `projects/stream_mc/reports/`, `results/canonical_r4/`; older rounds R1~R3: `archive/stream_mc/results/` |
| `reports/tds/main_final*`, `results/tds/main_final_v2/` | TDS Gate v8 results, model metrics, final reports | Active: `projects/tds/reports/`, `results/main_final_v2/`; developmental rounds: `archive/tds/reports/`, `results/` |
| `archive/gog_scimain_v1_preserved_panel/`, `data/benchmark/gog_scimain_v1/` | TDS historical GoG-SCIMain-v1 dataset & panel | `archive/tds/data/` |
| `evaluation/benchmark/`, `experiments/benchmark/`, `artifacts/` | Frozen benchmark campaign source & v1.0.0 artifacts | `projects/benchmark/evidence/frozen_a05_a06_evidence.zip` (portable reviewer evidence); historical: `archive/benchmark/artifacts/`, `experiments/` |
| `publication/benchmark/`, `release/dlg_gnn_benchmark/` | Historical v1.0.0 publication and release workspaces | `archive/benchmark/publication/`, `archive/benchmark/release/` |
| `outputs/` | Local raw/constructed graphs, checkpoints | `archive/<project>/outputs/` (Git-ignored) |
| `logs/` | Benchmark execution logs (`benchmark.log`) | `archive/benchmark/logs/` (Git-ignored) |
| `provenance/` | Benchmark historical environment manifests | `archive/benchmark/provenance/` |
| `environment/wheelhouse-a04/` | Offline installation wheels | Local Git-ignored wheel cache; exact hashes/locks in `environment/locks/` and A05 evidence |
| `evaluation/benchmark/v2/environment/legacy/.../venv-before-rename.tar.gz` | Pre-rename virtual-environment backup | `local_storage/benchmark/backups/` (Git-ignored) |

The repository does not treat intermediate `.aux`, `.log`, checkpoints, wheels, or
environment backups as active paper results. All four projects feature deterministic
reproduction commands via `scripts/reproduce_project.py` with zero reliance on legacy paths.

