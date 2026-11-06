# A27 Evidence and Reviewer Access

The public scientific release and commit (`9e7e3a940d7e754b497043cf3959557c82773928`, tag `benchmark-a24-supplementary-2026-10-30-r1`) are unchanged. The selected registry has 455 successful runs (420 primary and 35 external) and 84 of 91 complete primary pairs. Public `verify` and `tables` remain the integrity and numeric replay entrypoints for those results. A27 adds the complete orientation, population, exact-tail, and hardware measurement audits here. [Scientific release](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1)

## Per-Node Score Archive and Supplementary Vector Breakdown

| Collection | Retained arrays | Coverage |
|---|---:|---|
| Primary contract graphs | 105 | 35 each for Ethereum, BSC, and Polygon (7 models × 5 seeds) |
| Earlier BitcoinOTC reruns | 35 | All 7 models × 5 seeds (`ARCHIVED_A05_SCORE`) |
| Earlier DGraphFin AnomalyDAE reruns | 5 | 5 seeds (`ARCHIVED_A05_SCORE`) |
| Matched supplementary studies and resource checks (`local_storage/benchmark/a24/runs/`) | 113 | Separate from the 455 selected primary/external runs |
| Direct A27 RTX 3090 full-training measurement (`Reddit-Syn / DLG-Aug`) | 0 | Seed 42 timing and memory summary (`reddit_dlg_aug_measured.json`) |

### Breakdown of the 113 Supplementary Vector Sets (`local_storage/benchmark/a24/runs/`)

| Study Tier (`study_tier`) | Runs | Datasets × Models × Seeds |
|---|---:|---|
| `A24_SUPPLEMENTARY_FEATURE_ONLY` | 60 | 6 datasets (`Elliptic`, `Ethereum`, `BSC`, `Polygon`, `DGraphFin`, `LANL-RedTeam`) × 2 estimators (`IForest`, `ECOD`) × 5 seeds |
| `A24_CANONICAL_CONTROL` | 25 | `Elliptic` × 5 DLG variants (Base, Base-70, Aug, Aug-Permuted, Aug-Zero) × 5 seeds |
| `A24_ALERT_REPRODUCTION` | 20 | 2 datasets (`DGraphFin`, `LANL-RedTeam`) × 2 DLG detectors (`DLG-Base`, `DLG-Aug`) × 5 seeds |
| `A24_RESOURCE_FULL_DECLARED` | 4 | Four cases (Cora-Syn/DOMINANT, DGraphFin/DOMINANT, Ethereum/DLG-Aug, Reddit-Syn/DLG-Base) × seed 42 (full declared epochs) |
| `A24_RESOURCE_ONE_EPOCH_PILOT` | 4 | Four cases (Cora-Syn/DOMINANT, DGraphFin/DOMINANT, Ethereum/DLG-Aug, Reddit-Syn/DLG-Base) × seed 42 (1-epoch pilot) |
| **Total** | **113** | Verified in `provenance.json` (`supplementary_113_breakdown`) |

### Verification of the 310 Runs Without Retained Per-Node Arrays

The selected registry records `MISSING_NOT_STORED_BY_SOURCE_RUN` for 310 runs across 10 datasets (`275` primary runs in `archive/benchmark/outputs/sci_round5_final/raw` and `35` historical `LANL-RedTeam` runs in `archive/benchmark/outputs/sci_defense_extension_real/benchmark/raw`). `provenance.py` verified all 310 scalar JSON records on disk, confirmed that none contain `.npy`/`.npz` score file paths, and verified that both source directories contain `0` `.np*` files (`missing_310_audit` in `provenance.json`).

For controlled access, request the dataset, model, seed, and artifact from the corresponding author, Ki-Hyung Kim (kkim86@ajou.ac.kr), and explain the verification purpose. Anonymous reviewers can route the request through the handling editor.

The six public audit files were downloaded and compared byte for byte against the local files at the pinned A27 commit. Local feature and score arrays are subject to author approval and provider terms. [Pinned audit directory](https://github.com/Sam-7878/dlg_gnn/tree/71ffdbf9f7c75fc16163d17fd787f509dd6e2923/projects/benchmark/publication/a27)

## A28 access policy

The current manuscript keeps node-level artifacts internal. See the [A28 redistribution scope](../a28/SCORE_REDISTRIBUTION_SCOPE.md).
