# Benchmark local storage and binary inventory

The current reviewer checkout needs `projects/benchmark/evidence/frozen_a05_a06_evidence.zip`
(SHA-256 `459a3721efd1c0e17e49315ca31e92331ae754ae1683068303bdc793a5feab08`),
the project paper source, and the reproduction driver. It does not need the
multi-gigabyte local training workspace.

| Location | Approximate size at migration | Policy |
|---|---:|---|
| `outputs/benchmark/a05_anomalydae_large/checkpoints/` | 4.5 GB | Frozen local checkpoints; preserve, Git-ignore |
| `outputs/benchmark/a04_constructed_graphs/` | 3.8 GB | Constructed graphs; preserve, Git-ignore; see `DATASETS.md` for hashes/access |
| `outputs/benchmark/sci_defense_extension_real/` | 1.4 GB | Historical run workspace; Git-ignore |
| `outputs/benchmark/sci_round4c/checkpoints/` | 978 MB | Historical checkpoints; Git-ignore |
| `outputs/benchmark/sci_round5_final/checkpoints/` | 699 MB | Historical checkpoints; Git-ignore |
| `environment/wheelhouse-a04/` | 3.0 GB | Offline CUDA wheels; preserve locally, Git-ignore; exact wheel hashes are frozen in environment evidence |
| `local_storage/benchmark/backups/venv-before-rename.tar.gz` | 2,926,837,760 bytes | Pre-rename environment backup, SHA-256 `5ecd07e207e9a070410601e2ed9bd9e4fd85336ff350c197daf0f64ca862b094`; removed from current Git tree |
| `local_storage/benchmark/backups/A07_candidate_review_package.zip` | 24,132,132 bytes | Historical A07 candidate package, SHA-256 `562981c32263f76234e95eae6a2e9ead7a358cbf7dc7cd7cf2a7d05c7632876a`; Git-ignore |
| `local_storage/*/latex_cache/` | Small | Regenerable LaTeX intermediates; Git-ignore |

Removing the virtual-environment tarball from the **current tree** does not
erase its object from earlier Git history. Rewriting that history would change
existing commits and tags, so this migration does not do it. A shallow clone of
the eventual project tag avoids fetching unrelated earlier trees.
