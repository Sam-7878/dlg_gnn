# Round 7 Reacquisition Decision

## Decision

**Branch A — exact GoG-SCIMain-v1 recovery: PASS.**

The official Graph-of-Graphs files were reacquired from the upstream Google Drive distribution.
All 24,316 extracted transaction files match retained historical per-file hashes. The original SCI
v2 builder completed all three chains without failure, and the independent strict audit returned:

```text
records_checked    24,316
violations         0
incomplete_checks  0
paper_eligible     true
status             PASS
```

The packed benchmark reproduced all frozen hashes:

| Artifact | SHA-256 |
|---|---|
| `graph.pt` | `067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd` |
| `transactions.parquet` | `4d240fe8d5488f6f27fd1d475d039abfc96aa40aa6dc34d34a75dfa92be3df3d` |
| `split_manifest.json` | `7f388c5163293f5706cb07427747b1fc6988ae749c8d1a0e104e79b6d83accfa` |
| `future_edge_audit.csv` | `395cc4fe3c0c2198fbb25368f9cf843bd9de4352efcbcba8a7d8e77fd5e43f7f` |

The first Parquet rerun differed only because the reacquired raw-source path made the audit column
`original_source_available` true. Reproducing the historical post-derivation condition (raw-source
symlink temporarily hidden) restored the frozen byte hash. The symlink was immediately restored;
the physical 49 GB source tree was never moved or deleted. Graph, split, and future-edge-audit bytes
were already exact.

The conditional GoG-SCIMain-v2 protocol remains an unused preregistration artifact. No old-v1/new-v2
result mixing occurred.

## Primary sources

- NeurIPS 2024 paper: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/3205b048f9cc54b9f7963db0b0f52d53-Abstract-Datasets_and_Benchmarks_Track.html>
- Official repository: <https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs>
- Official dataset folder: <https://drive.google.com/drive/folders/1VV5ht9Eh8WGtKfkS0ipIk0FNI7g-WJfJ>
- Repository commit: `7264f1bf510f7ba4f5041ac7a29b606abc12f262`
- License: CC BY-NC-SA 4.0

