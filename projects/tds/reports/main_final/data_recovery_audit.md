# Frozen Data Recovery Audit

| Artifact | Expected SHA-256 | Found path | Actual SHA-256 | Status | Recovery source |
|---|---|---|---|---|---|
| real_dataset_manifest.json | `5758f1cf36d4a82019686b0612eb6c3a24f0e585cfbf70d959923b3831003a2c` | `/mnt/d/_Work/goat_bank/dlg_gnn/data/benchmark/gog_scimain_v1/real_dataset_manifest.json` | `5758f1cf36d4a82019686b0612eb6c3a24f0e585cfbf70d959923b3831003a2c` | EXACT_HASH_MATCH | expected frozen path |
| graph.pt | `067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd` | not found | n/a | MISSING | not found |
| transactions.parquet | `4d240fe8d5488f6f27fd1d475d039abfc96aa40aa6dc34d34a75dfa92be3df3d` | not found | n/a | MISSING | not found |
| split_manifest.json | `7f388c5163293f5706cb07427747b1fc6988ae749c8d1a0e104e79b6d83accfa` | not found | n/a | MISSING | not found |
| future_edge_audit.csv | `395cc4fe3c0c2198fbb25368f9cf843bd9de4352efcbcba8a7d8e77fd5e43f7f` | not found | n/a | MISSING | not found |
| upstream:ethereum | `1efed3a8977f56cc30bd79c97f95eee57b825fe25971e7f72b2e0a93402ae2be` | not found | n/a | MISSING | not found |
| upstream:bsc | `edcf3890377a985b8c03da68fda5237590e878672c2d6ac31b784c1ac1ef60b7` | not found | n/a | MISSING | not found |
| upstream:polygon | `d8948abd672a1abe68ae11229b4d482c8d8415ae3cfddff76d48cb776d63ce0c` | not found | n/a | MISSING | not found |

## Decision

Exact GoG-SCIMain-v1 recovery: **False**. Exact upstream derivative recovery:
**False**. Future-edge audit verified: **False**.

The expected hashes were searched before any rebuild. The constrained SHA-256 pass checked 5,418
artifact-shaped candidates totaling 6,713,820,195 bytes; only the two preserved manifest copies matched. No missing artifact was reconstructed from
summary statistics or its digest. Because neither exact v1 evidence nor a provenance-complete new
version is available, Option B3 applies and Gate M remains closed.
