# GoG-SCIMain-v1 Original Source Lineage

## Decision-relevant result

The reproducible raw training corpus has been reacquired. On 2026-09-03, all 10 files exposed by
the upstream distribution were downloaded, and all 24,316 transaction CSV byte streams were checked
against pre-existing frozen per-file SHA-256 records. The result was 24,316 exact matches, zero
missing files, zero unexpected files, and zero hash mismatches. This establishes source identity; it
does not by itself establish exact identity of the four packed GoG-SCIMain-v1 artifacts. Those are
accepted only if their separate frozen SHA-256 contract passes after preprocessing.

## Upstream source

- Paper: *Multi-Chain Graphs of Graphs: A New Approach to Analyzing Blockchain Datasets*, NeurIPS
  2024 Datasets and Benchmarks Track, DOI `10.52202/079017-0894`.
- Official repository: `https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs`.
- Frozen repository commit: `7264f1bf510f7ba4f5041ac7a29b606abc12f262` (main, 2024-10-29).
- Dataset folder: `https://drive.google.com/drive/folders/1VV5ht9Eh8WGtKfkS0ipIk0FNI7g-WJfJ`.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0), as stated
  in the upstream README and full `LICENSE` file.
- Upstream versioning limitation: the Drive distribution has no release tag or immutable dataset
  version identifier. We therefore bind this reacquisition to the repository commit, Drive file IDs,
  byte sizes, archive SHA-256 values, and extracted per-file SHA-256 values.

The local source-code archive
`/mnt/d/_Work/_data/GoG/Cryptocurrency-Graphs-of-graphs-main.zip` has SHA-256
`ad28954b0aca9c7ef3a83ae2d679b152375e0444348730c3fcdfd54bc912cf82`. Its 69 files are byte-for-byte
identical to the tree at upstream commit `7264f1bf510f7ba4f5041ac7a29b606abc12f262`.

## Download dates

- Original historical download date: not recorded by a trustworthy download log.
- Earliest retained local file timestamp: 2025-11-11 for the repository ZIP and `labels.csv`; this is
  filesystem metadata, not proof of the remote acquisition date.
- SCI v2 derivative generation records: BSC `2026-07-29T09:19:28.509649+00:00`; Polygon
  `2026-07-29T09:43:04.038154+00:00`. The Ethereum v2 manifest body was not retained in the Round 3
  package, although its frozen hash was retained.
- Audited external reacquisition: 2026-09-03 (Asia/Seoul workspace date).

## Chain and collection lineage

The upstream corpus contains 14,464 Ethereum, 7,499 BSC, and 2,353 Polygon token contracts. The
upstream README reports 81,788,211, 121,612,480, and 64,882,233 token-transfer rows respectively,
covering data through February 2024. `dataset/data_collection_script.py` identifies the collection
interfaces as Etherscan, BscScan, and Polygonscan token-transaction APIs. The downloaded archives,
rather than a fresh explorer crawl, are the canonical source for this recovery because current API
responses would not be guaranteed to reproduce the published bytes.

The label source is upstream `labels.csv`. The upstream README explicitly states that Category 0 is
fraud. The frozen binary mapping is therefore Category 0 -> fraud/positive (1), every non-zero
category -> benign/negative (0).

## Reacquired artifact identity

| Artifact | Drive file ID | Bytes | SHA-256 |
|---|---:|---:|---|
| `transactions/ethereum.zip` | `13wgfMbvdcpiwyM5GEQBcWfDviW_Eseta` | 4,484,986,142 | `6d1f6797475ecfb3b2edcd5eedbcfd8708c4808486d0afb88634d4951636c5d2` |
| `transactions/bsc.zip` | `1P97qDEBaWmZfs8DldXo-oNBtVpShCXzY` | 5,671,092,781 | `4c13344a6a7833a22576d7e1db89e0a55b50966fda499fd2e84f3bfe525177d1` |
| `transactions/polygon.zip` | `1OP7vC51RMFSmMp9dv1Mimb0aDcq1Nfv6` | 2,805,020,440 | `d4156b2cb85f92717df6734b205c69ca4da2ebe687999791a10f03b97d46f86e` |
| `labels.csv` | `1GggOxrUKHl_-2HaRlcVe9tiQPtJZWPR6` | 1,306,970 | `1356d063d675369984ef76277c4601e9f4fc780a06435c2e36a821e00ab1ea4a` |

The six global-graph/mapping artifacts and their hashes are recorded in
`results/main_final_v2/upstream_reacquisition_manifest.json`. They match the retained copies under
`/mnt/d/_Work/_data/GoG/global_graph` byte for byte.

Extracted source audit:

| Chain | Expected | Found | Missing | Unexpected | Hash mismatch | Exact |
|---|---:|---:|---:|---:|---:|---|
| Ethereum | 14,464 | 14,464 | 0 | 0 | 0 | yes |
| BSC | 7,499 | 7,499 | 0 | 0 | 0 | yes |
| Polygon | 2,353 | 2,353 | 0 | 0 | 0 | yes |

Ethereum expected hashes came from the complete Round 2 source manifest. BSC and Polygon expected
hashes came from the Round 3 SCI v2 manifests; those manifest byte hashes exactly equal the frozen
upstream hashes `edcf3890...` and `d8948abd...`. This use of older evidence is explicit because the
32.7 MB Ethereum Round 3 manifest body was omitted from its compact evidence ZIP.

## Reacquisition command

```bash
/mnt/d/_Work/goat_bank/.venv/bin/gdown --folder --continue \
  'https://drive.google.com/drive/folders/1VV5ht9Eh8WGtKfkS0ipIk0FNI7g-WJfJ?usp=share_link' \
  -O /mnt/d/_Work/_data/gog_round7_upstream/
```

The exact Drive file IDs and expected byte sizes are frozen in
`experiments/round7/upstream.py`. `experiments/round7/reacquisition.py` records archive hashes and
checks every extracted CSV against historical hashes.

## Preprocessing lineage

- Repository preprocessing commit: `a198438f099656adcfe673bce92596894e9a0abf`.
- Commit date: 2026-08-29T20:25:36+09:00.
- `scripts/build_sci_dataset_v2.py` SHA-256:
  `4ea125a00daacf4b6f1f3aeaf3ff63411b4360f780006d55265c1687484bbb94`.
- `src/gog_fraud/data/sci_v2/builder.py` SHA-256:
  `c2d724a651b947347267668f24ed01fcf14164d7bebbf93af79ab71215207447`.
- `src/gog_fraud/data/sci_v2/audit.py` SHA-256:
  `fb89096981a5a70c304e4a4686b2810c0abd07392c640a17272046ea598e4e93`.

These three hashes exactly match the Round 3 evidence index. The builder command is:

```bash
/mnt/d/_Work/goat_bank/.venv/bin/python scripts/build_sci_dataset_v2.py \
  --raw-root /mnt/d/_Work/_data/dataset/transactions \
  --legacy-root /mnt/d/_Work/_data/GoG \
  --output-root /mnt/d/_Work/_data/GoG_sci_v2 \
  --chains ethereum bsc polygon \
  --strict
```

The packed main-track builder is then run with the original logical paths:

```bash
/mnt/d/_Work/goat_bank/.venv/bin/python src/data_generation/build_gog_chronological_dataset.py \
  --source-root /mnt/d/_Work/_data/GoG_sci_v2 \
  --output-root data/benchmark/gog_scimain_v1 \
  --max-edges 128
```

## Environment

- WSL2 Ubuntu 24.04.4 LTS (the Ubuntu 20.04 legacy distribution is not used).
- Project interpreter: `/mnt/d/_Work/goat_bank/.venv/bin/python`, Python 3.12.13.
- WSL memory allocation: 20 GB.
- Raw and derivative bytes are physically stored on Ubuntu ext4 with explicit symlinks at the
  original `/mnt/d/_Work/_data/dataset` and `/mnt/d/_Work/_data/GoG_sci_v2` logical paths. This
  preserves path-sensitive metadata while avoiding D: capacity exhaustion.

## Acceptance boundary

Raw-source recovery and exact GoG-SCIMain-v1 recovery are both **PASS** (Branch A). The strict SCI v2 audit checked all 24,316 records with zero violations and zero incomplete checks. The packed artifacts match all four frozen SHA-256 values:

| Artifact | Frozen and reproduced SHA-256 |
|---|---|
| `graph.pt` | `067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd` |
| `transactions.parquet` | `4d240fe8d5488f6f27fd1d475d039abfc96aa40aa6dc34d34a75dfa92be3df3d` |
| `split_manifest.json` | `7f388c5163293f5706cb07427747b1fc6988ae749c8d1a0e104e79b6d83accfa` |
| `future_edge_audit.csv` | `395cc4fe3c0c2198fbb25368f9cf843bd9de4352efcbcba8a7d8e77fd5e43f7f` |

The first packed rerun differed only in `transactions.parquet`: all 24,316 values of the audit column `original_source_available` were true because the reacquired raw-source symlink was present, whereas the historical build recorded false after raw-source removal. Temporarily hiding only that symlink reproduced the original environmental precondition and the frozen Parquet hash; the symlink was immediately restored to `/home/sam/dlg_gnn_round7_data/dataset`. Graph, split, and future-edge-audit hashes were exact in both reruns. This distinction is recorded because it affects byte identity but not model inputs or labels.

Exact recovery authorized same-panel baseline completion. TGN-style, TGAT-style, and fraud-oriented GraphSAGE were subsequently trained with five isolated seeds, and temperature scaling was fitted on validation predictions only. No GoG-SCIMain-v2 branch was activated.

