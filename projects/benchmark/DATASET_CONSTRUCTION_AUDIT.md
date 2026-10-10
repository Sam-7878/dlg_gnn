# Dataset construction audit (2026-10-09)

## Frozen contract tensors: label-informed construction validity gate

**Historical snapshot:** the nine externally stored hybrid/knn/label files were deleted during the later migration. The following old-artifact checks describe their verified pre-deletion state and retained hashes, not current file availability. See the migration section at the end.

Safe inspection executes no pickle objects: `scripts/audit_hybrid_tensors.py` parses known torch ZIP storage and pickle opcodes, rejects unsupported layouts, and verifies the raw SHA-256 against the canonical manifest. All labels were compared with processed upstream CSV rows through the tensor’s contract-to-index JSON basenames; all 24,169 labels match. Twelve individual contract JSONs per chain were also checked.

| Dataset | Nodes | kNN edges | Hybrid edges | Added edges | Added edges sharing label |
|---|---:|---:|---:|---:|---:|
| Ethereum | 14,385 | 71,925 | 115,065 | 43,140 | 100% |
| BSC | 7,481 | 37,405 | 59,832 | 22,427 | 100% |
| Polygon | 2,303 | 11,515 | 18,411 | 6,896 | 100% |

Every kNN edge is retained. Feature columns 0, 1, 2, 4, 5, 6, 7 are constant zero; only column 3 varies. No feature equals the label vector. This does not establish label independence: the original hybrid builder, its inputs and the remaining feature’s definition are missing. Additional same-label edges are an observed fact, not a causal determination of how they were generated.

The local upstream `gog/dataset/process_graph_metrics.py` maps Category 0 to label 1, other categories to label 0; its dataset builder writes per-contract JSONs. This is a dataset-specific operational label, not independent validation of legal fraud status. Matching labels does not clear the graph-builder issue. Provider terms and access still need documentation.

The supplied `src/run_evaluation_pipeline.py` reads existing Polygon input and its `SemiSyntheticBuilder` copies x/edge/y to a different streaming dataset, or uses a simulated 5000×32 fallback. It cannot reconstruct the three original 8-feature tensors. Original kNN/label/hybrid producer evidence is required before using them as unsupervised fraud validation.

## Provider-native graphs

- **Elliptic:** PyG `EllipticBitcoinDataset`; remove unknown class 2 and take the induced graph of all known nodes (46,564). The shared repackager validates edge bounds, adds/coalesces self loops, converts features to float32 and maps non-finite feature values to zero. It does not automatically symmetrize edges. The benchmark is transductive on this labeled induced graph, not all 203,769 source nodes.
- **DGraphFin:** use `src/gog_fraud/data/dgraphfin_aligned.py`, not the older generic loader. Retain labels 0/1 and induce edges between retained nodes; preserve original node IDs, timestamps and optional edge types; remap official train/valid/test arrays and assert disjointness. The constructed graph has 1,225,601 known-label nodes and 746,271 edges; the manifest’s 367,702 evaluation nodes is the val/test population, not total graph size. Official masks do not imply an independently established temporal or inductive evaluation.

## Controlled injection on eight real graph substrates

`scripts/benchmark_8x10_pipeline.py` calls PyGOD contextual then structural generators with fixed dataset seed 42. Contextual count is `max(10, int(N*r_context))`, candidates k=50. Structural cliques count is `max(1, int(N*r_structure/m))`; injected labels are logical OR of the two generated vectors. Do not infer disjoint counts or post-coalescing edge increments from ratios; those intermediate counts were not archived.

| Graph | Context ratio | Structure ratio | Clique m |
|---|---:|---:|---:|
| Yelp | 0.01 | 0.01 | 8 |
| Amazon | 0.03 | 0.02 | 8 |
| BitcoinOTC | 0.03 | 0.03 | 8 |
| Reddit | 0.02 | 0.01 | 10 |
| Flickr | 0.02 | 0.02 | 8 |
| Cora / CiteSeer / PubMed | 0.03 | 0.03 | 10 |

These are archived loader settings, not a replacement for actual canonical tensor labels/counts. BitcoinOTC uses the last source snapshot, 64-dimensional Node2Vec plus degree/log-degree/normalized-degree features (67 columns); Node2Vec uses 100 epochs, LR .01, walk length 20, context 10, ten walks/node. The A05 canonical tensor supersedes older frozen BitcoinOTC results. Source and current implementation versions must be distinguished when recreating stochastic embeddings.

Full tensor/label hashes and actual counts remain in the immutable manifests. Statistical label-provenance sensitivity separates two provider-native graphs, three construction-pending contract artifacts and eight injected graphs. No frozen result or score was rewritten by this audit.

## Original labels.csv and global_graph comparison

The author identified these originals during review. Every tensor label matches `labels.csv` by chain/contract, Category 0 → positive, with zero missing contracts and zero mismatches. The original graph uses numeric contract IDs; the audit maps IDs to addresses, then addresses to tensor nodes, induces retained endpoints, and includes both edge orientations for a conservative undirected comparison. It ignores weights only for set overlap, not as a claim about the original task.

| Graph | Retained directed global edges (both orientations) | Hybrid overlap | Added hybrid overlap | Global same-label fraction |
|---|---:|---:|---:|---:|
| Ethereum | 3,036,618 | 4,032 | 1,267 / 43,140 | 91.1881% |
| BSC | 1,031,422 | 1,566 | 516 / 22,427 | 76.9256% |
| Polygon | 562,532 | 1,125 | 749 / 6,896 | 98.9387% |

Most added hybrid edges are absent from this address-aligned global edge union. Original global data availability therefore does not establish that the hybrid is the original interaction graph or resolve how added edges were produced. CSV and mapping hashes are recorded in the JSON audit. Raw provider data is not copied into the repository.

## Core pipeline call-path audit

The author identified `src/gog_fraud/pipelines/` as the dataset processing core. Static tracing confirms `run_fraud_benchmark.py → FraudDataset → label/transaction/global loaders`, plus `build_level2_graph` for embedding-kNN, temporal or shared-entity relations. Its relation builder attaches labels separately from edge construction. This hierarchy is distinct from the benchmark crypto campaign: `a03_run_crypto_production.load_gog_graph` directly reads existing `*_hybrid_graph.pt`, assigns stored embeddings to x, deduplicates stored edges and uses stored labels. The original ten-graph Round1 registry does not include the three crypto graphs. No frozen hybrid writer was found in these call paths. A public JSON audit records inspected source hashes; no claim is made that labels drive this relation builder.

## A03 manifest and archived working-tree patch: verified connection

The author identified `evaluation/benchmark/v2/manifests/datasets/dataset_manifests_a03.csv` and `evaluation/benchmark/v2/environment/legacy/20261002T145148Z/tracked-working-tree.patch`. These do connect the benchmark to the existing GoG inputs. In `a03_build_dataset_manifests.py`, `sha256_head` is the first 16 hexadecimal digits of **SHA-256 of the first 65,536 file bytes**, not a prefix of the full-file hash. All three current raw files match these recorded partial hashes and the A05 full hashes:

| Input | A03 head hash (verified) | A05 whole-file SHA-256 (verified) |
|---|---|---|
| Ethereum | fa87502ab0027800 | 4fa51d3e5fd09464f6ef7bb106376b365758e1e35d580c46c6cea65b4261c654 |
| BSC | f4f44bcb5c1336c0 | 2bd0a4943829241b84c694c1b301634e876e310bd63d415173a36de3ee510883 |
| Polygon | 58c222d510780252 | 292528febddc636ee2b5b09a76f7250e4473ff6d518d7977c49ff0beae7dad74 |

All 269 patch file sections were inspected. After normalizing CRLF in change hunks, 267 have identical removed/added content. The two remaining changes affect architecture and exact-reconstruction Markdown, not source code. Nine file sections mention `hybrid_graph`, comprising five experiment configurations, a statistics reader, a synthetic text-context generator and two TDS tests. The text-context generator reads the existing tensor labels and writes JSONL contexts; it does not write hybrid graph edges/features. Therefore the snapshot substantiates consumers and input paths but contains no semantic code change introducing a producer. It is a patch rather than a complete historical repository snapshot; absent/untracked code cannot be ruled out.

The crypto dimensions in this A03 manifest are read from the raw tensors. Several non-crypto fields are hardcoded source descriptions and are superseded by A04/A05 loader-observed evaluation manifests; the old CSV was preserved. Full file identities, per-patch-section classification and source references are recorded in `evidence/astra_revision/legacy_gog_connection_audit.json`. This establishes continuity of the supplied inputs, while the initial edge/feature construction remains unresolved.

## Previous GoG project and ngnn reuse

The author then identified `/mnt/d/_Work/goat_bank/gog/ngnn/train.py` and explained that earlier `.pt` files had likely been reused in later projects. Static tracing verifies `build_datasets → HierarchicalDataset → load_global_contract_graph`, which loads the configured existing Polygon hybrid tensor. `Trainer.save_checkpoint` writes `<run_name>_<tag>.pt` containing model/optimizer/scheduler state, epoch and validation metrics. It does not save the input tensor's `edge_index`, `embeddings` and `labels` schema. Thus this ngnn project is a consumer of the earlier graph and a producer of model checkpoints.

Ignored Python/notebook source files in the current GoG tree were searched too. The raw data's `Cryptocurrency-Graphs-of-graphs-main.zip` has 89 members including 45 Python/notebook/Markdown files; none names hybrid/knn/label graph artifacts. Its existing `.pt` writers save processed individual/link-prediction datasets. The `global_graph-20251111T131334Z-1-001.zip` has six data-only members and no code. Archive hashes and the ngnn call-path source hashes are recorded in `evidence/astra_revision/ngnn_input_reuse_audit.json`. These checks support input reuse while leaving the initial producer unlocated; they do not establish that all historical code has been recovered.

## Independent review of blockchain_dataset_audit.md

The supplied report adds sibling label_graph.pt artifacts. These were independently verified through non-executing metadata parsing and bounded full-edge scans. Their directed edge counts are Ethereum 106,208,628, BSC 41,877,464 and Polygon 5,032,346: exactly the complete same-label graphs, without self-loops or duplicates. Each family's feature bytes, labels and node-name order match. Hybrid new outgoing neighbors number 3 for all but 15 / 16 / 13 nodes respectively, which have 2; same-label kNN overlap candidates make a three-per-node union feasible. This strongly supports label-informed construction and keeps these inputs outside independently verified label-independent fraud claims.

The initial sampler/seed is unlocated and hidden overlap choices cannot be uniquely recovered. All artifacts record k=5 and method knn/label/hybrid; k=3 is inferred from constraints, not explicitly stored. Feature-column-3 tx_count is unverified; sampled current JSON edge/node counts differ. BSC's frozen maximum is 3227 rather than the reported 3012. The notebook's older polygon_hybrid.pt has 12,653 edges and a different schema.

See reports/astra_revision/GOG_Graph_Family_Review.md, evidence/astra_revision/gog_graph_family_audit.json and gog_source_writer_audit.json. This follow-up updates the scientific assessment while preserving prior frozen results and manuscript PDFs. Corrected primary inference needs label-independent reconstruction/re-evaluation or diagnostic exclusion of these three datasets. Finding an earlier label-based writer would not by itself clear validity.

## RelationBuilder migration: current final audit not passed

The three new Level2 files exist; all nine old graph-family files are absent. A read-only typed-storage audit confirms new edge counts 143,820 / 74,780 / 23,000, x dimension 9, graph y length 1 and node labels length N. Feature values exactly match the legacy features, whose normalized vectors all equal e_3; cosine similarities are all 1. The inspected edge function passes a fixed-feature label counterfactual CPU check, but feature provenance and informative relations remain unresolved. Current A03 loaders return the scalar-y Data unchanged, violating node-evaluation label shape. Repeating the generation script would select that scalar y rather than N labels.

The generator requires an existing graph tensor; it does not reconstruct the feature input from provider CSV/JSON. Archived paper canonical hashes and metrics still identify the old hybrid inputs. New topology and a changed source path do not make archived results into a new evaluation. See `reports/astra_revision/RelationBuilder_Migration_Final_Audit.md` and `evidence/astra_revision/relation_builder_migration_audit.json`. No input rewrite, training or paper rebuild occurred during this audit.
