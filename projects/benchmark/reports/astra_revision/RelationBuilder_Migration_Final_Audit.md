# RelationBuilder migration final audit — 2026-10-09

## Decision

**FINAL SCIENTIFIC AUDIT: NOT PASSED.** All nine old external graph files are absent and the three new Level2 files exist. Direct label-driven edge additions are removed from the inspected embedding_knn call. However, the current node-label interface is invalid, legacy features remain unexplained and cosine-degenerate, provider-to-feature rebuilding is missing, and the paper still uses the old frozen input/results. The supplied report is evidence reviewed here; its commands and conclusions were not adopted as instructions or proof.

## Observed artifacts

| Dataset | Nodes | New edges | New x shape | New y shape | Node labels in labels | Old paper edges |
|---|---:|---:|---|---|---|---:|
| Ethereum | 14,385 | 143,820 | [14385, 9] | [1] | [14385] | 115,065 |
| BSC | 7,481 | 74,780 | [7481, 9] | [1] | [7481] | 59,832 |
| Polygon | 2,303 | 23,000 | [2303, 9] | [1] | [2303] | 18,411 |

New file hashes, deleted-file checks, source hashes and CPU counterfactual results are in `../../evidence/astra_revision/relation_builder_migration_audit.json`. No raw pickle objects were executed. Known typed torch ZIP storages were parsed, and trusted repository APIs were called on small CPU arrays.

## What passes

For each chain, a 128-node fixture made from its actual embedding was passed to production build_level2_graph twice: labels all zero and labels all one. Edge indices, weights and x remained identical; only the graph-level target changed. Static tracing confirms that embedding_knn does not read label when building edges. This establishes the inspected builder property for fixed input embeddings, not upstream feature independence or an updated manuscript evaluation. Cross-class edge percentages alone would not establish label independence.

## Blocking findings

### 1. Graph-level y is used as node-level y

`build_level2_graph` intentionally derives a single graph-level y using strategy any. The new files have y=[1] but labels has N entries. Both A03 crypto loaders return the PyG Data directly. The evaluator then reads data.y and indexes it with N-node validation/test masks. This target contract is invalid. Preserve graph-level semantics in the hierarchical API and add a node-evaluation adapter with an explicit N-length target, e.g. labels or level1_label, and shape/range/mapping checks. The generator must also select N-length labels when reading a previously generated Data. Merely changing a filename cannot fix this.

### 2. Cosine neighbors contain no discriminating embedding information

Every new level1_embedding reproduces the legacy float64 feature values exactly after lossless casting back from float32 (verified against the prior feature-byte hashes). Columns 0,1,2,4,5,6,7 remain zero and column 3 is strictly positive. Hence each embedding is a_i times e_3 and every normalized embedding is exactly e_3: cosine(i,j)=1 for all distinct pairs. Top-k selects tied similarities; node ordering/device tie rules can determine the adjacency. A documented tie-selected graph could be a control, but these artifacts do not substantiate a learned informative Level1 relation. New x also adds a constant-zero score column, changing the evaluated feature dimension from 8 to 9.

### 3. Reconstruction depends on its own output and inherited features

The generation script searches level2_graph.pt first, then hybrid/knn files, and extracts their embedding. It does not reconstruct that embedding from provider CSV/JSON or an audited encoder. With only provider files and no graph .pt, it fails. After the initial migration, existing Data.y has length 1 and is preferred over Data.labels, so repeating generation would save one label instead of N. The migration is not a self-contained or idempotent provider-to-node evaluation pipeline. No graph regeneration was performed during this audit, so the current files were not damaged.

### 4. Frozen manuscript evidence still refers to old graph inputs

The public numerical ZIP canonical manifest identifies hybrid hashes and 115,065 / 59,832 / 18,411 edges, with 8 features. New inputs have different hashes, 143,820 / 74,780 / 23,000 edges and 9 features. No new input identity appears in that archived campaign evidence. Existing AP/ROC/F1, ranks, support and memory results cannot be reattributed to the new graph. Establish a new versioned graph/feature/label/split manifest, conduct a separately recorded evaluation, and regenerate affected tables/statistics/manuscript/PDF. Preserve historical results as historical records.

### 5. Migration scope and test meaning

Current GoG ngnn config and examples still reference deleted hybrid filenames. Some current benchmark scripts retain a hybrid fallback; historical manifests legitimately retain the old identities. Distinguish archival references from active loaders and enforce an explicit new dataset version in an actual campaign. The reported 15 Phase3 tests exercise the hierarchical graph API on synthetic fixtures; inspection shows graph-label derivation is intentional. They do not verify the new node-level crypto evaluation, raw-data reconstruction or manuscript/result linkage. They were not rerun as a substitute for the targeted artifact audit.

## Required follow-up

1. Define auditable source features and relation metric/tie handling; choose raw provider-derived relations or independently produced features with declared training/split history.
2. Repair the node label adapter and repeated-generation contract; rebuild from provider inputs rather than relying on an unqualified graph cache.
3. Give the new construction its own dataset identity and source/config/tensor/mask hashes.
4. Rerun the approved crypto comparisons and preserve scores, thresholds and confusion matrices; rebuild the affected paper evidence and PDFs. Alternatively, explicitly exclude these three datasets from primary inference and recompute the remaining analysis.
5. Re-audit scientific input/result linkage separately from archive arithmetic and unit-test integrity.

## Changes made during this review

Added read-only audit code and numeric evidence; corrected architecture claims to the observed scope; appended construction/summary findings. No graph repair, new training, old-file restoration, frozen ZIP/metric change, manuscript/PDF rebuild, commit or push was performed. The supplied report remains unchanged.
