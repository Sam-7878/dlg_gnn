# GoG graph-family report review — 2026-10-09

## Conclusion

The supplied report substantially advances the audit. Independently verified: the sibling label artifact is exactly a same-label complete directed graph, and the frozen hybrid is the stored kNN edge set plus same-label additions with two or three extra outgoing neighbors/node. Exact original sampling code/seed and feature semantics remain unverified. The full artifact family strongly supports label-informed construction; the current hybrid results are not cleared as label-independent unsupervised financial-fraud validation. Recovering an original label-based writer alone would not clear that validity issue.

## Independent checks

The new script parses pickle opcodes without executing pickle, hashes all nine inputs, verifies canonical hybrid identity, and streams label edges in 4 MiB blocks. A directed-edge bitset detects duplicates across blocks. All label edges are bounded, same-label and non-self, with no duplicates. Their counts and per-node degrees exactly equal the two class-complete graphs. Feature bytes, labels and node-name order match across each sibling family. The largest bitset is approximately 25.9 MB; no full large label tensor is loaded.

| Chain | Class 0 / class 1 nodes | Verified label edges | Added degree 2 nodes | Added degree 3 nodes | Overlap slots if 3 sampled/node |
|---|---:|---:|---:|---:|---:|
| Ethereum | 8,367 / 6,018 | 106,208,628 | 15 | 14,370 | 15 |
| BSC | 6,377 / 1,104 | 41,877,464 | 16 | 7,465 | 16 |
| Polygon | 2,243 / 60 | 5,032,346 | 13 | 2,290 | 13 |

Every degree-2 node has an available same-label non-self kNN neighbor to supply a hypothesized third sampled edge. Sampling three then deduplicating is therefore feasible for every node. The union does not uniquely identify overlapping choices, seed, sampler or probability distribution.

## Claim-by-claim disposition

| Supplied claim | Assessment |
|---|---|
| kNN k=5 retained in hybrid | Verified: five outgoing kNN edges/node, none removed. Original metric and tie-handling unverified. |
| label graph is complete within labels without self-loops | Verified exhaustively for all three chains, including uniqueness and per-node degrees. |
| exactly k=3 sampled from label graph | Compatible with every observed edge/degree and required overlap. The original procedure is not uniquely identified. Stored k=5 in all nine artifacts; no metadata explicitly records label-sampling k=3. |
| sibling features identical | Verified byte-for-byte, with matching labels and node-name order. |
| feature 3 is tx_count | Unverified: no source-to-feature mapping supplied/found. Sampled present JSON edge/node counts differ. This disproves equating feature 3 with those present JSON counts, but does not rule out every historical/time-window transaction-count definition. |
| BSC maximum 3012 | Incorrect for the hash-matched frozen artifact: maximum 3227; Ethereum 4873, Polygon 88. |
| same-label edges alone prove labels used because unsupervised edges must cross classes | The necessity claim is false; label-independent relations can be perfectly homophilous. The explicit method=label complete sibling graph provides stronger evidence. |
| low global overlap proves injected edges | Overlap is descriptive, not causal proof. Provider global graphs are weighted contract relations; current code includes common-address relations. They are not established as a complete list of direct fund transfers. |
| original producer definitely lost during reset | Not established. No writer found in the available tree; earlier external/deleted/untracked code remains possible. |
| graph is automatically valid for supervised evaluation | Validation/test labels used to build graph edges can leak targets in supervised evaluation too. A split-aware construction protocol is required. |

## Other sources checked

The current GoG source inventory contains 85 Python/notebook files and 12 identified torch.save calls. Source hashes and static-parse warnings are recorded in gog_source_writer_audit.json. Syntax errors encountered were not repaired or executed; those files were inspected as text.

- `ngnn/train.py → HierarchicalDataset → load_global_contract_graph` consumes the existing graph; the Trainer writes model/optimizer/scheduler checkpoints.
- Fraud/MC training writers store model checkpoints. Individual/link/multiclass dataset writers store differently structured processed datasets.
- `fraud_detection/graph_individual/utils.py` provides a Euclidean kNN fallback for local graphs; it is not linked to the frozen contract-level family.
- `analysis/global.py` counts transactions with a cutoff and annotates a graph; it does not save this hybrid family.
- The notebook references an older `polygon_hybrid.pt` with 12,653 edges and no embeddings/labels, distinct from the 18,411-edge benchmark input.

## Manuscript consequence

Preserve the frozen inputs and scores as historical evidence. The three hybrid datasets must not support independently verified unsupervised fraud conclusions. Remediation requires documented label-independent construction and a separately identified evaluation, or diagnostic-only use with primary tables/statistics recomputed on the auditable remaining datasets. The older kNN file also needs feature provenance before it can be certified as a corrected input. No replacement tensors, training, manuscript/PDF rebuild or push occurred in this review.

## Reproduction

```bash
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/audit_gog_graph_family.py --data-root /mnt/d/_Work/_data/GoG
```

This reads all label edges and hashes all nine inputs. Public verify checks the resulting numeric-evidence hashes without raw data. The supplied report was preserved unchanged.
