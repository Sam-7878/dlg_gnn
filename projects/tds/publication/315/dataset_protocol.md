# Dataset and upstream reconstruction contract

Upstream: https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs
at source version `7264f1bf510f7ba4f5041ac7a29b606abc12f262`, with Token Data
linked by that README. Required inputs are `labels.csv`, `ethereum.zip`,
`bsc.zip` and `polygon.zip`; direct credential-free bulk download is not automated
by this closure package. Upstream availability is an external dependency.

There are 24,316 token/contract graphs. The inherited label maps `Category == 0`
to 1 and other categories to 0; label discovery time is unavailable. A sample
cutoff is its last valid recorded transfer timestamp, not the onset of fraud.
Timestamp/sample-ID ordering precedes count splits 17,021/3,647/3,648, with
6,783/308/107 positives. The held-out UTC interval is 2024-02-29 05:02:49 through
23:59:59 (68,230 seconds). Ethereum/BSC/Polygon test counts are 1,469/1,491/688,
with 68/38/1 positives. Contract disjointness does not establish wallet independence.

Inclusive inputs use `t_edge <= cutoff`. Node inputs are full-history log1p
in/out/total degree at retained endpoints, computed before retaining the most
recent 128 topology edges. No k-hop expansion is specified. There are 3,910 tied
observations out of 24,316; deterministic ordering is not intra-block execution
order. Strict earlier-time results use fixed original checkpoints and no refit.

Packed graph SHA-256:
`067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd`.
Label-file SHA-256:
`1356d063d675369984ef76277c4601e9f4fc780a06435c2e36a821e00ab1ea4a`.
The packed graph and raw upstream ZIPs are excluded from the public candidate.

`original_source/build_gog_chronological_dataset.py` requires an explicit
`--source-root` containing historical SCI-v2 `manifests/{chain}.json`, referenced
per-contract graph tensors, source paths, and label-semantics files. Missing inputs
raise FileNotFoundError; there is no synthetic fallback. It is a historical
derivative-to-packed builder, not a turnkey raw-ZIP acquisition pipeline.
Historical derivative links are unavailable in this checkout. A new raw-to-packed
rebuild is therefore **not** claimed. `scientific/strict_time.py` reconstructs
held-out inputs from upstream ZIPs and checks their identity against an author-local
packed graph; it does not reconstruct the complete training artifact. Do not run
training to close this work order.

Use `TDS_REPO_ROOT`, `TDS_OUTPUT_ROOT`, and `TDS_RAW_ROOT` for opt-in scientific
orchestration. Point outputs to a new directory. Missing or mismatched graph
identity fails closed. AP is threshold-grouped average precision; historical
`auc_pr` column names do not denote trapezoidal PR integration. Original seven
comparisons and six exploratory comparisons have separate Holm families.
