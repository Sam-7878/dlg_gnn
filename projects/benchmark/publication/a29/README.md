# A29 revision and submission evidence

This record belongs to the final A29 submission files.

- The [submission hash manifest](A29_SUBMISSION_HASHES.sha256) identifies the two PDFs, the two source ZIPs, the cover letter and the inquiry to the editorial office.
- The [PDF quality assurance](A29_PDF_QA.md) gives the page maps and the build checks.
- The [work order response](A29_WORK_ORDER_RESPONSE.md) records how each A29 item was handled.
- [A29_RESULTS_INDEX.json](A29_RESULTS_INDEX.json) lists every summary file in `results/a29/` with its SHA-256.

The scientific archive is still [the A24 release](https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a24-supplementary-2026-10-30-r1): 455 successful primary runs and 84 of 91 supported primary pairs. The A28 record is at [the A28 evidence tag](https://github.com/Sam-7878/dlg_gnn/tree/benchmark-a28-evidence-2026-11-06-r1/projects/benchmark/publication/a28). A29 changes neither the primary runs nor the primary ranks.

## A29 extensions

Each extension is a separate table in the manuscript. None of them enters the primary Friedman views.

| Extension | Summary files | Scripts (`projects/benchmark/scripts/`) |
|---|---|---|
| Baseline checks and equal-budget tuning (A29-04) | `results/a29/baseline_sanity/` | `a29_baseline_sanity.py`, `a29_cola_reference.py` |
| Dense vs Gram equivalence (A29-05a) | `results/a29/dense_vs_gram/` | `a29_dense_vs_gram.py` |
| Memory scaling sweep (A29-05b) | `results/a29/scaling_sweep/` | `a29_scaling_sweep.py` |
| Measured resource matrix (A29-05c) | `results/a29/resource_matrix/` | `a29_resource_matrix.py` |
| GADNR retry with a 12 h limit (A29-06) | `results/a29/gadnr_retry/` | `a29_gadnr_retry.py` |
| Rank statistics (A29-07) | `results/a29/rank_statistics/` | `a29_rank_statistics.py` |
| Graphs with real labels (A29-08) | `results/a29/real_label/`, `results/a29/preregistration/` | `a29_real_graphs.py`, `a29_preregister.py`, `a29_real_label.py` |
| Supervised reference (A29-12) | `results/a29/supervised_reference/` | `a29_supervised_reference.py` |
| Reference check (A29-13) | `results/a29/references/` | `a29_verify_references.py` |

Per-node scores, labels, split masks and node identifiers are kept internally. The scope is unchanged from A28.

## Measurement conditions

- Device: an RTX 3090 (24 GiB) pinned by UUID. Some accuracy-only tuning ran on an RTX 4070 Laptop GPU (8 GiB).
- Under WSL, a GPU allocation larger than the card spills into shared system memory instead of failing. Runs whose reserved memory exceeded 24 GiB are marked "spilled". Their times are not comparable with runs that stayed in GPU memory.
- Timed runs were made one at a time, with a CPU-speed probe running alongside.
- Two spilling jobs at the same time caused out-of-memory failures. Those first attempts are kept as `attempt1_*` next to the repeated runs and are disclosed in the manuscript.
