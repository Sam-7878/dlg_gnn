# DLG Benchmark v2 protocol amendment A05

Date: 2026-10-03. This amendment applies to the benchmark publication release and can be reused by the Stream and TDS evaluations where their dataset/split contracts permit. It supplements the frozen protocol and A04 amendment; it does not rewrite historical run provenance.

## Publication evidence tiers

**Tier P (required for a paper cell):** run/model/seed identity, dataset and split identity, preserved metric JSON, source commit or frozen source snapshot, campaign environment record, and original or unambiguously reconstructed model configuration. A historical run lacking raw predictions or an exact per-run package lock may be reported only with that limitation disclosed.

**Tier R (required for new A05 targeted runs):** all Tier P fields plus raw anomaly score array and checksum, exact graph/tensor hashes, run config hash, current environment lock hash, source file hashes, device identity, and resource telemetry. Do not assign today's CUDA lock to a legacy run.

The fixed current environment is `.venv_cuda`: Python 3.14.4, PyTorch 2.14.1+cu130, CUDA 13.0, PyG 2.8.0.post1, PyGOD 1.1.0. The hashed package lock is `environment/locks/benchmark-a04-cuda.hashed.txt`. Record the actual GPU name in each new run because PyTorch's `CUDA_VISIBLE_DEVICES` index order can differ from `nvidia-smi` ordering under WSL2.

## Input identity

A changed feature hash is a hard input-identity mismatch even if nodes, edges, and labels match. An affected paper row must use the historical matching tensor or a complete targeted rerun on a new frozen tensor. Keep the superseded records in an audit archive.

## Exact AnomalyDAE execution

The allowed optimization materializes exact structural prediction/target rows of size $B\times N$ from the full graph. It retains the GAT encoder, attribute decoder, sigmoid structural decoder, complete double reconstruction loss, positive weights, optimizer, epochs, and score semantics. No edge/node sampling or graph truncation is permitted. Verify forward scores, loss, gradients, one optimizer step, score ordering, ROC-AUC, and PR-AUC against the dense reference before using the block path.

Use a block size selected from memory profiling before predictive results are inspected. The A05 base block size is 256; for graphs above 100,000 nodes the preflight times 16 consecutive exact blocks and excludes the first as warmup. The projected 50-epoch cell time is compared with the 24 GPU-hour per-cell guard. A resource OOM and an operational timeout are different support statuses. Exact row blocks reduce storage but preserve $O(N^2)$ arithmetic.

## Publication claims

CONAD's audited PyGOD 1.1 reference path is diagnostic only and excluded from ranks. BitcoinOTC's injected node labels on a real trust graph must not be called real fraud ground truth. Report S1 seven-model complete case, S2 continuity-five across 13 primary graphs, S3 financial/blockchain six with mixed label provenance, S4 synthetic seven, and S5 LANL external separately. No universal DLG superiority claim follows from S3's nonsignificant omnibus test.

Memory plots may use only hash-linked raw measurements. Show OOM as a status symbol, never as an invented high-memory bar. State that selected measured cases do not provide a complete detector memory profile or a portable GPU guarantee.
