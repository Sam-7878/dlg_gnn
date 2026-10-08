# DLG-GNN Benchmark v2 — A04 effective amendment

**Effective date:** 2026-10-03  
**Authority:** [A04 final correction work order](../../work_reports/benchmark/226_A04/DLG_Benchmark_v2_Final_Correction_Work_Order_A04_2026-10-03.md)  
**Status:** effective for the A04 evidence release; Gate G4 remains HOLD until its validator passes.

This amendment overrides conflicting execution and publication statements in the 2026-09-30 FROZEN protocol and the A02 effective protocol. Those files retain their historical decisions. A03 artifacts remain source evidence, but their completion declaration and validator PASS are withdrawn for publication purposes.

## Frozen current CUDA qualification environment

| Component | A04 current qualification version |
|---|---|
| Host | WSL2 Ubuntu 26.04, actual kernel/driver in toolchain probe |
| Python | 3.14.4 |
| PyTorch | 2.14.1+cu130 |
| CUDA runtime | 13.0 |
| PyG | 2.8.0.post1 |
| PyGOD | 1.1.0 |
| PyG extension | `pyg_lib==0.9.0+pt214cu130` |
| Interpreter | repository root `.venv_cuda/bin/python` |
| Package inventory | `environment/locks/benchmark-a04-cuda.lock.txt` (same frozen versions as A03) |
| Exact wheel lock | `environment/locks/benchmark-a04-cuda.hashed.txt` |

No package upgrade or in-place environment replacement is authorized for this benchmark release. The earlier Python 3.12 / PyTorch 2.11 / cu128 plan was a qualification candidate, not the environment that produced A03 results. The PyG 2.8 release compatibility statement does not formally include PyTorch 2.14; this combination is treated as *empirically qualified*. A04 preserved all 74 wheel files and their SHA-256 values in the local `environment/wheelhouse-a04/` and the release evidence bundle. A clean temporary venv installed from those wheels without network access; `pip check`, the eight-detector smoke, exact reconstruction, and fused GCN checks all passed on the RTX 3090. The 3.0-GiB wheelhouse is deliberately Git-ignored and must be retained or copied with the release when moving to another machine.

This current stack is not the execution environment for every inherited result. The original Round5 `environment_freeze.json` records Python 3.12.13, PyTorch 2.5.1+cu121, CUDA 12.1, and PyG 2.7.0. The A04 lock is separately labeled as qualification evidence. A03 crypto run files and LANL run files do not carry an exact per-run package lock. Never fill their execution-lock field with the current lock merely because the current environment can run the code.

The RTX 3090 24 GB eGPU is the primary execution envelope. The laptop RTX 4070 8 GB and a same-3090 8-GiB allocator cap are separately identified diagnostics. The physical GPU and allocator cap must never be described as equivalent hardware. A future AMD/ROCm worker for Stream, TDS, or LLM work has its own lock and qualification; it does not change this release.

## Scientific and evidence decisions

1. **Dataset identity:** The publication manifest has 13 primary datasets plus external LANL. Each canonical row records the model-consumed tensor shape and hashes, raw source hash, constructed artifact path and hash, graph population, evaluation population, positive evaluation count, prediction unit, directedness, split hashes, and builder revision/source hash. A missing tensor or split hash is recorded as missing and blocks G4; no file-name or hand-copied estimate substitutes for the loader output. The newly constructed A04 input snapshots are in `outputs/benchmark/a04_constructed_graphs/` (about 3.8 GiB, Git-ignored) and are not asserted to be the historical Round5 input artifacts. Elliptic's full 203,769-node source population and the labeled/evaluated population are separate fields. Native-chain feature dimension is measured after the runner's graph construction. LANL's event, node, or other prediction unit must be established from its actual evaluator before publishing a prevalence.
2. **CONAD:** `CONAD-PyGOD-1.1-reference` is diagnostic-only. `CONAD-corrected` is excluded from the primary results, fair ranking, and inferential statistics. A03's experimental squared loss is not the paper's unsquared Eq. (1). A small integration check may compare reference and paper-Eq1 paths, but its result stays in the appendix/debug record. The seven functioning primary models are DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, and DLG-Aug; support remains cell-specific.
3. **Ablation:** The Elliptic 25-row A03 CSV is the sole aggregate source. Permutation did not reduce PR-AUC or F1 versus aligned DLG-Aug, so node-wise local/global alignment is not established as the cause of the gain. Zero augmentation improving over Base suggests capacity or input-path effects; Base-70 does not support an extra-epochs-only explanation. Gate reports distinguish raw `alpha` from `sigmoid(alpha)`.
4. **Memory:** The generated 224-row A03 memory table is a modeled planning artifact, not 224 measured runs. Publication memory rows require an actual raw full/cap telemetry record. Present only measured selected cells and explicitly leave absent envelope comparisons unreported.
5. **Statistics:** S1 uses the seven-model complete case. S2 retains the frozen continuity-five set; S3 is the six-dataset financial/blockchain domain view with mixed label provenance, S4 the seven `-Syn` datasets, and S5 LANL descriptive. The `BitcoinOTC` runner overwrites node labels and injects synthetic anomalies on a real trust network, so the former name **real-label six** is scientifically inaccurate. Diagnostic CONAD never enters Friedman or Wilcoxon ranks. S3's approximately 0.082 omnibus p-value is non-significant; DLG-Aug superiority on this domain view is not claimed.
6. **Claims:** Every paper metric cell maps to approved dataset/model/seed runs, metric JSON hashes, unique dataset and split identities, model configuration, and the exact environment lock hash. A source CSV row is not silently presented as a separately archived raw score. Missing source files, score hashes, cap logs, or identity fields keep G4 on HOLD. Claims are generated from tables, including heterogeneous Ethereum/BSC/Polygon outcomes.

## Release and stopping gate

The A04 release directory is `evaluation/benchmark/v2/paper_ready_final/`; its evidence archive is `publication_evidence_a04/`. The cross-file validator must reject contradictory metadata, unsupported provenance, a corrected-CONAD primary cell, a diagnostic-CONAD rank, fabricated telemetry, and claims contradicted by source tables. Once all A04 conditions pass, stop benchmark experiments and proceed to manuscript revision. Do not label a partially reconstructed evidence chain as publication-ready.

The current validator result is **G4 HOLD (20/24)**. The BitcoinOTC Node2Vec feature tensor rebuilt from the current loader does not match the historical Round5 feature hash, although its edge and label hashes match. No historical constructed tensor was preserved. None of the 430 successful, metric-producing historical run JSONs preserved a raw score array or its checksum, and none carries an exact execution-environment lock tied to the run. An archived legacy venv inventory matches the Round5 environment manifest's main versions but is not bound to each run. Moreover, 120 successful A03 crypto/LANL runs have reconstructed script/default configuration descriptors, not original run-bound config hashes. The seven new measured memory runs that completed have raw score hashes, but they do not replace the historical performance runs. These four deficits require source artifact recovery or a new explicitly scoped evidence campaign before the manuscript can use this release as validated publication data.

## Porting boundary

Stream and TDS may reuse the environment capture, dataset/run identity schemas, telemetry, job isolation, and validator interfaces. They must declare their own prediction unit, split protocol, metrics, baseline set, and inferential family. LLM work may reuse the artifact and environment process but must qualify new hardware and backend locks separately. No future stack change retroactively alters the frozen Benchmark A04 evidence.
