# DLG-GNN Benchmark v2 Protocol — EFFECTIVE A02 CONSOLIDATED EDITION

**Protocol ID:** DLG-BENCH-V2.0+A02  
**Consolidation date:** 2026-10-02 (Asia/Seoul)  
**Historical freeze:** 2026-09-30, preserved in the original FROZEN file  
**Authority:** `DLG_Benchmark_v2_Protocol_Amendment_A02_PORTABLE_2026-10-02.md`  
**Work order:** `DLG_Benchmark_v2_Revised_Work_Order_A02_PORTABLE_2026-10-02.md`  
**Status:** Ready for author adoption; hardware/software qualification and prior-result declaration are not yet certified.  
**Publication route:** Benchmark Preprints.org release, then Applied Sciences Special Issue — *Graph Neural Networks: Theory, Methods and Applications*.

> **Version integrity.** This is an effective consolidated protocol, not a replacement historical freeze. Preserve the original FROZEN bytes and the A02 amendment. Complete the adoption record, including whether earlier v2 performance has been inspected, before new production runs. Do not retrospectively claim that A02 preceded results already inspected. Changes driven by confirmed errors, compatibility or actual hardware require an impact record and affected-result correction/rerun; unfavorable performance is not a reason for post-hoc protocol changes.

The original section numbering is retained. Sections not overridden by A02 retain their scientific rules. CUDA-first scope and actual hardware replace the former mandatory dual-backend/server plan. Stream/TDS/LLM reuse the infrastructure, not this Benchmark's model, split, seed or statistical contract.

---

# 1. Benchmark v2 Research Objective

Benchmark v2 evaluates how DLG-GNN variants should be used reliably across graph domains and declared computational environments, rather than requiring one variant to win every dataset.

### RQ1 — Cross-domain and native-domain predictive behavior
How do DLG-Base and DLG-Aug behave on the 13 primary datasets, including Ethereum/BSC/Polygon native-domain regression validation, and the separate LANL external cybersecurity domain?

### RQ2 — Conditional local-to-global utility
Under which observable graph/anomaly conditions does local augmentation improve, preserve or degrade performance? Use the predeclared minimal ablations and descriptive diagnostics; do not tune away negative cases.

### RQ3 — Qualified CUDA execution and resource support
Can original detector semantics be executed on the modern CUDA stack, and how does selected-cell support change between physical 4070 8 GB, 3090 24 GB, and a clearly labeled same-3090 allocator-cap diagnostic?

Future CUDA/ROCm portability is a separate qualified extension, not a current production prerequisite or an already demonstrated result.

### RQ4 — Triggered graph-fidelity diagnostics
Where the historical index-cut issue requires explanation, compare recoverable historical cuts against full graphs and, only when needed, frozen random induced samples. Partition/halo or host-assisted execution requires a separate explicit trigger and must not silently replace a primary exact result.

> Preserve detector semantics first; report unsupported execution rather than silently changing the task. Stop the current experiment cycle when core evidence is sufficient for manuscript revision.

---

# 2. Dataset Portfolio: 13 + 1

Benchmark v2 contains **13 primary datasets** and **1 external validation dataset**.

The three original DLG blockchain datasets — **Ethereum, BSC, and Polygon** — are added to the v1 ten-dataset primary portfolio. Their labels are frozen as **real external fraud labels**, not synthetic injections.

LANL-RedTeam remains an external cybersecurity validation dataset and is not inserted into the primary omnibus statistical block.

## 2.1 Cross-domain primary datasets inherited from Benchmark v1

| ID | Dataset | Label provenance | v2 role |
|---|---|---|---|
| P01 | Elliptic | Real financial anomaly labels | Cross-domain / real financial |
| P02 | DGraphFin | Real financial anomaly labels | Cross-domain / real financial |
| P03 | Yelp-Syn | Controlled synthetic injection | Cross-domain / synthetic |
| P04 | Amazon-Syn | Controlled synthetic injection | Cross-domain / synthetic |
| P05 | BitcoinOTC | Real blockchain trust/anomaly labels | Cross-domain / real blockchain |
| P06 | Flickr-Syn | Controlled synthetic injection | Cross-domain / synthetic |
| P07 | Reddit-Syn | Controlled synthetic injection | Cross-domain / synthetic / density stress |
| P08 | Cora-Syn | Controlled synthetic injection | Cross-domain / synthetic |
| P09 | CiteSeer-Syn | Controlled synthetic injection | Cross-domain / synthetic |
| P10 | PubMed-Syn | Controlled synthetic injection | Cross-domain / synthetic |

## 2.2 Native-domain blockchain validation added in Benchmark v2

| ID | Dataset | Label provenance | v2 role |
|---|---|---|---|
| P11 | Ethereum | **Real external fraud labels** | Native-domain DLG validation / scalability stress |
| P12 | BSC | **Real external fraud labels** | Native-domain DLG validation / scalability stress |
| P13 | Polygon | **Real external fraud labels** | Native-domain DLG validation / scalability stress |

These datasets MUST NOT be described as independent unseen discovery datasets if they were used in the preceding DLG-GNN study. Their role is:

- native-domain regression validation,
- backward compatibility of the benchmark implementation,
- practical DLG usage validation, and
- large/high-degree blockchain scalability evaluation.

## 2.3 External validation

| ID | Dataset | Label provenance | v2 role |
|---|---|---|---|
| E01 | LANL-RedTeam | Official red-team labels | External cybersecurity validation |

LANL is excluded from the 13-dataset primary omnibus inference and is reported as an external domain-shift result.

---

# 3. Dataset Grouping Frozen for Analysis

The following group definitions are frozen before v2 results.

## G1 — Real-label financial/blockchain primary subset

- Elliptic
- DGraphFin
- BitcoinOTC
- Ethereum
- BSC
- Polygon

This six-dataset subset replaces the need to market a mixed real/synthetic set as a single "fraud-oriented" group.

## G2 — Synthetic-injection primary subset

- Yelp-Syn
- Amazon-Syn
- Flickr-Syn
- Reddit-Syn
- Cora-Syn
- CiteSeer-Syn
- PubMed-Syn

Synthetic results MUST be explicitly reported as synthetic-injection anomaly detection and MUST NOT be described as real fraud detection.

## G3 — Full 13-dataset primary portfolio

G1 + G2.

## G4 — External cybersecurity

- LANL-RedTeam only.

---

# 4. Dataset Artifact Freeze Requirements

Before the first production v2 performance run, every dataset artifact MUST have a canonical manifest containing at least:

- dataset name and protocol ID,
- source/provenance reference,
- raw-data version or download identifier,
- preprocessing script commit,
- node count \(N\),
- stored edge count \(E\),
- feature dimension \(F\),
- number and fraction of positive labels,
- directed/undirected status,
- self-loop policy,
- duplicate-edge policy,
- isolated-node policy,
- feature normalization policy,
- split masks or split-generation metadata,
- anomaly-injection configuration where applicable,
- SHA-256 hashes of the final feature matrix, edge representation, labels, and split masks,
- graph construction date,
- code commit used to generate the artifact.

**No production model run is allowed before the artifact manifest is complete.**

For Ethereum/BSC/Polygon, the exact \(N/E/F\), positive rate, feature construction, split masks, and hashes are intentionally **not invented in this protocol**. They must be imported from the verified existing DLG dataset artifacts and frozen before execution.

---

# 5. Synthetic Anomaly Policy

The seven `-Syn` datasets retain the Benchmark v1 controlled contextual + structural anomaly construction.

Frozen principles:

- main injection seed: **42**,
- synthetic labels replace the corresponding anomaly-label vector used for benchmark evaluation,
- contextual and structural injection settings remain identical to Benchmark v1 unless a documented implementation incompatibility is discovered before result inspection,
- injection labels are used for evaluation only, not for detector optimization,
- no v2 model-specific injection tuning is permitted.

Any injection sensitivity study is secondary and MUST NOT replace the frozen primary graph instance.

---

# 6. Detector Suite: 8 Configurations

The primary detector suite is frozen to exactly eight configurations:

| ID | Detector | Family | Primary role |
|---|---|---|---|
| M01 | DOMINANT | Reconstruction | Established baseline |
| M02 | AnomalyDAE | Dual autoencoder / reconstruction | Established baseline |
| M03 | CoLA | Contrastive | Established baseline |
| M04 | CONAD | Contrastive + reconstruction | Established baseline |
| M05 | GAD-NR / GADNR | Neighborhood reconstruction | Established baseline |
| M06 | OCGNN | One-class GNN | Established baseline |
| M07 | DLG-Base | Local/global reconstruction reference | DLG non-augmentation reference |
| M08 | DLG-Aug | Frozen-local augmented reconstruction | DLG benchmark variant |

### Excluded from the primary 8-model matrix

The following may be analyzed separately but MUST NOT be inserted post hoc into the main 8-model statistical matrix:

- DLG-Local
- DLG-Fusion
- DLG-Aug-Zero
- DLG-Aug-Permuted
- DLG-Base-70
- any newly added contemporary challenger model

If a recent GAD method is added for journal review, it must be declared in a separate **Contemporary Challenger** protocol block before its results are inspected.

---

# 7. Model Training Policy

Unless a compatibility correction is required and documented before performance inspection:

- model hyperparameters retain the frozen Benchmark v1 / upstream semantics,
- all supported primary detectors train for **50 epochs**,
- no newly introduced early stopping is allowed,
- DLG-Aug performs **20 local-pretraining epochs + 50 global epochs**,
- DLG-Aug historical reconstruction weighting retains \(\lambda = 0.5\),
- hidden dimensions MUST NOT be reduced to force a model to fit in memory,
- epochs MUST NOT be reduced to force a model to finish,
- objectives MUST NOT be replaced by negative-sampling approximations,
- no silent CPU fallback is allowed,
- score direction MUST NOT be reversed after observing labels.

Exact per-model hyperparameters MUST be serialized to a machine-readable run manifest before production execution.

---

# 8. Seeds

## 8.1 Primary model seeds

The five frozen model seeds are:

```text
42, 43, 44, 45, 46
```

These seeds are used for every supported detector × dataset primary cell.

## 8.2 Synthetic graph seed

```text
42
```

for the frozen main synthetic graph instance.

## 8.3 Random-sampling diagnostic seeds

Random graph-reduction diagnostics use a separate frozen sampling-seed set:

```text
142, 143, 144, 145, 146
```

Sampling seeds MUST NOT be selected based on metric results.

## 8.4 Partition seed

If a partitioner requires a random seed, the default partition seed is:

```text
242
```

The same partition artifact is reused across detector seeds unless the method itself requires otherwise.

---

# 9. Splits and Label Usage

The evaluation setting remains **transductive and unsupervised for detector fitting**.

- labels MUST NOT be used to optimize detector parameters,
- labels MAY be used for deterministic split construction,
- labels MAY be used for validation-only threshold selection,
- labels MAY be used for final metric computation.

### DGraphFin

Retain the official aligned split used by Benchmark v1.

### Existing v1 datasets without official splits

Retain the existing deterministic Benchmark v1 split implementation **bit-for-bit** where possible. Export the resulting masks and freeze their hashes.

### Ethereum / BSC / Polygon

1. If the verified DLG dataset artifact already contains an externally defined evaluation split, preserve it.
2. Otherwise, use the same deterministic split utility as the v1 benchmark.
3. The split masks MUST be generated and hashed before production runs.

The protocol MUST NOT choose split ratios after viewing detector performance.

---

# 10. Modernized Software Stack and Environment Identity

## 10.1 Qualification candidates, not pre-certified locks

```text
Python                  3.12, exact patch frozen after qualification
PyTorch public release  2.11.0
PyTorch Geometric       2.8.0
PyGOD                   1.1.0
Current CUDA wheel      cu128
Future ROCm             exact GPU/OS/runtime/build to be qualified separately
```

Preserve the original ROCm Core SDK 7.14.1 as a future SDK review candidate, not as a presumed PyTorch wheel index. Official PyTorch 2.11.0 installation guidance lists a rocm7.2 wheel channel; SDK release, wheel build, driver and GPU support are separate compatibility facts. Do not invent a rocm7141 channel or a completed future lock.

## 10.2 Required environment migration

Before installing the modern stack, stop workers and archive the actual `.venv` inventory, original absolute path, configuration, source state and bytes with hashes. Rename `.venv` to `.venv_old` without overwrite. Only then create `.venv_cuda` from the selected Python 3.12 base interpreter. Future ROCm uses `.venv_rocm`.

The renamed `.venv_old` is an archive, not a guaranteed runnable environment: activation scripts and entry-point shebangs may retain absolute paths. For legacy reruns restore the exact original location with compatible base dependencies, or recreate a separate environment from preserved artifacts. Never install new packages into the legacy archive. Full commands and rollback boundaries are in Work Order Phase A.

## 10.3 Common versions versus backend builds

Common Python-level dependency versions and source commits should match. Use a common exact constraint set plus complete task/backend/platform hash locks. Keep CUDA/HIP wheels, required graph extensions, vendor libraries and compiled caches separate. A common constraints file alone is not a complete reproducibility lock.

Match build frontend tools, project C++ language standard and supported host compiler versions where feasible. NVCC versus HIP/Clang, GPU architecture targets, runtime/driver and wheel build ABI necessarily remain backend-specific. Record actual compiler versions, PyTorch build configuration, ABI, compile flags, wheel/source hashes and differences in an allowlist. Do not force a mismatched ABI to make version strings look identical.

```text
environment/constraints/common.in
environment/constraints/common.lock.txt
environment/locks/benchmark-cuda.lock.txt
environment/locks/<task>-<backend>.lock.txt
environment/toolchains/cuda.json
environment/toolchains/rocm.json       # only after future qualification
environment/backend_differences.yaml
```

The CUDA lock must be independently recreated and tested before production. No in-place package upgrade is permitted mid-benchmark without impact accounting. Future LLM requirements that need a newer stack use a new release/worktree rather than mutate the frozen GNN environment.

External compatibility references are recorded in Work Order §26. None replaces testing all required detector operators on the actual system.

---

# 11. Environment Qualification Gate

Current 13+1 production requires the modern **CUDA** environment to pass qualification. Future ROCm has its own gate and does not block current manuscript completion.

Required CUDA checks:
1. All eight detectors import, instantiate, train and produce finite scores on fixed small graphs.
2. Actual required sparse/scatter/sampling/kNN operations complete forward/backward tests.
3. Seed handling, score orientation and raw-score-to-metric identity are verified.
4. Exact sparse/Gram reconstruction and fused GCN agree with fixed-weight references in loss, per-node score and gradients; record precision and tolerances before inspection.
5. DLG-Base, DLG-Aug pretrain/freeze/global and CONAD-specific branches execute as defined.
6. Expected backend, explicit GPU identity, memory telemetry and complete manifests are verified.
7. The full environment can be recreated from locked artifacts; shared-core/task-isolation fixtures pass.

A failed qualification is an environment/implementation result, not permission to change a detector objective or silently use CPU. Required correctness gates still apply before production on the already available 3090.

---

# 12. CONAD–DOMINANT Mandatory Preflight Audit

Because Benchmark v1 produced unexpectedly similar DOMINANT and CONAD results on multiple datasets, Benchmark v2 MUST audit this before full production execution.

For qualification datasets, record at every relevant epoch:

- reconstruction loss,
- CONAD contrastive loss,
- total loss,
- effective CONAD `eta`,
- number of augmented nodes,
- original and augmented edge counts,
- gradient norm from the reconstruction term,
- gradient norm attributable to the contrastive term where measurable,
- final DOMINANT and CONAD raw anomaly scores,
- Pearson score correlation,
- Spearman score correlation,
- model-parameter distance or checksum.

### Required audit logic

If DOMINANT and CONAD metrics or raw scores are numerically identical across multiple seeds:

- do **not** automatically accept this as a scientific finding,
- first verify that the contrastive branch executed,
- verify that augmentation occurred,
- verify that the loss weighting is non-degenerate,
- verify that result-table aggregation did not duplicate another model's output,
- verify that no shared wrapper bypassed CONAD-specific training.

Only after those checks may identical behavior be reported as a model/result property.

---

# 13. Actual Hardware Envelopes

## H8-CUDA — physical 8 GB development/diagnostic envelope
NVIDIA RTX 4070 Laptop GPU, nominal 8 GB VRAM, qualified CUDA build. Use for correctness, small diagnostic work and selected physical 8 GB checks; not a second full 560-run matrix.

## H24-CUDA — current primary production envelope
Gigabyte AORUS RTX 3090 Gaming Box eGPU, nominal 24 GB VRAM, attached to the current host. Use the same qualified CUDA software lock as the 4070. The device is already available; availability does not waive qualification.

## H8cap-CUDA — same-device allocator diagnostic
On the 3090, restrict the PyTorch allocator to a predeclared target, normally 8 GiB = 8 × 2^30 bytes, using the supported memory-fraction interface before model allocation. Record target bytes, fraction, total physical bytes and actual telemetry. The historical '8GB-cap' name must always be explained as an allocator diagnostic, not physical 8 GB hardware or a hard cap on every process/library allocation.

## H-ROCm-FUTURE — deferred AMD platform
Exact Radeon model, gfx target, VRAM, host RAM, OS/kernel and driver are unspecified until procurement/installation and official support checks. Do not report an RX 7900 XTX or a dedicated 64 GB RAM server as current measured hardware.

Probe actual host CPU/RAM, operating system/WSL, driver, eGPU link information and device identity. A missing telemetry field is UNAVAILABLE with a reason, never an invented numeric value.

---

# 14. Host/eGPU Stability and Measurement Policy

The current setup remains a notebook host with an eGPU; it does not eliminate laptop thermal/power, host scheduling or interconnect confounds. Record and control them instead of claiming server isolation.

Production runs must use fresh processes, prohibit unrelated GPU work, avoid competing heavy CPU/RAM/disk tasks on the shared host, retain declared vendor-default/non-manually-overclocked settings, and log temperature/power/clocks/utilization/VRAM and host RSS at a frozen interval where available. Record eGPU attachment/link details and driver/kernel information.

Decompose measured scope into preprocessing, transfer, synchronized model execution and end-to-end wall time where implemented. State warmup, synchronization, cache policy, instrumentation and repeat counts. Do not compare a Stream event E2E measurement with a TDS batched model panel timer or call either a hardware-independent GPU speed result.

4070 and 3090 may process independent non-timing development tasks outside production measurement windows. Shared-host contention must not contaminate paper resource comparisons. Primary rankings remain environment-specific.

---

# 15. Full-Graph Exact Execution Policy

## 15.1 Primary rule

```text
FULL GRAPH EXACT OR FAIL CLOSED
```

The primary exact benchmark MUST NOT make a cell successful by:

- index truncation,
- random node truncation,
- graph partition substitution,
- neighbor-sampling substitution,
- negative-sampling substitution,
- hidden-dimension reduction,
- fewer epochs,
- objective modification,
- silent CPU fallback.

If the exact full-graph model does not fit, the result is an execution-support status.

## 15.2 Semantics-preserving implementation optimization is allowed

Implementation transformations are allowed only when the model objective and score semantics are preserved and equivalence is verified.

Examples include:

- exact sparse structural reconstruction,
- Gram-identity reconstruction,
- chunked exact reconstruction,
- fused sparse GCN/message passing,
- caching of static normalized sparse operators,
- other numerically verified exact transformations.

---

# 16. Support Status Taxonomy

Each model × dataset × environment cell receives exactly one support status.

```text
SUPPORTED_EXACT
UNSUPPORTED_RESOURCE_OOM
UNSUPPORTED_OPERATIONAL_TIMEOUT
UNSUPPORTED_BACKEND
UNSUPPORTED_DEPENDENCY
FAILED_NUMERICAL_VALIDATION
FAILED_RUNTIME_ERROR
```

A supported predictive-performance row exists only for `SUPPORTED_EXACT`.

Missing cells:

- are never assigned zero,
- are never assigned worst rank,
- are never filled with a partitioned or sampled substitute,
- are never copied from another backend.

---

# 17. Runtime Guard

The per model × dataset × seed production guard remains:

```text
24 GPU-hours
```

If exact execution is valid but exceeds the predeclared guard:

```text
UNSUPPORTED_OPERATIONAL_TIMEOUT
```

The status describes the declared environment and implementation, not mathematical impossibility.

---

# 18. Selected 8 GB versus 24 GB Evaluation Policy

Keep hardware/resource envelopes separate. Do not construct a same-environment ranking by mixing supported cells from different devices/backends.

Predeclare a limited memory diagnostic cell list using historical OOM/index-cut evidence and model/graph relevance, including high-memory AnomalyDAE/GAD-NR, a CONAD large graph, a DLG exact-path control and a representative native blockchain case. Do not rerun the whole matrix on a second device.

Prefer the **same3090 full envelope versus allocator-cap** contrast. Add a physical4070 check when that distinction matters to reviewer-critical evidence. A cap may omit context and non-PyTorch allocations; report its implementation boundary explicitly.

```text
8 GB failure -> 24 GB success: declared memory-envelope/implementation dependent support
8 GB failure -> 24 GB failure: unresolved resource/implementation limitation at these envelopes
both supported: shared-support descriptive comparison
```

A physical4070 versus3090 comparison also changes GPU architecture/power/interconnect, so it does not isolate memory alone. Timing remains descriptive.

A clearly deterministic allocation OOM may be confirmed once after cleanup in a fresh process and used to avoid four redundant seed failures. For stochastic/seed-dependent memory use, preserve seed-level status and do not infer all seeds fail. Incomplete-seed cells do not become five-seed complete cases.

---

# 19. Future CUDA/ROCm Portability Evaluation — DEFERRED

The present paper may describe portability-oriented interfaces but must not claim tested ROCm support without actual runs. When the AMD platform exists, use the same source/task semantics and as many common package versions as compatibility permits, with a separate `.venv_rocm`, full binary lock and toolchain manifest.

Qualify fixed-input/fixed-weight forward/loss/score/gradient behavior before training comparisons. Record operator support, precision/nondeterminism policy and predeclared tolerances. PyTorch ROCm uses the cuda device API; identify the backend through build metadata, not the API spelling alone.

For shared supported model × dataset × seed cells, compare raw-score correlation, ROC/PR/F1 differences, seed variability, support, memory and time descriptively. Do not copy missing CUDA cells from ROCm. Equal seeds do not guarantee identical random streams or bitwise training trajectories across vendors.

Initial interoperability is independent job dispatch to separate CUDA/ROCm processes with CPU-visible artifacts and validated checkpoints. A shared Python environment, automatic VRAM pooling or mixed-vendor single-job DDP is not part of this protocol.

---

# 20. Graph Reduction Policy — Triggered Diagnostics Only

Graph reduction cannot substitute for the exact primary input/result.

```text
F0 = full graph with original detector semantics — sole primary path
F1 = structure-aware partition/halo — conditional separate study
F2 = random induced-subgraph sampling — diagnostic when needed
F3 = recoverable historical index cut — diagnostic/history only
```

A01's focused scope is effective: use F3 only to explain the historic issue when its exact cutoff/order can be recovered; use F2 only if index-order bias remains confounded with graph-size effects. Do not invent a new cutoff. F1 is not mandatory.

If a reviewer-critical24 GB cell fails, first examine avoidable allocations and verified exact sparse/chunked paths. Host-Assisted Exact Execution requires a declared, equivalence-checked separate execution block; it is not a silent CPU fallback in the primary GPU-only matrix. Consider partition/halo only after an explicit amendment, not as a new framework contribution required to finish this paper.

Sections21–24 define conditional experiment semantics if triggered; they do not require all four modes to be executed.

---

# 21. F1 — Structure-Aware Partition / Halo Policy

Partitioning is a scalability study, not an automatic fallback.

A partition experiment must:

1. create balanced core partitions using a documented topology-only procedure,
2. avoid label-informed partition decisions,
3. preserve original node/edge IDs,
4. expand each core by the full message-passing receptive-field halo required by the evaluated GNN path where feasible,
5. use halo nodes for message passing but avoid duplicate scoring of non-core nodes,
6. preserve original directed edges for model computation even if an undirected surrogate is used to construct partitions,
7. preserve global normalization quantities where required by the reference GCN semantics,
8. separately handle/record high-degree hubs if replication or vertex-cut logic is introduced,
9. record overlap/replication factor,
10. never label the path `exact` unless numerical equivalence has been demonstrated.

### Reconstruction caution

For DOMINANT, CONAD, DLG-Base, and DLG-Aug, preserving the GCN receptive field does not by itself prove preservation of the all-pairs structural reconstruction objective.

Therefore:

- exact/chunked structural reconstruction remains the preferred mechanism,
- partitioned reconstruction MUST be treated as approximate unless mathematical/numerical equivalence is separately established.

---

# 22. F2 — Random Sampling Policy

Random sampling is a **diagnostic sensitivity test**, not a primary execution method.

Rules:

- node sampling must not use anomaly labels,
- sampled subgraphs are induced subgraphs unless a separately frozen sampler is specified,
- use sampling seeds `142–146`,
- when comparing with a historical index cut, use the same target node count as the historical cut,
- retain all five random samples rather than choosing the best one,
- report sampling variability explicitly.

Random sampling results MUST NOT replace an OOM primary cell.

---

# 23. F3 — Historical Index-Cut Policy

Simple index truncation is prohibited in new primary v2 execution.

Historical index-cut experiments are allowed only to diagnose whether the former memory workaround altered model behavior.

Rules:

- reproduce the exact historical cutoff/order only if it can be recovered from the old code/configuration,
- do not invent a new cutoff,
- label every result `HISTORICAL_INDEX_CUT`,
- do not include it in model-ranking or omnibus statistical tests,
- use it only for fidelity/collapse analysis against F0/F2/F1.

This experiment is especially relevant to the DOMINANT–CONAD similarity investigation.

---

# 24. Partition / Sampling Fidelity Metrics

Where F0 full-graph reference exists, compare F1/F2/F3 against it using:

### Score fidelity
- Pearson correlation of raw anomaly scores,
- Spearman rank correlation of raw anomaly scores,
- normalized score MAE/RMSE after clearly documented score normalization if required,
- top-k anomaly-set overlap.

### Predictive fidelity
- \(\Delta\) ROC-AUC,
- \(\Delta\) PR-AUC,
- \(\Delta\) validation-selected F1,
- precision/recall budget differences.

### Ranking fidelity
Where multiple detectors are available:
- detector-rank changes,
- Kendall/Spearman association of detector rankings.

No partition/sampling method is called "semantics-preserving" merely because its metric change is small. Exactness requires a mathematical or numerical equivalence argument.

---

# 25. Primary Evaluation Metrics

For every supported exact seed:

## Threshold-free primary metrics

- **ROC-AUC**
- **PR-AUC**

PR-AUC is emphasized under severe class imbalance.

## Threshold-dependent primary metric

- **Validation-selected F1**

Threshold selection:

\[
\tau^* = \arg\max_{\tau} F_1(y^{val}, 1[s^{val} \ge \tau]).
\]

The selected \(\tau^*\) is applied unchanged to test scores.

Test-oracle F1 may be retained only as a diagnostic and MUST NOT be used for model selection or headline comparison.

---

# 26. Secondary Metrics

Report where defined:

- validation-selected precision,
- validation-selected recall,
- MCC,
- balanced accuracy,
- Precision@1% alert budget,
- Recall@1% alert budget,
- Precision@5% alert budget,
- Recall@5% alert budget,
- prevalence-matched top-k precision/recall.

The top-k ranking is based on the detector's raw anomaly-score ordering.

---

# 27. Resource / Scalability Metrics

For each production run, record:

- support status,
- wall-clock training time,
- total end-to-end runtime,
- peak allocated GPU memory,
- peak reserved GPU memory where supported,
- system RAM peak / process RSS,
- node count,
- edge count,
- \(E/N\),
- maximum degree,
- degree percentiles (at least P50/P90/P99/P99.9),
- feature dimension,
- GPU temperature summary,
- GPU power summary,
- GPU clock summary.

For blockchain datasets, high-degree/hub statistics are mandatory.

---

# 28. DLG Usage Diagnostics

To study conditional local-to-global usefulness, compute where feasible:

- DLG-Aug minus DLG-Base delta for ROC-AUC,
- DLG-Aug minus DLG-Base delta for PR-AUC,
- DLG-Aug minus DLG-Base delta for validation-selected F1,
- raw/adjusted homophily,
- anomaly-conditioned neighborhood mixing,
- feature smoothness / Dirichlet-energy diagnostics where implemented,
- local reconstruction separation,
- embedding norm/variance diagnostics,
- graph density and degree-skew descriptors.

These analyses are explanatory and MUST NOT be used to retroactively remove unfavorable datasets.

---

# 29. Statistical Analysis Plan

Statistical analysis occurs **after seed aggregation**.

The experimental unit for cross-dataset inference is the dataset, not an individual seed.

Unsupported cells are never imputed.

## S1 — All-eight complete-case primary view

For each environment separately:

- include all eight detectors,
- use only the subset of the 13 primary datasets for which all eight are `SUPPORTED_EXACT`,
- perform Friedman omnibus testing for each primary metric where the complete-case block is sufficiently populated,
- use matched pairwise Wilcoxon signed-rank tests,
- apply Holm correction.

## S2 — Frozen continuity five-detector view

The predeclared five-detector continuity subset is:

- DOMINANT
- CoLA
- OCGNN
- DLG-Base
- DLG-Aug

This subset is retained because these five formed the fully supported scalable view in Benchmark v1.

For each environment separately, use all primary datasets on which all five are `SUPPORTED_EXACT`.

## S3 — Real-label financial/blockchain view

Use the six predeclared real-label primary datasets:

- Elliptic
- DGraphFin
- BitcoinOTC
- Ethereum
- BSC
- Polygon

Analyze the predeclared detector set using complete cases only.

This is the preferred real-domain usage view.

## S4 — Synthetic-injection view

Use the seven `-Syn` primary datasets.

Interpret conclusions as controlled injected-anomaly behavior only.

## S5 — LANL external validation

LANL is descriptive/external and is not added retroactively to the 13-dataset omnibus test.

### Statistical interpretation rule

Failure to reject a null hypothesis is not evidence of equivalence.

Average rank MUST always be reported with:
- absolute metric values,
- support coverage,
- uncertainty,
- and significance limitations.

---

# 30. Seed Aggregation

For each supported model × dataset × environment cell, summarize the five model seeds using:

- mean,
- sample standard deviation,
- median,
- min/max or range,
- seed-level 95% confidence interval.

Raw per-seed scores and logs MUST be preserved.

---

# 31. Benchmark v1 versus v2 Versioning

Benchmark v1 is the historical preprint-era experiment: ten primary datasets plus LANL, legacy8-GB-class laptop/WSL CUDA stack. Preserve its files and identity; a detected bug must be disclosed rather than defended as validated merely because it is historical.

Benchmark v2 under A02 is a distinct qualified protocol: thirteen primary datasets plus LANL, current3090 eGPU24 GB production,4070 selected development/resource diagnostics, modern CUDA-first stack, original model semantics with full declared graph or fail closed. ROCm and broad reduction studies are deferred/conditional.

Do not merge v1/v2 numbers or hardware strata. Record whether any v2 results preceded A02 adoption. If so, describe new results as amended reruns rather than a retrospectively preregistered experiment. Release each paper and environment snapshot independently.

---

# 32. Reproducibility Repository Layout

```text
<repo>/
  .venv_old/                           # private historical archive
  .venv_cuda/                          # current runtime
  .venv_rocm/                          # future runtime only
  environment/{constraints,locks,toolchains}/
  research_core/{backends,artifacts,execution,metrics,reporting}/
  configs/common/
  configs/backends/
  configs/tasks/{benchmark,stream,tds,llm}/
  tests/{core,benchmark,stream,tds}/
  benchmark_v2/
    PROTOCOL/                         # original + amendment + effective + work order
    environment/{legacy,journal_cuda}/
    manifests/{datasets,splits,models,runs}/
    results/{production_3090_24g,memory_8g_vs_24g,external_lanl}/
    diagnostics/
    scripts/
    logs/{rtx4070,rtx3090}/
    analysis/{tables,statistics,figures}/
    paper_ready/
  evaluation/stream/v2/
  evaluation/tds/v2/
  releases/
```

This is a thin incremental layout, not a requirement to rewrite working model code. Reuse existing modules through documented adapters where equivalent. Keep secrets, environment bytes, caches and restricted raw data out of source control; preserve protocol/lock/hash/probe manifests and reconstruction instructions.

---

# 33. Run and Publication Manifest Requirements

Every run records protocol/adoption/amendment identity, source commit/dirty patch/adapter hash, environment lock/toolchain hash, study/task/prediction unit/supervision/evidence lane, all dataset/split/feature/edge/label and ordered-ID hashes, model/config/seed-role identity, checkpoint/calibration/selection lineage, backend/device/physical VRAM/allocator cap, precise library/compiler/runtime/driver versions, precision/compile policy, UTC times and resource summaries, support/failure status, artifact paths and hashes, metric-definition/statistics-plan IDs and paper eligibility.

Work Order §18 provides the complete field list. The dispatcher uses separate job/attempt identifiers and queue states; NOT_RUN or DEFERRED is not a zero performance value or an experimentally established unsupported status.

Write immutable input manifests before execution, temporary attempt outputs during execution, and validated atomic commits on completion. Preserve failed attempts. A paper row resolves to one approved result and its raw evidence; no manual table overwrites, duplicate score substitution or cross-backend filling is allowed.

Task-specific temporal/reference/state fields remain in Stream/TDS profiles. Their success means conformity with their declared bounded temporal task, not Benchmark-style global full-graph exactness.

---

# 34. Protocol Amendment Rule

After this freeze, an amendment is allowed only for:

- confirmed implementation bugs,
- required dependency/backend compatibility corrections,
- data-integrity corrections,
- numerical-equivalence failures,
- hardware failure/replacement.

An amendment is NOT allowed solely because:

- a detector performs poorly,
- DLG loses rank,
- a model is unsupported,
- a statistical test is unfavorable,
- a dataset produces an inconvenient result.

Every amendment must include:

1. amendment ID,
2. date,
3. reason,
4. affected cells,
5. old behavior,
6. new behavior,
7. whether any result had already been inspected,
8. mandatory rerun scope.

If a bug affects a model path, **all affected cells must be rerun**. Old and corrected results must not be selectively mixed.

---

# 35. Pre-Run Qualification Checklist — A02

## Governance and migration
- [ ] A02 author adoption and prior-v2-result-inspection declaration completed.
- [ ] Original FROZEN and A01 work order hashes preserved.
- [ ] Existing `.venv` inventory/archive verified and renamed to `.venv_old`.
- [ ] `.venv_cuda` newly created; old activation/shebang paths not reused.
- [ ] Actual source/config/data availability and affected-cell inventory recorded.

## Data and model identity
- [ ] 13 primary + LANL manifests, native-chain labels/provenance and frozen splits verified.
- [ ] Synthetic injection settings and model hyperparameters match the declared protocol.
- [ ] PR-AUC implementation, threshold/tie/undefined rules identified and versioned.
- [ ] Base-70 is global epoch70, not hidden dimension70.
- [ ] All eight models smoke-test; exact reconstruction/fused tests pass.
- [ ] CONAD augmentation/contrastive/gradient/aggregation audit is resolved.
- [ ] DLG-Base/Aug architecture and pretraining/freeze paths verified.

## Software and current hardware
- [ ] Exact Python/PyTorch/PyG/PyGOD and all required dependencies locked.
- [ ] Toolchain/ABI/extensions/precision/cache policy recorded.
- [ ] Full CUDA lock recreated and qualification repeated.
- [ ] Actual4070/3090 identity, physical VRAM, host RAM, WSL/OS and eGPU link recorded.
- [ ] Background load controlled, available telemetry and timing scope verified.
- [ ]3090 three-dataset/five-model preflight passed or unresolved cells blocked.
- [ ] Selected memory diagnostic cells and allocator-cap bytes/method declared.

## Evidence and scope
- [ ] Task isolation, future-reference/future-edge negative fixtures and manifest validator pass.
- [ ] Production registry is empty/new or historical attempts explicitly preserved.
- [ ] No claim of ROCm qualification is made without actual ROCm evidence.
- [ ] No Stream/TDS/LLM full experiment is required to release this Benchmark.

---

# 36. Final Effective Decision Rules — A02

1. Thirteen primary datasets plus one external LANL dataset; real6 and synthetic7 are separate.
2. Eight primary detectors; model seeds42–46 and synthetic injection seed42 remain fixed.
3. Preserve original fitting/score semantics, training budgets and validation-only threshold policy.
4. Current production uses qualified3090 eGPU24 GB CUDA;4070 physical8 GB is a selected diagnostic environment.
5. Preserve `.venv` before renaming to `.venv_old`; create `.venv_cuda` afresh. Future `.venv_rocm` is separate.
6. Common source/Python-level versions and build policy are aligned where supported; vendor binaries/compiler/ABI facts are recorded, not falsely declared identical.
7. Full declared graph with original semantics is the sole primary path; no new truncation/objective/dimension/epoch/CPU fallback.
8. OOM/timeout/dependency/backend/numerical failures are support evidence, never zero scores or worst-rank imputations.
9. A same3090 approximately8-GiB allocator cap is diagnostic only and not physical8-GB emulation.
10. CUDA/ROCm and4070/3090 strata are never silently pooled; future ROCm requires its own qualification.
11. Historical cut/random sampling/partition or host assistance is conditional and separate; no primary substitution.
12. Mandatory CONAD audit and exact/fused equivalence precede production.
13. Statistical inference uses dataset-level aggregates, the original complete-case views and explicit support coverage.
14. Stream/TDS reuse core tools but keep distinct units, splits, labels, seeds, calibration and statistical contracts.
15. Any correction requires explicit version/impact accounting and all affected outputs to be corrected or rerun.
16. Complete core evidence, freeze the release, prepare Preprints.org registration, then the journal submission package; do not add unrelated ROCm/LLM research to the stop rule.

---

# 37. Protocol Interpretation

Benchmark v2 is designed to make three forms of evidence distinguishable:

### Predictive evidence
How accurately does a detector rank anomalies?

### Execution-support evidence
Can the original detector semantics be executed under the declared software/backend/memory envelope?

### Fidelity evidence
If graph reduction is studied, how far does the reduced execution deviate from the exact full-graph reference?

No one of these is allowed to masquerade as another.

The central methodological position is:

> **A detector that cannot execute exactly under a declared environment is not assigned an artificial predictive score, and an approximate graph-reduction result is not silently substituted for an exact full-graph result.**

This effective edition, the preserved historical parent and the A02 amendment govern the declared A02 reruns.

# 38. Shared Development Process and Paper-Specific Contracts

The normative implementation companion is the A02 Work Order, especially Phase A/B and §§27–30, plus `DLG_Family_Porting_Profiles_A02_2026-10-02.md`.

Use a small common layer for backend probes, locked environments, input/result hashing, fresh-process jobs, task-aware metrics, telemetry, validation and publication export. Model/data code stays in task adapters. Unimplemented interface names in the work order are development requirements, not claims of existing APIs.

Stream retains its contract-snapshot supervised GIN/GATv2 selective architecture, seeds11/22/33/44/55, deterministic primary routing, validation-calibrated operating points and separated offline/runtime/replay lanes. TDS retains its own causal-event profile, seeds7/17/27/37/47, temporal graph/state semantics, fixed0.5 primary F1, validation temperature scaling and MC/ensemble controls. Their shared total24,316 does not prove shared rows/labels/splits; reconcile raw lineage before reuse.

A future LLM adapter shares artifact/environment/job procedures, not GNN graph metrics or statistical units. CUDA and ROCm cooperate initially through independent processes and CPU-visible artifacts. Preserve each paper's completed environment and evidence release; newer LLM requirements cannot mutate it in place.

Archive manuscript-ready tables, figures, generated number macros and claims-to-evidence mappings. Use the original DLG-GNN DOI as a citation, not a new Benchmark identifier. Record the Benchmark's own preprint DOI/version after registration, disclose related manuscripts/shared data, and recheck journal instructions on submission. No acceptance or cross-vendor interoperability guarantee is implied by this document.

