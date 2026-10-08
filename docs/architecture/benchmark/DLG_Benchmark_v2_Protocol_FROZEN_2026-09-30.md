# DLG-GNN Benchmark v2 Protocol — FROZEN PRE-RUN SPECIFICATION

> **Historical pre-run record.** Current A04 development, qualification, and publication decisions, including the Python 3.14.4 / PyTorch 2.14.1+cu130 environment and diagnostic-only CONAD, are in [the A04 amendment](DLG_Benchmark_v2_Protocol_Amendment_A04_2026-10-03.md). Historical Round5 metric runs used an older execution environment. Preserve the original freeze below as a dated protocol record.


**Protocol ID:** DLG-BENCH-V2.0  
**Freeze date:** 2026-09-30 (Asia/Seoul)  
**Status:** **FROZEN BEFORE V2 PERFORMANCE RUNS**  
**Purpose:** Journal-version evaluation protocol for the DLG-GNN benchmark  
**Target use:** Applied Sciences Special Issue — *Graph Neural Networks: Theory, Methods and Applications*

> **Freeze principle.** Benchmark v2 is defined before inspecting any v2 performance result.  
> Performance, support, sampling, partitioning, backend, or memory policies MUST NOT be changed after result inspection in order to improve a model's apparent rank or coverage. Any required correction must be recorded as a versioned protocol amendment and all affected cells must be rerun.

---

## 1. Benchmark v2 Research Objective

Benchmark v2 evaluates **how DLG-GNN should be used reliably across graph domains and computational environments**, rather than asking only whether a single DLG variant achieves the best predictive score.

The benchmark therefore separates four questions:

### RQ1 — Cross-domain and native-domain predictive behavior
How do DLG-Base and DLG-Aug behave across:
1. heterogeneous cross-domain graph anomaly datasets,
2. the original blockchain application domains of Ethereum, BSC, and Polygon, and
3. an external cybersecurity domain?

### RQ2 — Conditional local-to-global utility
Under what graph and anomaly conditions does the DLG local-to-global augmentation improve, preserve, or degrade anomaly detection performance?

### RQ3 — Execution support and portability
How stable are detector results and execution support across:
- a modernized software tool-chain,
- NVIDIA CUDA and AMD ROCm backends, and
- declared 8 GB and 24 GB GPU-memory envelopes?

### RQ4 — Memory-constrained graph fidelity
When exact full-graph execution is not feasible, how much do graph-reduction strategies alter detector scores and rankings relative to the full graph?

The primary scientific principle is:

> **Preserve detector semantics first; treat unsupported execution as evidence rather than silently changing the problem.**

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

# 10. Modernized Software Stack

Benchmark v2 uses a modernized stack rather than treating the v1 two-year-old accelerator environment as the journal-version reference.

## 10.1 Frozen target stack

```text
Python             3.12.x
PyTorch            2.11.x
PyTorch Geometric  2.8.x
PyGOD              1.1.0
CUDA               12.8.x family
ROCm               7.14.1
```

### Patch-version rule

The exact Python, PyTorch, PyG, CUDA runtime/driver, compiler, and backend-extension builds MUST be locked **after environment qualification but before production runs**.

The final repository must contain:

```text
environment/
  requirements-v2-common.lock.txt
  requirements-v2-cuda128.lock.txt
  requirements-v2-rocm7141.lock.txt
  system-cuda.json
  system-rocm.json
```

Common Python-level package versions SHOULD match across CUDA and ROCm. Backend-specific binary packages MAY differ only where required by the accelerator backend.

No package version may be upgraded mid-benchmark without a protocol amendment.

---

# 11. Environment Qualification Gate

No 13+1 production benchmark is allowed until both modern environments pass qualification.

Qualification includes:

1. import and model-instantiation tests for all eight detectors,
2. small-graph forward/backward smoke tests,
3. deterministic seed verification,
4. score-orientation verification,
5. exact sparse reconstruction equivalence tests,
6. fused GCN equivalence tests,
7. DLG-Base / DLG-Aug training-path verification,
8. CONAD augmentation-path verification,
9. memory telemetry validation,
10. result-manifest generation validation.

A failed qualification is an environment result, not a reason to silently alter a detector.

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

# 13. Hardware Envelopes

Benchmark v2 uses a dedicated test server with:

```text
System RAM: 64 GB
```

and two accelerator environments when available:

## H8-CUDA — physical 8 GB CUDA envelope

```text
GPU: NVIDIA RTX 3060 Ti
VRAM: 8 GB physical
Backend: CUDA 12.8-family qualified build
```

Purpose:

- modern CUDA reference,
- continuity with the former 8 GB-class resource envelope,
- full-graph support/failure characterization.

## H24-ROCm — physical 24 GB memory-relaxed envelope

```text
GPU: AMD Radeon RX 7900 XTX
VRAM: 24 GB physical
Backend: ROCm 7.14.1 qualified build
```

Purpose:

- memory-relaxed full-graph evaluation,
- ROCm portability,
- recovery analysis for cells that fail under 8 GB.

## H8cap-ROCm — diagnostic 8 GB allocation cap

The 7900 XTX MAY be software-limited to an approximately 8 GB process allocation envelope for diagnostic experiments.

This condition MUST be labeled:

```text
ROCm 8-GB allocation cap
```

and MUST NOT be described as physically equivalent to an 8 GB GPU.

It is used only to help isolate memory-headroom effects within the same AMD hardware/backend.

---

# 14. Dedicated-Server Stability Policy

Benchmark v2 removes the laptop thermal/power-management confound from production timing/resource measurements.

Production runs must:

- execute on the dedicated test server,
- prohibit unrelated GPU workloads,
- use vendor-default clocks unless a fixed non-overclocked policy is explicitly declared,
- prohibit overclocking,
- prohibit adaptive manual tuning between model runs,
- log GPU temperature, power, core clock, memory clock, utilization, and VRAM use at a fixed interval,
- log host RAM and process RSS,
- record driver and kernel information,
- execute each model × dataset × seed cell in a fresh process.

Cross-vendor CUDA-vs-ROCm wall-clock results MUST NOT be interpreted as a direct GPU architecture speed competition.

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

# 18. 8 GB vs 24 GB Evaluation Policy

The 8 GB and 24 GB environments are analyzed as **separate declared resource envelopes**.

### Rule A — No merged hardware ranking

A result obtained on H8-CUDA MUST NOT be silently mixed with an H24-ROCm result to form a single "same-environment" ranking.

Environment-specific performance/support matrices are retained separately.

### Rule B — Cross-environment intersection

CUDA/ROCm numerical comparison uses only model × dataset × seed cells supported in both environments.

### Rule C — Memory recovery analysis

For every H8-CUDA cell classified `UNSUPPORTED_RESOURCE_OOM`, rerun the same semantic configuration on H24-ROCm.

Classify:

```text
8 GB OOM -> 24 GB supported     : memory-envelope dependent
8 GB OOM -> 24 GB OOM           : unresolved / >24 GB or implementation bottleneck
8 GB supported -> 24 GB supported: shared-support case
```

### Rule D — H8cap-ROCm diagnostic trigger

A ROCm 8-GB allocation-cap rerun is triggered for:
- H8-CUDA OOM cells, and
- selected high-memory shared-support controls.

The capped experiment is diagnostic only.

---

# 19. CUDA / ROCm Portability Evaluation

The portability study asks whether the same Python-level model semantics remain operationally and numerically consistent across accelerator backends.

For shared supported cells, record:

- ROC-AUC difference,
- PR-AUC difference,
- validation-selected F1 difference,
- raw-score Pearson correlation,
- raw-score Spearman correlation,
- seed-wise variance,
- peak allocated VRAM,
- peak reserved VRAM where meaningful,
- process RSS,
- total wall time,
- support status.

Timing is descriptive only and MUST NOT be used to claim that CUDA or ROCm is inherently faster based on the RTX 3060 Ti vs RX 7900 XTX comparison.

---

# 20. Graph Reduction Policy

Graph reduction is **not** allowed as a substitute for the exact primary result.

It is evaluated separately to answer RQ4.

The four graph-context modes are:

```text
F0 = Full graph exact
F1 = Structure-aware partition / halo execution
F2 = Random induced-subgraph sampling
F3 = Historical index cut
```

Only F0 may populate the exact primary performance matrix.

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

# 31. Benchmark v1 vs v2 Versioning

Benchmark v1 remains historically valid as the preprint-era frozen experiment.

Benchmark v2 is a **new predeclared journal protocol**.

The paper/repository must state this explicitly.

Recommended terminology:

```text
Benchmark v1:
  10 primary datasets + LANL external
  legacy 8-GB-class laptop/WSL2 environment
  legacy CUDA/PyTorch/PyG stack

Benchmark v2:
  13 primary datasets + LANL external
  dedicated 64-GB-RAM test server
  modern CUDA and ROCm environments
  physical 8-GB and 24-GB resource envelopes
  full-graph exact primary policy
  explicit reduction-fidelity diagnostics
```

v1 and v2 results MUST NOT be merged as if they were generated under one identical protocol.

---

# 32. Reproducibility Repository Layout

Recommended frozen structure:

```text
./evaluation/benchmark/v2/
├── PROTOCOL/
│   ├── DLG_Benchmark_v2_Protocol_FROZEN_2026-09-30.md
│   └── protocol_amendments.md
├── environment/
│   ├── requirements-v2-common.lock.txt
│   ├── requirements-v2-cuda128.lock.txt
│   ├── requirements-v2-rocm7141.lock.txt
│   ├── system-cuda.json
│   └── system-rocm.json
├── manifests/
│   ├── datasets/
│   ├── models/
│   ├── splits/
│   └── runs/
├── results/
│   ├── exact/
│   │   ├── cuda8/
│   │   └── rocm24/
│   ├── diagnostics/
│   │   ├── rocm8cap/
│   │   ├── partition/
│   │   ├── random_sampling/
│   │   └── historical_index_cut/
│   └── portability/
├── logs/
│   ├── cuda8/
│   └── rocm24/
└── analysis/
```

---

# 33. Run Manifest Requirements

Every run must write a machine-readable manifest containing at least:

```text
protocol_id
git_commit
dataset_id
dataset_hashes
model_id
model_config_hash
model_seed
injection_seed
sampling_seed (if applicable)
partition_id (if applicable)
backend
gpu_model
physical_vram
memory_cap (if applicable)
python_version
torch_version
pyg_version
pygod_version
driver_version
cuda_or_rocm_version
start_time
end_time
support_status
failure_reason
peak_gpu_memory
peak_host_rss
result_artifact_hash
```

A performance table row without a valid run manifest is not considered a production Benchmark v2 result.

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

# 35. Pre-Run Qualification Checklist

The protocol remains frozen, but production execution MUST NOT start until every item below is checked.

## Dataset freeze
- [ ] 13 primary + LANL artifact manifests complete
- [ ] Ethereum manifest complete
- [ ] BSC manifest complete
- [ ] Polygon manifest complete
- [ ] all labels/provenance verified
- [ ] all split masks exported and hashed
- [ ] all synthetic injection configs verified

## Software freeze
- [ ] Python 3.12 exact patch frozen
- [ ] PyTorch 2.11 exact builds frozen
- [ ] PyG 2.8 exact build frozen
- [ ] PyGOD 1.1.0 frozen
- [ ] CUDA 12.8 exact driver/runtime recorded
- [ ] ROCm 7.14.1 stack recorded
- [ ] PyG backend-extension versions recorded

## Model validation
- [ ] all 8 models import
- [ ] all 8 models complete small-graph smoke test
- [ ] exact reconstruction equivalence passes
- [ ] fused message-passing equivalence passes
- [ ] DLG-Base path verified
- [ ] DLG-Aug local-pretrain/freeze/global path verified
- [ ] CONAD contrastive path verified
- [ ] score orientation verified

## Hardware validation
- [ ] RTX 3060 Ti 8 GB telemetry verified
- [ ] RX 7900 XTX 24 GB telemetry verified
- [ ] dedicated-server background load controlled
- [ ] thermal/power/clock logging verified
- [ ] 8-GB ROCm cap method documented if used

## Reproducibility
- [ ] repository commit frozen
- [ ] protocol file hash stored
- [ ] environment lock files committed
- [ ] result directory empty or archived before first production run

---

# 36. Final Frozen Decision Rules

The following rules are the core Benchmark v2 contract.

1. **13 primary datasets + 1 external LANL dataset.**
2. **Ethereum, BSC, and Polygon use real external fraud labels.**
3. **Eight primary detector configurations only.**
4. **Five model seeds: 42–46.**
5. **Synthetic injection seed: 42.**
6. **Modern stack: Python 3.12 / PyTorch 2.11 / PyG 2.8 / PyGOD 1.1 / CUDA 12.8 / ROCm 7.14.1.**
7. **Full graph exact execution is the only path allowed into the primary exact performance matrix.**
8. **OOM is reported as OOM; it is not hidden with truncation.**
9. **Physical 8 GB CUDA and physical 24 GB ROCm are separate declared envelopes.**
10. **ROCm 8-GB allocation cap is diagnostic only.**
11. **Structure-aware partitioning is a separate scalability/fidelity experiment.**
12. **Random sampling is diagnostic only.**
13. **Historical index cut is diagnostic/history only and is prohibited as a new primary method.**
14. **No zero/worst-rank imputation for unsupported cells.**
15. **No hidden CPU, epoch, dimension, or objective fallback.**
16. **CUDA/ROCm performance matrices are not silently merged.**
17. **LANL remains external to the 13-dataset omnibus inference.**
18. **Real-label and synthetic subsets are reported separately.**
19. **CONAD–DOMINANT execution-path audit is mandatory before production runs.**
20. **Any post-freeze change requires an explicit protocol amendment and affected-cell rerun.**

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

This protocol is the frozen basis for all DLG-GNN Benchmark v2 reruns.

