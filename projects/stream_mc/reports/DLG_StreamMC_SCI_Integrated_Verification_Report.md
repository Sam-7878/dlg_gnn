# DLG-StreamMC SCI Integrated Verification Report

**Overall Status:** `NOT_READY`  
**Submission Readiness Score:** `32.0/100`  
**Valid Experiments:** `0 / 1`  
**Critical / High Issues:** `1 / 1`  
**Temporal Leakage:** `DATASET_AUDIT_FAIL`  
**Baseline Fairness:** `NOT_DIRECTLY_COMPARABLE`  
**Reproducibility Grade:** `F`  

**Top Blocking Issue:** Sample-level leakage audit is not PASS; no paper-eligible experiment results exist.

---

## 1. Executive Summary

현재 구현 코어와 106-test suite는 검증됐지만 immutable experiment result가 전혀 없어 SCI 성능 주장은 검증할 수 없다. 최종 판정은 **NOT_READY**이다.

---

## 2. Scope and Version Baseline

- Repository: `/mnt/d/_Work/goat_bank/dlg_gnn`
- Branch / SHA: `experiment/dlg-streammc-sci-v1` / `eb1fafc4b5f370edcc3739b5a953ec68492c34d3`
- Dirty: `True`
- Python / PyTorch / PyG / PyGOD / CUDA: `3.12.13` / `2.5.1+cu121` / `2.7.0` / `1.1.0` / `12.1`
- Hardware: `{"cpu_count": 32, "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU", "machine": "x86_64", "processor": "x86_64", "ram_bytes": 20971339776}`
- Generated: `2026-07-29T05:46:10.522080+00:00`
- Dataset basis requested: `/mnt/d/_Work/_data/GoG`
- Manifest source observed: `/mnt/d/_Work/_data/dataset/transactions`
- Included experiment period: `strict prerequisite run only`

---

## 3. Repository Implementation Status

| Component | Required | Implemented | Tested | Evidence | Status |
|---|---|---|---|---|---|
| Temporal split | Yes | True | True | src/gog_fraud/data/splits/temporal_split.py, tests/data/test_temporal_integrity.py | PASS |
| Leakage validator | Yes | True | True | src/gog_fraud/data/validation/temporal_leakage.py, tests/data/test_temporal_integrity.py | PASS |
| Stateful stream | Yes | True | True | src/gog_fraud/data/io/streaming_dataset.py, tests/streaming/test_stateful_stream.py | PASS |
| Incremental L1 store | Yes | True | True | src/gog_fraud/streaming/subgraph_store.py, tests/streaming/test_bounded_state.py | PASS |
| Incremental L2 graph | Yes | True | True | src/gog_fraud/streaming/relation_state.py, tests/streaming/test_bounded_state.py | PASS |
| MC inference | Yes | True | True | src/gog_fraud/models/extensions/mc/mc_dropout.py, tests/unit/test_mc_dropout.py | PASS |
| Dual-threshold router | Yes | True | True | src/gog_fraud/selection/router.py, tests/selection/test_router.py | PASS |
| Risk-sensitive router | Yes | True | True | src/gog_fraud/selection/router.py, tests/selection/test_router.py | PASS |
| TTL/LRU cache | Yes | True | True | src/gog_fraud/streaming/embedding_cache.py, tests/streaming/test_bounded_state.py | PASS |
| Queue/backpressure | Yes | True | True | src/gog_fraud/streaming/queue_manager.py, tests/streaming/test_bounded_state.py | PASS |
| Checkpoint/recovery | Yes | True | True | src/gog_fraud/streaming/checkpoint.py, tests/streaming/test_stateful_stream.py | PASS |
| Latency profiler | Yes | True | True | src/profiling/streaming_profiler.py, tests/profiling/test_streaming_profiler.py | PASS |
| Memory profiler | Yes | True | True | src/profiling/streaming_profiler.py, tests/profiling/test_memory_slope.py | PASS |
| Result provenance | Yes | True | True | src/gog_fraud/experiments/manifest.py, tests/experiments/test_provenance.py | PASS |

---

## 4. Dataset and Label Audit

| Chain | Start | End | Transactions | Contracts | Addresses | Fraud | Benign | Unlabeled | Positive Ratio | Missing TS | Duplicates | Evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bsc | 1599277696 | 1709251192 | 121612480 | 7499 | 6550399 | 6394 | 1105 | 0 | 0.8526470196026137 | 0 | 0 | results_sci/manifests/bsc.json |
| ethereum | 1455214837 | 1709251199 | 81788211 | 14464 | 10247767 | 8433 | 6031 | 0 | 0.5830337389380531 | 0 | 0 | results_sci/manifests/ethereum.json |
| polygon | 1598455127 | 1709251199 | 64882233 | 2353 | 1801976 | 2291 | 62 | 0 | 0.9736506587335316 | 0 | 0 | results_sci/manifests/polygon.json |

Dataset audit status: `COMPLETE`. Full Ethereum, BSC, and Polygon manifests are present. Label ratios describe this fraud-oriented labeled corpus and are not population prevalence estimates.

---

## 5. Temporal Integrity and Leakage Verification

Fixed and rolling split artifacts present: `True`. Sample-level status: `FAIL` with `358` raw within-contract ordering violations. Processed feature/relation/normalizer/KNN provenance remains unobservable. Overall leakage status: `DATASET_AUDIT_FAIL`. Main results therefore cannot be marked VALID.

---

## 6. Experimental Protocol Matrix

`MISSING`: no experiment rows under `results_sci`.

---

## 7. Baseline Fairness Audit

**NOT_DIRECTLY_COMPARABLE** — PyGOD, DLG-GNN, StreamMC results with a common split/config are absent.

---

## 8. Main Detection Results

`NOT_RUN`: ROC-AUC, PR-AUC, F1, recall, CI source rows are absent.

---

## 9. Selective Inference and Routing Analysis

`NOT_RUN` for paper evidence. Three 100-contract smoke traces exist only for path verification and are explicitly marked non-paper-eligible.

---

## 10. Monte Carlo Sensitivity

`NOT_RUN`: required T={1,3,5,8,10,20,30} result matrix absent.

---

## 11. Calibration Analysis

`NOT_RUN`: NLL/Brier/ECE/reliability source data absent.

---

## 12. Streaming Evaluation

**SMOKE_ONLY_NOT_PAPER_ELIGIBLE**: deterministic replay, checkpoint primitives, and bounded queue/cache tests pass; three 100-contract smoke paths completed. Normal/burst/overload/cache-pressure/100k-event paper scenarios are `NOT_RUN`.

---

## 13. Resource Evaluation

`NOT_RUN`: VRAM/RSS/cache/queue/memory-slope and LPP comparison absent.

---

## 14. Latency and Throughput

`NOT_RUN`: cold-start and steady-state component timing rows absent.

---

## 15. Ablation Summary

`NOT_RUN`: MC/routing/L2/fusion/legacy/LPP ablations absent.

---

## 16. Temporal Robustness

`NOT_RUN`: rolling-origin result folds absent.

---

## 17. Cross-Chain Generalization

`NOT_RUN`: held-out chain matrix absent.

---

## 18. Statistical Verification

`NOT_RUN`: bootstrap/DeLong/Wilcoxon/Friedman/Nemenyi evidence absent.

---

## 19. Failure, Exception, and Exclusion Audit

| Exp ID | Failure Type | Count | Affected | Resolution | Included |
|---|---|---|---|---|---|
| 20260729T052803Z_dlg_streammc_main_636630ca32d1 | RuntimeError | 1 | none; prerequisite gate failure | sample-level leakage audit not PASS: ethereum; sample-level leakage audit not PASS: bsc; sample-level leakage audit not PASS: polygon; strict evaluation requires explicit real pipeline_commands; no paper stage is configured | True |

---

## 20. Paper Claim Verification Matrix

| Claim | Required Evidence | Available | Result | Status |
|---|---|---|---|---|
| comparable to PyGOD | main baseline table, identical split, 95% CI | none | NOT_RUN | MISSING_EVIDENCE |
| reduced deep inference | sample routing trace and deep-route ratio | none | NOT_RUN | MISSING_EVIDENCE |
| bounded memory | 100k+ event memory trajectory and slope | none | NOT_RUN | MISSING_EVIDENCE |
| lower latency | fair cold/steady latency comparison | none | NOT_RUN | MISSING_EVIDENCE |
| better calibration | ECE/Brier/NLL and reliability data | none | NOT_RUN | MISSING_EVIDENCE |
| streaming capable | stateful/recovery tests and scenario replay | unit-level deterministic replay/checkpoint tests | core state behavior verified; production/load scenario evidence missing | PARTIALLY_SUPPORTED |
| multi-chain robust | per-chain and held-out cross-chain results | none | NOT_RUN | MISSING_EVIDENCE |
| analyst workload reduction | review rate and direct-exit FNR | none | NOT_RUN | MISSING_EVIDENCE |

---

## 21. Result Consistency Checks

| Issue | Severity | Category | Message | Resolution |
|---|---|---|---|---|
| ISS-001 | CRITICAL | experiment | No immutable scientific experiment result rows exist; detection, routing, calibration, resource, and statistical claims are not verified. | UNRESOLVED |
| ISS-002 | MEDIUM | label | ethereum fraud positive ratio is 0.583; label semantics and fraud-oriented corpus sampling must be disclosed. | UNRESOLVED |
| ISS-003 | MEDIUM | label | bsc fraud positive ratio is 0.853; label semantics and fraud-oriented corpus sampling must be disclosed. | UNRESOLVED |
| ISS-004 | MEDIUM | label | polygon fraud positive ratio is 0.974; label semantics and fraud-oriented corpus sampling must be disclosed. | UNRESOLVED |
| ISS-005 | HIGH | temporal | Sample-level leakage audit is not PASS: bsc_leakage_audit_v1.json, ethereum_leakage_audit_v1.json, polygon_leakage_audit_v1.json, pooled_leakage_audit_v1.json | UNRESOLVED |

---

## 22. Reproducibility Assessment

Score: **32.0/100**, Grade **F**. Dependency lock, three-chain manifests, and split artifacts are present; leakage freedom and paper-result provenance remain incomplete.

---

## 23. SCI Submission Readiness

Decision: **NOT_READY** (32.0/100).

### Weighted Assessment

| Area | Points |
|---|---|
| method_implementation | 15.0 |
| dataset_integrity | 5.0 |
| experimental_fairness | 0.0 |
| statistical_rigor | 0.0 |
| streaming_evidence | 5.0 |
| selective_inference_evidence | 4.0 |
| resource_evaluation | 0.0 |
| reproducibility | 3.0 |

### Blocking Issues

- No immutable scientific experiment result rows exist; detection, routing, calibration, resource, and statistical claims are not verified.
- Sample-level leakage audit is not PASS: bsc_leakage_audit_v1.json, ethereum_leakage_audit_v1.json, polygon_leakage_audit_v1.json, pooled_leakage_audit_v1.json

### Non-blocking Issues

- ethereum fraud positive ratio is 0.583; label semantics and fraud-oriented corpus sampling must be disclosed.
- bsc fraud positive ratio is 0.853; label semantics and fraud-oriented corpus sampling must be disclosed.
- polygon fraud positive ratio is 0.974; label semantics and fraud-oriented corpus sampling must be disclosed.

### Recommended Next Actions

1. Repair raw ordering and regenerate processed graph feature/relation/normalizer provenance, then rerun leakage audit
1. Run fair PyGOD/DLG/StreamMC baselines
1. Run 5-seed MC/routing/calibration experiments
1. Run 100k-event streaming/resource scenarios
1. Run temporal and cross-chain generalization
1. Run confidence intervals and significance tests

---

## Appendix A. Complete Experiment Registry

`EMPTY`

---

## Appendix B. Complete Metric Tables

`EMPTY`

---

## Appendix C. Config and Hyperparameter Index

See machine-readable JSON and evidence index.

---

## Appendix D. Evidence Index

217 evidence records. See `DLG_StreamMC_SCI_Evidence_Index.csv`.

---

## Appendix E. Test Results

Status: `PASS`; passed=106, failed=0, warnings=23.

---

## Appendix F. Failure Logs

The strict fail-closed prerequisite run is retained under `results_sci/manifests/*/run_manifest.json` and `audit.json`; see Section 19.

---

## Appendix G. Generated Figures

`EMPTY`: no figure source data.

---

## Appendix H. Claim-to-Evidence Trace

See Section 20 and report JSON `claim_verification`.
