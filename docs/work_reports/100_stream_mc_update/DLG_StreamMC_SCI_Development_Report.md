# DLG-StreamMC SCI 개발 및 검증 보고서

작성일: 2026-07-28  
기준 브랜치/커밋: `main` / `f71ca5a15f4595a4650c4ac5357578daa9a43af7`  
개발 환경: WSL2 Ubuntu 24.04, Python 3.12.13, PyTorch 2.5.1+cu121, PyG 2.7.0, PyGOD 1.1.0

## 1. Executive Summary

본 작업은 작업지시서의 P0 항목 중 논문 수치 재생성에 앞서 필요한 시간 무결성, stateful stream, bounded state, selective routing, MC-Dropout, provenance 기반을 구현했다. 기존 `run_sci_evaluation.py`에서 실제 실행 없이 chain 성능과 latency를 추정하던 경로는 제거했다. 새 오케스트레이터는 명시된 실제 experiment command가 성공하지 않는 한 논문 metric을 생성하지 않는다.

현재 상태는 **개발 기반 완료, SCI 실험 미완료(`PARTIAL / NOT_SUBMISSION_READY`)**이다. 실제 PyGOD/DLG-GNN/DLG-StreamMC 재학습, 5-seed/5-fold 실험, cross-chain, calibration, 장기 streaming, significance test는 아직 실행하지 않았으므로 논문 성능 우월성은 주장할 수 없다.

## 2. Baseline Audit

- 작업 시작 전 사용자 변경: `scripts/fig_roc_auc_comparison.py`, `scripts/scratch_stats.py`
- 위 두 파일은 수정하지 않았다.
- 기존 `outputs/`와 `results/`는 이동·삭제하지 않았다. 사용자 결과를 자동 변경하지 않고 새 결과 root를 `results_sci/`로 분리했다.
- 데이터 경로 확인:
  - transactions: `/mnt/d/_Work/_data/dataset/transactions/{ethereum,bsc,polygon}`
  - labels: `/mnt/d/_Work/_data/dataset/labels.csv`
  - legacy GoG artifacts: `/mnt/d/_Work/_data/GoG`
- 기존 SCI orchestrator에는 graph 수 fallback, chain 비율, 성능 및 latency 파생값을 이용한 synthetic reporting 경로가 있었다. 이를 provenance-first orchestrator로 대체했다.

## 3. 구현 내용

### 3.1 Temporal integrity

- `temporal_split.py`: `(event_time, sample_id)` 안정 정렬, 70/15/15 split, boundary와 SHA-256 split hash
- `rolling_origin.py`: expanding-window 방식의 5-fold 이상 rolling-origin split
- `temporal_leakage.py`: feature/relation source timestamp, scaler fit boundary, split entity overlap 검사
- test/validation 데이터에서 threshold를 선택하면 예외를 내는 validation-only threshold optimizer

### 3.2 Stateful stream core

- 정규화된 `StreamEvent`와 `StreamCheckpoint`
- deterministic chronological replay, checkpoint/restore, malformed quarantine
- seeded delayed/out-of-order simulation, merged multi-chain replay, replay rate와 event lag
- temporal window 및 node/edge bound를 적용하는 incremental Level-1 subgraph store
- future relation 차단, TTL edge removal, intra/cross-chain metadata를 보존하는 Level-2 relation state
- model/feature version, TTL, LRU, max entry/max byte를 적용하는 host embedding cache
- ingest/direct/deep/review bounded queue와 risk-priority backpressure
- model-agnostic selective streaming engine 및 sample routing trace/prediction hash
- JSON checkpoint save/load primitive

### 3.3 Selective MC

- dual threshold: `tau_b < tau_f`, uncertainty threshold, abstention region
- risk-sensitive override: score risk threshold와 graph risk prior threshold
- 표준 `TriageOutput` 및 `RoutingDecision`
- MC-Dropout에서 Dropout만 train mode로 전환하고 다른 layer는 eval 유지
- estimator 종료 후 원래 model train/eval state 복구
- 고정 seed를 local RNG context에서 사용하여 외부 RNG state 오염 방지
- T=1 variance=0 의미 명시
- mean, variance, standard deviation, predictive entropy, MC latency 기록

### 3.4 Provenance와 schema

- Git SHA/dirty/diff hash, CLI, config, seed, dependency version, dataset/output SHA-256, failure를 포함하는 `RunManifest`
- chain별 dataset JSON/CSV manifest builder
- 작업지시서 long-format 필수 열 및 값 범위를 검사하는 result schema
- `results_sci/{manifests,...,logs}` 구조 생성
- 명시적 `pipeline_commands`만 실행하고 return code/log를 보존하는 SCI orchestrator

### 3.5 기존 결함 수정

- `LearnedFusionNet` 입력 차원이 실제 feature 7개/5개와 달리 16개/8개로 하드코딩돼 forward가 실패하던 문제 수정
- Fusion trainer 반환값에 과거 API의 `total_epochs` alias 복구
- Level-2 bundle 실제 feature width와 config `in_dim`이 다르면 silent truncation 없이 명시적 오류를 발생시켜 잘못된 실험을 차단

## 4. 테스트 증거

실행 명령:

```bash
cd /mnt/d/_Work/goat_bank/dlg_gnn
PYTHONPATH=src /mnt/d/_Work/goat_bank/.venv/bin/python -m compileall -q src/gog_fraud scripts/build_dataset_manifest.py
PYTHONPATH=src /mnt/d/_Work/goat_bank/.venv/bin/python -m pytest -q \
  tests/data tests/streaming tests/selection tests/experiments \
  tests/unit/test_mc_dropout.py tests/unit/test_phase4_fusion.py
```

결과: **60 passed, 1 dependency deprecation warning**.

검증된 핵심 속성:

- temporal ordering/split hash 재현
- rolling-origin expanding window와 split disjointness
- future feature/relation/scaler leakage 탐지
- deterministic replay와 restore suffix 동등성
- malformed quarantine와 multi-chain merge
- subgraph bound/snapshot restore
- cache LRU/TTL/version invalidation
- queue overload risk priority/expiration
- relation future-edge 거부/expiration
- routing boundary/risk override
- validation-only threshold serialization
- streaming trace와 prediction hash 재현
- MC dropout state 복구 및 통계
- LearnedFusion forward/train/save/load/ensemble
- result schema validation

SCI orchestrator smoke test도 성공했고 run manifest와 audit JSON을 생성했다. smoke test는 `--skip-dataset-scan`을 사용했으므로 dataset audit 완료 증거로 간주하지 않는다.

실제 3-chain manifest full scan도 시도했으나 첫 chain에서 장시간 동안 산출물 없이 지속되어 해당 명령의 WSL wrapper/child PID만 확인 후 종료했다. 부분 JSON/CSV는 생성되지 않았다. 따라서 dataset manifest builder의 기능 테스트와 실제 full dataset audit 완료를 구분하며, 후자는 미완료로 판정한다. 후속 작업에서는 chain/file progress, resumable hash index, file-level parallelism을 추가해야 한다.

## 5. 기존 Suite의 계약 불일치

전체 unit suite 감사에서 신규 영역 외 세 가지 assertion 불일치가 남았다.

1. `test_level1_dataset_load_and_infer_dims`는 16차원 `x` fixture를 생성한 뒤 `infer_in_dim == 8`을 기대한다. 구현의 실제 tensor dimension 반환과 테스트 기대가 모순된다.
2. Level-2 코드/architecture는 node-level logits를 primary output으로 정의하지만 과거 forward 테스트 두 건은 batch graph-level shape를 기대한다.
3. 위 항목은 수치를 왜곡할 수 있어 임의로 테스트 기대에 맞추지 않았다. Level-2 public contract를 node-level 또는 graph-level 중 하나로 확정하고 downstream trainer/fusion과 함께 migration해야 한다.

이 세 항목은 신규 P0 test 실패가 아니라 기존 API 계약 blocker이며, main experiment 전에 해소해야 한다.

## 6. Definition of Done 상태

| 항목 | 상태 | 증거/비고 |
|---|---|---|
| Dataset manifest builder | 구현 | 실제 3-chain full scan은 별도 장기 실행 필요 |
| Temporal leakage test | 완료 | 신규 테스트 통과 |
| Rolling temporal fold 5+ | 완료 | deterministic unit test |
| Stateful streaming | 완료(코어) | replay/recovery tests |
| Dual/risk router | 완료 | boundary tests |
| Incremental L2 state | 완료(primitive) | add/remove/future test; real embedding equivalence 미실행 |
| TTL/LRU cache, bounded queue | 완료 | pressure/expiration tests |
| Checkpoint/restart | 완료(코어) | stream/subgraph/queue state primitive; full GPU model equivalence 미실행 |
| MC metric 보강 | 완료 | variance/entropy/T=1/seed |
| Provenance/result schema | 완료 | smoke manifest |
| PyGOD/DLG direct baseline 재실행 | 미완료 | compute experiment 필요 |
| 5-seed MC sensitivity | 미완료 | compute experiment 필요 |
| Latency P50/P95/P99, cold/warm | 미완료 | real pipeline profiling 필요 |
| Memory slope/100k events | 미완료 | long replay 필요 |
| Calibration/reliability | 미완료 | prediction artifacts 필요 |
| Cross-chain/temporal robustness | 미완료 | training/evaluation 필요 |
| 95% CI/significance | 미완료 | multi-run predictions 필요 |
| Paper tables/figures | 미완료 | 임의 수치 생성 금지 원칙에 따라 보류 |

## 7. 재현 명령

Provenance 및 실제 dataset audit:

```bash
cd /mnt/d/_Work/goat_bank/dlg_gnn
PYTHONPATH=src /mnt/d/_Work/goat_bank/.venv/bin/python \
  -m gog_fraud.pipelines.run_sci_evaluation \
  --config configs/sci/experiments/main.yaml \
  --output-root results_sci
```

이 명령은 audit와 manifest를 생성한다. 실제 학습/평가 단계는 config의 `pipeline_commands`에 재현 가능한 명령을 명시하고 `--run-configured-stages`를 함께 사용해야 한다. 명시되지 않은 실험을 성공한 것으로 간주하지 않는다.

## 8. 다음 단계와 논문 주장 제한

P0의 다음 순서는 Level-1/Level-2 output contract 확정, 실제 transaction timestamp normalization adapter 연결, 동일 temporal split에서 direct baseline과 StreamMC 5-seed 실행, real sample trace 기반 latency/memory/calibration 산출이다. 그 전까지 허용되는 주장은 “stateful selective inference framework가 구현되고 unit-level 결정성/경계조건이 검증됐다”까지다.

다음 주장은 현재 증거로 금지한다.

- DLG-StreamMC가 baseline보다 detection 성능이 높다.
- 특정 MC sample 수가 최적이다.
- memory가 bounded 또는 200 MB 이하라고 실험적으로 검증됐다.
- analyst workload 또는 false negative가 특정 비율 감소한다.
- cross-chain generalization 또는 통계적 우월성이 입증됐다.

## 9. 최종 판정

`IMPLEMENTATION_FOUNDATION_VALID / EXPERIMENT_EVIDENCE_INCOMPLETE / SCI_NOT_READY`

코어 구현은 테스트 가능한 상태지만, 작업지시서 Definition of Done의 계산 실험과 paper package는 완료되지 않았다. 본 보고서는 미실행 항목을 성공으로 표시하거나 기존 결과 CSV를 재사용하지 않는다.
