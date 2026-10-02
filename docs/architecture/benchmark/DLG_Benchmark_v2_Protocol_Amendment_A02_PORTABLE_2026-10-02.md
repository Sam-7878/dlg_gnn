# DLG-GNN Benchmark v2 Protocol Amendment A02
## CUDA-first completion, preserved environments and portable evaluation contracts

**Amendment ID:** DLG-BENCH-V2-A02  
**Date:** 2026-10-02 (Asia/Seoul)  
**Historical parent:** `DLG_Benchmark_v2_Protocol_FROZEN_2026-09-30.md`  
**Operational predecessor:** `DLG_Benchmark_v2_Revised_Work_Order_A01_2026-10-01.md`  
**Effective protocol ID on adoption:** `DLG-BENCH-V2.0+A02`  
**Status:** 작성 완료, 연구책임자 채택 및 실제 qualification 대기.

> 원 FROZEN 문서의 날짜·내용·hash를 바꾸지 않는다. 이 amendment와 별도 EFFECTIVE A02 통합본을 추가한다. A02는 아직 수행하지 않은 시험을 PASS로 만들거나, 이미 본 결과를 소급하여 pre-registered라고 부르는 문서가 아니다.

## A. 변경 이유와 근거 경계

사용자가 현재 RTX 4070 Laptop 8 GB와 Gigabyte AORUS RTX 3090 Gaming Box eGPU 24 GB를 사용하고 있다고 명시했다. 원 FROZEN의 전용 64 GB RAM 서버·RTX 3060 Ti·RX 7900 XTX 기반 필수 CUDA/ROCm 비교는 현재 실행 계획과 맞지 않는다. A01 작업지시서의 CUDA-first 집중 범위를 공식 protocol override로 반영한다.

또한 Benchmark 이후 Stream/TDS, 이후 LLM 연구로 개발·평가 process를 이식하기 위해 공통 source/API/schema, backend별 environment lock, task별 scientific profile을 분리한다. 이것은 새 모델 제안이나 성능을 보고 유리한 조건을 선택하기 위한 변경이 아니다.

A01이 지칭하는 별도 `DLG_Benchmark_v2_Protocol_Amendment_A01_FOCUS_2026-10-01.md`는 제공되지 않았다. A02는 업로드된 A01 작업지시서에 실제 적힌 범위를 채택하고, 아래 override를 독립적으로 명시한다. 후일 A01 amendment 원본을 확보하면 문서 일치 여부를 감사하되 A02 시행 후 관측 결과에 맞추어 의미를 바꾸지 않는다.

첨부 ZIP은 논문 LaTeX/BIB/생성 표를 포함하지만 실제 repository, raw data, checkpoint, raw scores와 Grok 원문 review는 포함하지 않는다. 따라서 실제 PASS는 각 evidence 검증 이후에만 부여한다.

## B. 채택 전 필수 declaration

다음 예시를 `benchmark_v2/PROTOCOL/protocol_adoption_record.yaml`로 작성한다. `null`은 미확인이지 false가 아니다. 최종 production queue 생성 전에 연구책임자가 채워야 한다.

```yaml
amendment_id: DLG-BENCH-V2-A02
adopted_by: null
adopted_at_utc: null
prior_v2_performance_inspected: null
prior_v2_result_locations: []
prior_v2_run_ids: []
change_motivation: hardware_alignment_and_portable_infrastructure
metric_definition_audit_complete: false
legacy_archive_verified: false
cuda_qualification_verified: false
affected_cell_inventory_path: null
rerun_decision_approved: false
```

이미 v2 결과를 관측했다면 그 범위·날짜와 이번 변경 이유를 기록한다. 후속 run은 A02 rerun/correction으로 기술하고 최초 결과 전부터 A02가 동결되었다고 주장하지 않는다. 성능을 본 뒤 모델·dataset·threshold·통계 집합을 유리하게 선택하는 변경은 허용하지 않는다.

## C. 유지되는 Benchmark 과학적 계약

| 항목 | A02 유지 규칙 |
|---|---|
| Portfolio | 13 primary + LANL external, native Ethereum/BSC/Polygon 포함 |
| Labels | 실제 external fraud labels provenance 검증; synthetic7과 real-label6 분리 |
| Detector suite | DOMINANT, AnomalyDAE, CoLA, CONAD, GAD-NR, OCGNN, DLG-Base, DLG-Aug |
| Model seeds | 42,43,44,45,46 |
| Injection/sampling | primary injection42; 필요 시 sampling142–146 |
| Training | primary50 epochs; Aug20 local +50 global; 기존 upstream semantics |
| Primary execution | full declared input graph, exact original model semantics or fail closed |
| Prohibitions | 새 index cut, dimension/epoch 축소, objective 변경, silent CPU fallback, OOM 점수 대치 금지 |
| Selection | 원 transductive unsupervised fitting, validation-only threshold, test selection 금지 |
| Metrics | ROC-AUC, 기존 정의를 확인한 PR-AUC, validation-selected threshold의 test F1 및 secondary metrics |
| Statistics | dataset별 seed aggregate; S1 eight/S2 continuity-five/S3 real6/S4 synthetic7/S5 LANL 분리 |
| Resource guard | model × dataset × seed마다 기존24 GPU-hour operational guard |
| Integrity | CONAD audit, exact/fused equivalence, input/result hash, fresh process, raw-to-table 연결 |

`full graph`는 입력 graph를 새로 잘라 넣지 않는다는 뜻이다. 원 detector가 정의한 내부 contrastive sampling 등까지 임의로 다른 학습 알고리즘으로 변경하라는 의미는 아니다. 원 detector semantics를 명확히 기록한다.

## D. 원 FROZEN section별 override

| 원 section | 이전 | A02 effective rule |
|---|---|---|
| §1 RQ3/RQ4 | CUDA/ROCm·graph reduction을 전체 핵심 질문으로 설정 | 현재 RQ3는 qualified CUDA stack 및8/24 GB support; ROCm은 후속 별도 block. RQ4는 필요한 historical/index-order 진단과 trigger된 fidelity만 |
| §§10–11 | 두 backend qualification 필수 | CUDA target 유지, 공통 source/constraints + backend binaries 분리. 현재 CUDA만 production prerequisite |
| §13 | RTX3060Ti, RX7900XTX,64 GB server | 현재4070 Laptop8 GB +3090 eGPU24 GB; actual host specs probe. AMD exact GPU 미정 |
| §14 | 전용 서버로 laptop confound 제거 | notebook+eGPU confound를 기록·통제; 제거했다고 주장 금지 |
| §§18–19 | 모든8 GB OOM을 ROCm24로 rerun, 현재 portability 비교 | selected CUDA8/24 및 동일3090 cap 진단; ROCm comparison은 future qualified lane |
| §§20–24 | reduction/fidelity study 전체 | F0 primary 유지; F3 역사 진단, F2 필요 시, F1/HAEE는 explicit trigger·별도 protocol block |
| §31 | v2 dedicated server·두 backend description | v2 CUDA-first,3090 production,4070 selected diagnostics, 미래 ROCm deferred |
| §32 | CUDA8/ROCm24 중심 output tree | 저장소 공통 environment/core + task별 결과 directory |
| §33 | basic per-run manifest | study/task/unit/lane/backend/lock/ABI/checkpoint/calibration/attempt/metric definition 추가 |
| §§35–36 | AMD·양 backend Gate와 hardware를 fixed mandatory로 지정 | CUDA-first actual hardware Gate, future ROCm non-blocking |
| 추가 §38 | 없음 | task-specific inheritance, environment migration, job-level interoperability, publication release |

명시되지 않은 §§2–9,12,15–17,25–30,34,37의 기본 규칙은 유지한다. 위 override와 같은 topic에 속한 표현은 해당 section의 다른 문장에도 적용된다. EFFECTIVE A02 통합본은 이 대응표를 문장 수준으로 반영한 실행용 본문이며 새로운 과거 freeze 문서가 아니다.

## E. Environment·compiler 표준

1. 기존 `.venv`를 변경 전 inventory/archive/hash로 보존한 뒤 `.venv_old`로 rename한다. 새 환경을 먼저 기존 `.venv` 안에 설치하지 않는다.
2. `.venv_old`는 archive identity다. venv 절대 shebang/activation path 때문에 rename 후 실행 가능성을 가정하지 않는다. 원 위치 복원 또는 lock 기반 재생성으로 legacy를 재현한다.
3. 현재 CUDA 실행은 저장소 루트 `.venv_cuda`, 미래 ROCm 실행은 `.venv_rocm`만 사용한다. 다른 worktree의 동명 venv는 해당 worktree의 release/lock에 속한다.
4. Python3.12 exact patch, torch public version2.11.0, PyG2.8.0/PyGOD1.1.0, 공통 CPU dependencies/build tools는 가능한 한 맞춘다. 현재는 qualification candidate이며 완성된 binary lock이 아니다.
5. CUDA wheel `cu128`와 ROCm build는 다른 binary다. NVCC와 HIP/Clang, driver/runtime/BLAS/sparse/collective 및 GPU target은 backend별로 기록한다. compiler chain을 최대한 맞춘다는 말은 vendor compiler를 동일 binary로 강제한다는 뜻이 아니다.
6. ROCm Core SDK7.14.1과 torch2.11.0 공식 `rocm7.2` wheel 표기는 다른 version domain이다. 미래 GPU/OS 지원과 exact binary compatibility를 검증한 후 선택한다. `rocm7141` wheel index를 추정하여 만들지 않는다.
7. 공통 constraints와 backend·task별 전체 hash lock을 분리한다. PyG accelerator extension, Triton, compiled cache와 ABI는 backend별 검증·격리한다.
8. production lock 이후 변경은 새로운 environment stratum과 영향 범위를 만들고 해당 결과를 다시 계산한다. 더 큰 LLM을 위해 현재 논문 환경을 in-place upgrade하지 않는다.

외부 근거와 설치 명령은 A02 작업지시서 §4, §26의 [W1]–[W9]를 따른다.

## F. Hardware·memory 해석

H24-CUDA가 current primary environment이다. H8-CUDA는4070 physical8 GB diagnostic, H8cap-CUDA는 같은3090에 설정한 약8 GiB **PyTorch allocator cap**이다. cap fraction·목표 bytes·실제 physical bytes·allocated/reserved 및 가능한 외부 device memory를 구분한다. cap은 physical8 GB card와 같지 않으며 framework 외부 모든 VRAM allocation을 제한하는 보장도 아니다.

같은3090의 uncapped/capped 결과가 memory-headroom 비교의 우선 자료다. 서로 다른4070/3090 card의 결과는 architecture·power·eGPU link·host overhead도 다르므로 순수 memory 효과 또는 GPU 속도 우열로 해석하지 않는다. 8 GB 실패→24 GB 성공은 declared envelope/implementation 의존 실행성으로 기록한다.

선택 cell 목록과 선정 이유는 resource history와 scientific relevance로 predeclare한다. deterministic OOM이 명확하면 clean process에서1회 확인 후 남은 seed를 무의미하게 반복하지 않는다. seed-dependent memory consumption이면 cell 전체를 deterministic OOM으로 일반화하지 말고 run-level status를 보존한다. 지원 seed가 부족한 cell을 five-seed complete case로 넣지 않는다.

## G. A01 내용의 명시적 해석 정정

### G1. DLG-Base-70

A01 Phase F의 `hidden-size control` 표현은 첨부 Benchmark LaTeX의 `DLG-Base-70` 정의와 다르다. 첨부 원고는 **global training70 epochs의 epoch-budget control**로 기술한다. Base의 hidden dimension을70으로 바꾸는 지시가 아니다. A02는 원고 정의를 사용하고 이 정정을 change log에 기록한다. 원 실행 code의 정의도 확인하며, 이미 hidden=70으로 실행한 cell이 있다면 그 결과는 Base-70 evidence로 사용할 수 없다.

### G2. Metric 정의

첨부 문서의 PR-AUC 명칭만으로 average precision 또는 trapezoidal area를 선택하지 않는다. code audit로 기존 정의를 확인하고 versioned metric contract에 기록한다. threshold-selection으로 선택한 F1은 validation 점수 자체가 아니라 frozen threshold를 적용한 test F1이다. definition correction은 영향을 받는 모든 raw score에서 metric/table/statistics를 재생성한다.

### G3. 다른 DLG 연구의 모델을 대체하지 않음

Benchmark의 flat-graph GCN Base/Aug와 Stream의 GIN/GATv2 selective hierarchy 및 TDS의 causal GIN/temporal-style models는 다른 task/model identity이다. 공통 package refactor라는 이유로 서로 대체하지 않는다.

## H. Porting의 범위와 필수/후속 구분

**현재 필수:** common artifact/backend/runner/metric-definition interface, task profile isolation, temporal negative fixture, lock/probe/release. 기존 Benchmark correctness/13+1 evidence는 유지한다.

**Benchmark 이후:** Stream과 TDS의 actual model/data migration·재학습·논문 갱신. 각자의 seeds, labels, prediction units, chronological splits, calibration, MC mode, 통계 단위를 보존한다. 두 PDF의 total24,316이 같아도 split/positive count가 다르므로 raw lineage가 입증되기 전에는 동일 dataset으로 합치지 않는다.

**미래 선택:** AMD GPU qualification 및 CUDA/ROCm shared-support numerical study, independent worker job dispatch, LLM task adapter. mixed-vendor single-process training이나 unified VRAM은 제공 범위가 아니다. 같은 scientific config로 다른 backend에서 얻은 결과는 별도 environment evidence로 남긴다.

## I. Affected-cell 및 rerun 결정

| 변경 | 무효화/재실행 범위 |
|---|---|
| path rename만, 수치 경로 변화 없음 | legacy 원본 보존 및 환경 재실행 확인; 자동으로 모든 과거 training을 무효화하지 않음 |
| modern stack/code로 신규v2 production | 모든v2 production은 새 qualified lock/A02 manifest에서 수행; v1 숫자로 대체 금지 |
| CONAD branch/aggregation bug | 모든 affected CONAD cells; aggregation-only이면 검증된 raw outputs로 재생성 가능, score lineage 불명확하면 training rerun |
| shared exact/fused operator 변경 | 해당 operator를 사용하는 모든 model × dataset × seed × environment cells |
| feature/edge/label/split 오류 | 해당 artifact를 소비하는 모든 affected model/seed; calibration/threshold·통계도 재생성 |
| metric/table 계산 오류만 | trusted raw prediction으로 전체 affected metrics/statistics/표 재생성; training을 불필요하게 반복하지 않음 |
| CPU/GPU/driver/runtime 변경 | 새 environment identity; resource/time은 새로 측정, primary stratum에 혼합 금지 |
| Base-70을 hidden-dim70으로 잘못 실행 | 해당 control 전부 invalid; global epoch70의 원 정의로 재실행 |
| common wrapper가 semantics 변경 | affected task profile 영향 분석 후 해당 paths rerun |
| 문구/오탈자/문서 경로만 변경 | numerical rerun 없음; 문서 version/hash 갱신 |

실제 영향 범위는 `affected_cell_inventory.csv`에 run ID와 함께 기록한다. 불리한 결과만 옛 버전으로 남기거나 유리한 일부 cell만 새 버전으로 교체하지 않는다. qualification 실패를 workaround로 감춘 결과는 production에 들어갈 수 없다.

## J. Adoption acceptance 및 publication

A02 채택 기록, source/environment inventory, CUDA Gate, exact/CONAD Gate, actual3090 preflight, data/metric identity, support-aware tables가 완료되면 Benchmark core 실험을 종료한다. Preprints.org package를 먼저 만들고 등록 정보를 기록한 뒤 Applied Sciences Special Issue package를 준비한다. 선행 DLG-GNN DOI를 새 Benchmark DOI로 사용하지 않는다. 관련 원고와 shared data/code/evidence의 중복·차이를 공개한다.

원 FROZEN hash: `cc396d837ce7b8279abac1cd1b3691f90fde195a911047cad36914b1d939404c`  
A01 작업지시서 hash: `21814ec577d6b343ac3ab06e1b6c3788ec17024eada4596c356e19a65bae263a`

이 hash는 이번에 제공된 원본 bytes의 식별자다. 향후 사용자 repository에 있는 다른 version까지 동일하다고 가정하지 않는다.
