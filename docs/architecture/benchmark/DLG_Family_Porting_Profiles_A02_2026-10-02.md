# DLG 연구군 공통 Process 및 논문별 Porting Profiles — A02

**Date:** 2026-10-02  
**Role:** Benchmark 작업지시서의 이식 부록. 각 논문의 기존 scientific contract를 보존하면서 공통 runtime/artifact/검증 계층을 재사용하기 위한 개발 명세.  
**Scope:** 현재는 adapter/profile/fixture 준비, actual Stream/TDS 재실험은 Benchmark release 이후. LLM/ROCm은 미래 확장.

> **공통 stack ≠ 공통 실험 protocol.** task 간 package/source/API는 공유하지만 prediction unit, label provenance, model architecture, split, seed, threshold, temporal state와 statistical unit은 독립적이다.

## 1. 공통화와 비공통화

| 계층 | 공유할 항목 | task 안에 남길 항목 |
|---|---|---|
| Environment | Python/공통 CPU dependencies exact pins, backend probe, 재설치 방식 | CUDA/ROCm binary lock, task 추가 dependency |
| Data provenance | hash 함수, manifest schema, ordered ID 검증 | graph builder, label 생성, split, cutoff 의미 |
| Execution | fresh process, device assignment, attempts, atomic output | fit/predict/update 순서, recurrent state, routing |
| Metrics | validated 함수·serialization·undefined 처리 mechanism | AP 정의, threshold source, ECE bins, population, score 종류 |
| Statistics | paired-index 검증, RNG 기록, report generator | independent unit, resampling unit, family, effect/claim boundary |
| Publication | raw→registry→표/figure/숫자 macro | 논문별 primary/secondary evidence 및 주장 |

공통 core가 GNN full graph만 받도록 설계하지 않는다. `input_artifact_manifest`가 node graph, contract snapshot, ordered event stream, 미래 tokenized corpus를 각각 가리키게 한다. task adapter는 해당 unit/semantics를 검증하고 구체 연산을 실행한다.

## 2. Profile BENCHMARK — current priority

**Source:** A01/FROZEN 및 `_42_01_Benchmark_PrePrints.zip` 내 `DLG-Benchmark_Preprint.tex`.

```yaml
study_id: dlg_benchmark_v2
prediction_unit: node
learning_mode: transductive_unsupervised
model_seeds: [42, 43, 44, 45, 46]
synthetic_injection_seed: 42
sampling_diagnostic_seeds: [142, 143, 144, 145, 146]
primary_environment: H24-CUDA
primary_graph_policy: full_declared_graph_original_semantics
primary_detector_count: 8
primary_dataset_count: 13
external_dataset: LANL-RedTeam
primary_epochs: 50
augmentation_local_epochs: 20
augmentation_global_epochs: 50
threshold_source: validation_only
```

이는 profile의 핵심 값만 나타낸 부분 명세다. dataset path/hash·모델별 frozen hyperparameters·metric definition은 실제 code/artifact 확인 후 완성한다. 예시 YAML을 production-ready config로 실행하지 않는다.

### B-P1. 모델 identity

첨부 원고의 DLG-Base/Aug는 public flat graph에서 사용하는 GCN 기반 변형이며 선행 DLG-GNN의 전체 hierarchical GATv2 architecture를 그대로 재현한 실험이 아니다. Base는 local/global GCN과 gate·reconstruction 경로, Aug는 local pretraining 이후 frozen embedding을 원 feature에 concatenate하는 경로를 보존한다. Aug reconstruction의 attribute target과 score가 원래 X를 기준으로 하는지 code에서 확인한다. Stream의 calibrated score fusion으로 바꾸지 않는다.

`DLG-Base-70`은 **70 global epochs** control이다. width70 model로 해석하지 않는다. Permuted/Zero control의 generation seed·shape·trainable path·학습 budget을 원 code와 대조한다. epoch count가 같아도 actual FLOPs/runtime가 동일하다고 주장하지 않는다.

### B-P2. Exactness와 operator 경계

선형 inner-product structural decoder의 Gram identity와 sigmoid decoder의 row/chunk exact 계산을 구분한다. 한 모델에 맞는 algebra를 다른 decoder에 적용하지 않는다. CONAD의 augmented graph에는 그 graph에 해당하는 normalized operator를 사용해야 하며 원 graph용 cache 재사용으로 augmentation을 무력화하지 않는지 검사한다. fixed weights/input에서 forward, loss, score, gradient와 가능하면 단일 optimizer step을 비교한다.

### B-P3. 통계·범위

dataset별5-seed aggregation 뒤 complete-case Friedman/Wilcoxon-Holm과 continuity-five view를 수행한다. seed를 독립 dataset으로 세지 않는다. LANL은 외부 descriptive evidence다. native blockchain이 선행 연구와 lineage를 공유하면 unseen discovery라고 표현하지 않는다.

## 3. Profile STREAM — DLG-SelectiveStream

**Source:** `_41_01_Stream.pdf`, pp.5–6 Tables II–V, p.9 Table VI, pp.10–11 runtime/replay, pp.21–22 reproducibility.

### S-P1. 상속해야 할 계약

| 항목 | 원고 기준 상속 값/규칙 |
|---|---|
| Prediction | contract snapshot; supervised classification |
| Corpus support | total24,316; train17,020 / validation3,648 / test3,648 |
| Positive support | train4,567 / validation393 / test161 |
| Seeds | 11,22,33,44,55 |
| Local model | 2-layer GIN, hidden32, dropout0.2, 3 degree features, mean–max readout |
| Reported local training | epochs5, batch64; 실제 L2 schedule은 original config에서 확인 |
| Relational model | 2-layer GATv2, 4 heads, hidden32, dropout0.2, historical k=8 |
| Routing | validation-calibrated confidence margin, primary T=1 deterministic |
| Calibration/fusion | validation-only logistic map in log-odds space; calibrated log-odds fusion |
| State | 90-day window, local nodes/edges≤128/128, active-contract store≤5,000 |
| MC | ablation T∈{1,3,5,8}; primary에 강제 재도입 금지 |
| Statistics | paired contract predictions, stratified bootstrap2,000, McNemar/Holm, descriptive interpretation |

embedding cache byte/entry cap, queue bound, precise training/selection hyperparameters가 이 표에 없다는 이유로 임의 기본값을 만들지 않는다. repository의 frozen config와 원고를 대조한다. Stream을 Benchmark의50epochs/seeds42–46/unsupervised model로 통일하지 않는다.

### S-P2. Temporal 및 state audit

각 target에 `target_id`, `chain_id`, `target_cutoff`, `eligible_reference_count`, `selected_reference_ids`, `max_selected_reference_cutoff`, `source_chain_policy`를 기록한다. selected reference는 training split 소속이면서 `t_ref <= t_target`여야 한다. source-only cross-chain transfer에서는 source chain restriction을 추가한다. 다른 test target을 reference로 넣지 않는다.

split chronology 통과와 pooled reference cutoff 통과는 다른 검증이다. reference cutoff 통과만으로 training parameters·raw feature·label availability까지 prospective하게 검증되었다고 주장하지 않는다. feature construction 시점, transformation fitting population, 모델 학습 데이터의 시간 가용성도 scope statement에 기록한다. raw event 또는 feature 경로를 변경하면 별도 causal provenance audit를 수행한다.

정해진 local cap/TTL/LRU/cache/queue 동작은 Stream의 선언된 semantics다. Benchmark full-graph 규칙을 적용해 이를 제거하지 않는다. 반대로 Benchmark의 OOM을 해결하기 위해 Stream cap을 가져다 쓰지 않는다. logical bounds와 실제 process RSS/allocator growth를 별도 기록한다.

### S-P3. 세 evidence lane의 분리

| lane | unit/population | 필수 산출물 |
|---|---|---|
| offline_contract | held-out3,648 contracts | per-seed Local/Selective/Full predictions, routes, calibrated scores, label support |
| runtime_prefix | 동일500-event prefix, warmup25, policy별5 repeats | 실제 E2E mean/P95/P99/throughput, route execution, synchronized timing policy |
| integrated_replay | 100,000 retrospective events, 3,378 distinct contracts | state/queue/cache/memory timeline, loss/duplicate, checkpoint50,000와 restart trace |

세 policy는 같은 seed/checkpoint/reference/calibration lineage를 공유하되 policy_id를 구분한다. event에 상속된 contract label의 반복 관측을100,000개의 독립 accuracy sample로 취급하지 않는다. route-skip 비율을 runtime saving으로 대체하지 않는다. batch model forward만 측정해 event E2E라고 부르지 않는다.4070→3090 환경 변경 시 timing은 새 환경에서 재측정한다.

### S-P4. Porting Gate 및 산출물

`S0` source/config/checkpoint inventory → `S1` dataset/unit/split/label reconciliation → `S2` GIN/GATv2·calibration·router smoke → `S3` reference temporal audit → `S4` 동일 checkpoint의 Local/Selective/Full held-out outputs → `S5` warm prefix 및 bounded replay/restart → `S6` canonical table/macro regeneration 순으로 수행한다.

Polygon test353개에 positives0인 원 split은 편의를 위해 다시 나누지 않는다. class-dependent metrics는 정의 가능 여부와 reason을 명시한다. PR-AUC/F1/coverage/risk의 각 support 조건을 구분하고 zero division을 좋은 성능으로 오인하지 않게 한다.

Outputs: `contract_predictions`, `calibration_selection`, `reference_audit`, `risk_coverage`, `runtime_prefix`, `state_replay`, `restart_equivalence`, `claims_to_evidence`. zero future violations 또는 zero restart disagreement는 새 trace 검증 결과가 있을 때만 갱신한다.

## 4. Profile TDS — Validity-First Temporal Distribution Shift

**Source:** `_43_01_TDS.pdf`, p.3 Tables I–II, p.4 Table III/Algorithm1, pp.5–7 results/statistics/runtime.

### T-P1. 상속해야 할 계약

| 항목 | 원고 기준 상속 값/규칙 |
|---|---|
| Unit | 원고상 transaction event; raw provenance 대조 필요 |
| Corpus | GOG-SCIMAIN-V1, total24,316 |
| Split sizes | train17,021 / validation3,647 / test3,648 |
| Positive support | train6,783 / validation308 / test107 |
| Seeds | 7,17,27,37,47 |
| Graph | `t_edge <= t_target`, recent edges≤128 |
| Models | CausalLocalGIN / TGAT-style / TGN-style / FraudSAGE |
| Main hidden/dropout | 48 /0.3 |
| Optimization | batch64, Adam lr1e-3, weight_decay1e-4, max_epochs20, patience5 on validation AUC-PR |
| F1 | primary fixed threshold0.5 |
| Calibration | per-seed positive temperature, validation NLL only |
| MC | T=10 primary; sensitivity1,5,10,20,30 |
| ECE | 15 equal-width bins; adaptive ECE separately |
| Statistics | paired ordinary/stratified bootstrap and randomization10,000; same ordered event IDs |
| Slices | six chronological test slices of608 |
| Runtime | complete3,648-event panel, batch128 model inference; not event E2E latency |

### T-P2. 실제 이식 전 mandatory audit

**TDS-I: unit/label/split lineage.** §5의 Stream–TDS reconciliation을 먼저 수행한다. transaction-event label인지 contract label이 상속된 snapshot/event인지 source row mapping으로 증명한다. 원고 용어가 artifact와 맞지 않으면 원고/data documentation을 정정하고 영향 범위를 기록한다.

**TDS-II: input dimension.** p.3 본문은3 degree features에 chain one-hot covariate를 추가한다고 쓰지만 p.4 Table III는 input dimension3을 보고한다. covariate가 node input인지 graph-level head input인지 실제 tensor assembly를 확인한다. 임의로3 또는6으로 바꾸지 않는다. architecture manifest에는 node_dim/graph_covariate_dim/head_input_dim을 분리한다.

**TDS-III: temporal/state semantics.** edge cutoff뿐 아니라 stable event order와 같은 timestamp tie policy, target event의 관측 가능 attribute, prediction-before-update 또는 update-before-prediction, memory reset/replay 규칙을 명시한다. TGN-style memory에 미래/test label이 들어가지 않아야 한다. train→validation→test에서 어디까지 historical events를 무라벨 state update에 사용할 수 있는지 freeze한다. test replay는 training gradient를 갱신하지 않는다. batch 처리로 시간 순서가 바뀌지 않는지 검사한다.

**TDS-IV: real edge counts.** 원 Table II의 audited edge 수와 실제 graph edges를 대조한다. padding/duplicate/self-loop를 포함한 tensor length인지 실제 admissible edge 수인지 정의한다. 단순히 sample_count×128을 계산하여 zero-violation audit를 생성하지 않는다. timestamp 누락은 PASS가 아니다.

**TDS-V: baseline fidelity.** TGAT/TGN은 원고의 controlled '-style' 구현이다. exact upstream reproduction으로 이름을 올리지 않는다. 새 faithful baseline은 task-matched temporal input, 동일 supervision, 별도 protocol을 정의한다. 기존 comparison을 조용히 바꾸지 않는다.

### T-P3. Calibration·MC·ensemble 재사용

temperature는 validation logits만으로 fitting하고 positive constraint 및 fitting population hash를 저장한다. deterministic score와 positive-temperature scaling의 within-seed ranking invariance를 점검하되 probability saturation/ties로 수치상 ranking이 바뀔 수 있으므로 logits와 probabilities를 모두 보존한다. temperature-scaled ensemble의 ranking은 단일모델 monotonicity와 별개다.

MC inference는 Dropout만 활성화하고 BatchNorm running statistics는 evaluation 상태로 유지한다. original implementation이 달랐다면 correction으로 기록하고 affected outputs를 재생성한다. inference RNG와 training seed를 분리한다. T=1 deterministic control과 stochastic single draw를 혼동하지 않는다.

Deep Ensemble/MC Ensemble/TS Deep Ensemble은 다섯 checkpoint를 합친 **각각 하나의 ensemble prediction artifact**다. 이를 독립적인 다섯 ensemble 반복처럼 세지 않는다. 멤버 checkpoint ID, 확률/로짓 aggregation 순서, per-member temperature, MC draw count를 저장한다.

### T-P4. 통계 정확성과 의존성 Gate

paired test는 항상 동일 ordered event IDs·labels에 적용한다. 원고는10,000 randomization을 exact라고 표현한다. 실제로 전수 열거가 아니라 Monte Carlo random permutations를 사용했다면 exact enumeration p-value로 표현하지 않고 finite Monte Carlo estimate로 설명한다. resampling seed, statistic, two-sided rule, plus-one correction 등 실제 구현을 확인하고 정정 시 모든 해당 comparison을 재생성한다. 원 p=0.0001을 코드 없이 그대로 재현했다고 주장하지 않는다.

event가 같은 contract/시간 구간에서 반복되면 iid event bootstrap/randomization 가정은 별도 위협이다. 원 분석은 historical track으로 보존하면서 dependence를 감사한다. 필요하면 결과 관측 전에 고정한 contract-cluster 또는 time-block sensitivity를 추가하며 유리한 p-value만 선택하지 않는다. comparison family와 multiplicity policy를 명시한다. seed checkpoint와 event resampling을 한 층의 independent samples로 섞지 않는다.

### T-P5. Porting Gate 및 산출물

`T0` provenance/input-dimension audit → `T1` causal builder·recurrent-state negative tests → `T2` 네 model smoke와 validation checkpoint selection → `T3` frozen test logits/predictions → `T4` TS/MC/ensemble → `T5` paired stats/6 slices → `T6` correct timing scope와 paper export.

Outputs: `causal_graph_audit`, `event_memory_audit`, `checkpoint_registry`, `validation_temperatures`, `mc_predictions`, `ensemble_registry`, `paired_statistics`, `temporal_slices`, `panel_runtime`, `claims_to_evidence`. 원고 숫자는 새 결과가 검증된 후 갱신하며 기존 출력과 같은 이름으로 overwrite하지 않는다.

## 5. Stream–TDS dataset lineage reconciliation

현재 첨부물만으로 아래 차이의 원인을 확정할 수 없다. total N만 일치한다는 이유로 같은 raw rows·labels·split이라고 단정하면 안 된다.

| 항목 | Stream PDF | TDS PDF |
|---|---|---|
| 명시 unit | contract snapshot | transaction event |
| N total | 24,316 | 24,316 |
| Train/Val/Test | 17,020 /3,648 /3,648 | 17,021 /3,647 /3,648 |
| Positive Train/Val/Test | 4,567 /393 /161 | 6,783 /308 /107 |
| Split rule | per-chain chronological | reported chronological70/15/15 |
| Temporal focus | selected historical training references | event-local edges and temporal state |

`dataset_lineage_reconciliation.csv`에는 source artifact ID/hash, upstream row ID, chain/contract/transaction ID, label source/transform, filtering rule, snapshot cutoff, event time, split generator/seed/hash, duplicate policy, task-specific inclusion reason을 둔다. raw sources가 동일해도 task transform이 다르면 새 derived artifact ID를 발급한다.

shared numerical findings의 중복도 점검한다. Stream의 MC routing cost 결론과 TDS의 MC calibration/ranking trade-off는 다른 task·MC setting·population의 주장이다. 하나의 결론을 다른 논문에 그대로 복사하지 않는다.

## 6. LLM future profile — 지금 구현하지 않는 확장 계약

공통 환경/worker/artifact 구조를 재사용하되 `task_id: llm`을 reserved namespace로만 둔다. 현재 논문 제출을 위해 Transformers/PEFT/quantization/serving runtime을 설치하거나 평가할 필요는 없다.

미래에 기록할 항목은 model/tokenizer repository revision, weight hash/license, dataset/corpus split, prompt/template hash, tokenizer/preprocessing version, context length, precision/quantization, attention backend, training/inference config, generation seed/stopping rule, semantic quality metrics와 serving measurement scope이다. prefill/decode/TTFT/TPOT/throughput은 GNN의 runtime metric과 별개 registry에 둔다.

CUDA-only fused kernel/optimizer가 있으면 capability registry에서 explicit dependency로 노출하고 ROCm port는 unsupported 또는 검증된 semantic-equivalent path로만 처리한다. vendor-specific packages 차이는 allowlist로 관리한다. 같은모델을 여러GPU에 분산하는 기능은 task별 별도 qualification이 필요하다.

## 7. 논문별 종료와 release isolation

Benchmark release를 먼저 완성한다. Stream/TDS는 각자의 audit와 필요한 correction을 완료하면 추가 architecture/대규모 search 없이 paper update로 넘어간다. 새 LLM/ROCm 개발이 기존 manuscript의 immutable release를 바꾸면 안 된다.

각 release에는 task protocol, full environment lock, compiler/system probe, dataset reconstruction instructions, input/result hashes, source commit, evidence validation report, paper tables/macros, correction log를 담는다. unresolved evidence는 OPEN으로 남기고 publication claim에서 제외한다.
