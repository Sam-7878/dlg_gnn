# DLG-GNN Repository Multi-Project Architecture & Organization Guide

> **Document Version**: 1.1  
> **Date**: 2026-09-10  
> **Status**: Active / Production Standard  
> **Location**: `dlg_gnn/docs/architecture/04_repository_project_structure.md`

---

## 1. 개요 및 재배치 배경 (Executive Summary)

`dlg_gnn` 저장소는 금융 트랜잭션 이상탐지, 동적 로컬 그래프 신경망(GNN), 스트리밍 불확실성 보정, 그리고 그래프 RAG 연구를 포괄하는 대규모 연구 개발 리포지토리입니다. 현재 본 저장소는 다음 **4개의 핵심 연구 프로젝트(Paper Branches)**를 단일 코드베이스에서 공유하고 있습니다:

1. **`dlg_gnn`**: Dynamic Local Graph GNN 핵심 프레임워크 및 초기 절제 연구 (Paper #40: `_40_DLG_GNN`)
2. **`stream_mc`**: 스트리밍 환경 불확실성 정량화(MC Dropout), 캘리브레이션 및 선택적 리스크 라우팅 (Paper #41: `_41_01_Stream`)
3. **`benchmark`**: 5x5 다중 체인 벤치마크 및 DARPA Theia/LANL 국방 확장 연구 D1~D4 (Paper #42: `_42_DLG_GNN_Benchmark`)
4. **`graph_rag`**: GraphRAG 스캠 캠페인 분석 및 시간적 분포 변화(Temporal Distribution Shift, TDS) 인과 GNN 평가 (Paper #43: `_43_01_TDS`)

### 개편 원칙
공용 라이브러리(`src/`, `utils/`)는 공유 체계를 유지하고, 모든 설정 및 산출물/평가 폴더(**`configs/`, `figures/`, `outputs/`, `reports/`, `results/`, `experiments/`, `tests/`, `tables/`**)를 **4대 프로젝트 폴더(`dlg_gnn`, `stream_mc`, `benchmark`, `graph_rag`)만 존재하는 직관적이고 깔끔한 구조**로 통일하여 재배치하였습니다.

---

## 2. 4대 프로젝트 정의 및 연구 범위 (Project Taxonomy)

| 프로젝트 식별자 | 대응 논문 폴더 | 핵심 연구 주제 | 주요 데이터셋 및 특징 |
| :--- | :--- | :--- | :--- |
| **`dlg_gnn`** | `docs/papers/_40_DLG_GNN/` | Dynamic Local Graph GNN, 프라이버시 벡터, 불확실성 가중치 융합 (Base Architecture) | 초기 BSC, Ethereum, Polygon 로컬 트랜잭션 그래프 시뮬레이션 및 다중 시드(7, 17, 27, 37, 47) 기반 기본 절제 연구 |
| **`stream_mc`** | `docs/papers/_41_01_Stream/`<br>`manuscript/_41_01_DLG_StreamMC/` | 스트리밍 추론 하 불확실성 기반 선택적 연산(Selective Streaming), 캘리브레이션 캐스케이드, 리스크 제어(RCPS) | `GoG_sci_v2` / `smoke_dataset_v5`, 5개 시드 기반 MC Dropout 민감도($T \in \{1, 5, 10, 20, 30\}$), ECE/Brier/NLL 보정 평가 |
| **`benchmark`** | `docs/papers/_42_DLG_GNN_Benchmark/` | 대규모 다중 체인 5x5 벤치마크, PyGOD 통합, 비대칭 국방/침입탐지 데이터셋 확장(Defense Extension) | 12개 벤치마크 데이터셋(Ethereum, BSC, Polygon, Elliptic, DGraphFin, Reddit, DARPA Theia, LANL RedTeam 등), GAD-NR/DOMINANT/AnomalyDAE 비교 |
| **`graph_rag`** | `docs/papers/_43_01_TDS/` | GraphRAG 스캠 캠페인 탐지에서 엄격한 인과성 및 시간적 분포 변화(TDS) 스트리밍 GNN 검증 프레임워크로 발전 | `GoG-SCIMain-v1` (24,316 트랜잭션, 0 future leakage audited), 70/15/15 시간적 분할, 불균형 prior shift($39.85\% \to 2.93\%$), CausalLocalGIN vs TGAT/TGN |

---

## 3. 리포지토리 전체 디렉토리 매핑 조견표 (Directory Mapping Matrix)

모든 주요 폴더의 1단계 하위 경로는 4대 프로젝트 폴더로 동일하게 구성되어 있습니다:

```
dlg_gnn/
├── docs/
│   ├── architecture/                   # 시스템 및 리포지토리 아키텍처 문서
│   │   ├── 01_architecture.png
│   │   ├── 04_repository_project_structure.md   <-- [본 문서]
│   │   └── system_architecture.*.md
│   ├── papers/                         # 논문 원고 및 LaTeX 소스 (4개 프로젝트별)
│   │   ├── _40_DLG_GNN/                # [dlg_gnn] DLG-GNN 6p 논문 (IEEE style)
│   │   ├── _41_01_Stream/              # [stream_mc] Stream MC Selective 논문
│   │   ├── _42_DLG_GNN_Benchmark/      # [benchmark] 5x5 Benchmark 논문
│   │   └── _43_01_TDS/                 # [graph_rag] Temporal Distribution Shift 논문 (IEEEtran)
│   └── work_reports/                   # 연구 작업지시서 및 리뷰 보고서 (4개 프로젝트별)
│       ├── _2026/                      # 2026년 이전 아카이브
│       ├── dlg_gnn/                    # [dlg_gnn] Rounds 000 ~ 024
│       ├── stream_mc/                  # [stream_mc] Rounds 100 ~ 115
│       ├── benchmark/                  # [benchmark] Rounds 030 ~ 036, 200 ~ 211
│       └── graph_rag/                  # [graph_rag] Rounds 040 ~ 045, 300 ~ 311
│
├── figures/                            # [4대 프로젝트별 그림 및 시각화 자료]
│   ├── dlg_gnn/                        # [dlg_gnn] 01_architecture.png, 02_dlg-gnn_results.png 등
│   ├── stream_mc/                      # [stream_mc] ablation_performance.png, calibration_plot.png 등
│   ├── benchmark/                      # [benchmark] 벤치마크 및 국방 확장 플롯
│   └── graph_rag/                      # [graph_rag] scam_revision/*, main_final/* (Figures 1~4), main_final_v2/*
│
├── outputs/                            # [4대 프로젝트별 실험 중간 결과 및 실행 산출물]
│   ├── dlg_gnn/                        # [dlg_gnn] ablation/, experiment_runs/ (EXP-001 ~ EXP-005)
│   ├── stream_mc/                      # [stream_mc] dlg_streammc_sci_evaluation/
│   ├── benchmark/                      # [benchmark] sci_round2_pilot ~ round5_final, sci_defense_extension*
│   └── graph_rag/                      # [graph_rag] GraphRAG / TDS 파이프라인 산출물
│
├── reports/                            # [4대 프로젝트별 연구 검증 보고서]
│   ├── dlg_gnn/                        # [dlg_gnn] ablation_study_report.md
│   ├── stream_mc/                      # [stream_mc] round_4/ (통합 검증 보고서 및 Evidence Index)
│   ├── benchmark/                      # [benchmark] 국방 확장 감사 및 데이터 적합성 보고서
│   └── graph_rag/                      # [graph_rag] round_1~4/, scam_revision*/, main_final/ (v7/v8 게이트)
│
├── results/                            # [4대 프로젝트별 최종 정량 평가 수치 및 체크포인트]
│   ├── dlg_gnn/                        # [dlg_gnn] multiseed/, latency/, leakage/, privacy_utility/, tables/
│   ├── stream_mc/                      # [stream_mc] results_sci_v2/, sci_v3_final/, sci_v3_submission_r1~r4/
│   ├── benchmark/                      # [benchmark] 대규모 벤치마크 종합 결과
│   └── graph_rag/                      # [graph_rag] graphrag/, main_final/, main_final_v2/, paper_ready_gate_v6~v8.json
│
├── experiments/                        # [4대 프로젝트별 실험 실행 스크립트 및 드라이버]
│   ├── __init__.py                     # [공용] 하위 패키지 __path__ 자동 확장 (from experiments.round7... 지원)
│   ├── dlg_gnn/                        # [dlg_gnn] simulation/ (run_multiseed_simulation.py)
│   ├── stream_mc/                      # [stream_mc] run_all_paper_experiments.py, generate_figures.py 등
│   ├── benchmark/                      # [benchmark] 벤치마크 실행 스크립트
│   └── graph_rag/                      # [graph_rag] round3/ ~ round7/, scam_revision/, reports/
│
├── tests/                              # [4대 프로젝트별 단위 테스트 스위트]
│   ├── conftest.py                     # [공용] dlg_gnn 및 goat_bank 루트 sys.path 자동 주입
│   ├── dlg_gnn/                        # [dlg_gnn] unit/, mock/, llama/, data/, check_env/, test_ngnn.py 등
│   ├── stream_mc/                      # [stream_mc] sci_v3_final/, selection/, streaming/, submission*/
│   ├── benchmark/                      # [benchmark] sci_round1~5/, defense_extension*/, pygod_integration/
│   └── graph_rag/                      # [graph_rag] round4/, round6/, round7/, scam_revision*/, micro_rag/
│
├── tables/                             # [4대 프로젝트별 LaTeX/CSV 표 산출물]
│   ├── dlg_gnn/                        # [dlg_gnn] main_results.tex, context_baselines.tex 등
│   ├── stream_mc/                      # [stream_mc] 스트리밍 MC 표
│   ├── benchmark/                      # [benchmark] 벤치마크 비교표
│   └── graph_rag/                      # [graph_rag] scam_revision/, scam_revision_round2/
│
├── configs/                            # [4대 프로젝트별 실험 및 모델 설정 YAML/JSON]
│   ├── dlg_gnn/                        # [dlg_gnn] base.yaml, ablation.yaml, privacy.yaml, ngnn_mc/, mc/ 등
│   ├── stream_mc/                      # [stream_mc] sci/, sci_v2/, sci_v3_submission_r1~r4/ 등
│   ├── benchmark/                      # [benchmark] full_system.yaml, sci_round1~5, defense_extension 등
│   └── graph_rag/                      # [graph_rag] round4_sci_main_frozen.yaml 등
│
└── src/                                # [공용] 공통 라이브러리 및 모델 코어 (소스 코드 일원화)
    ├── analysis/                       # 사후 분석 및 통계 유틸리티
    ├── fusion/                         # 불확실성 가중치 융합 모듈 (Fixed, Learned, Uncertainty Fusion)
    ├── gog_fraud/                      # 다중 체인 Graph-of-Graphs 핵심 모델 및 어댑터
    ├── graphrag/                       # GraphRAG 핵심 알고리즘 (Local KB, Retriever, Risk Encoder/Head)
    ├── models/                         # GNN 백본 (GIN, GraphSAGE, TGAT, TGN 등)
    ├── privacy/                        # 프라이버시 벡터 코덱, 노이즈 주입 및 양자화
    ├── profiling/                      # 하드웨어/레이턴시 측정 프로파일러
    └── validation/                     # 누락/누출 방지 검증 모듈
```

---

## 4. 논문 작성 시 프로젝트별 데이터 및 보고서 참조 가이드

### 4.1. Paper #40: DLG-GNN (`docs/papers/_40_DLG_GNN/`)
- **주요 실험 결과 수치**: `results/dlg_gnn/multiseed/multiseed_results.json`
- **절제 연구(Ablation) 결과**: `outputs/dlg_gnn/ablation/ablation_summary.md` 및 `results/dlg_gnn/ablation/`
- **정량 지표 표(LaTeX Tables)**: `tables/dlg_gnn/main_results.tex`, `results/dlg_gnn/tables/`
- **참조 도표**: `figures/dlg_gnn/01_architecture.png`, `02_dlg-gnn_results.png`, `03_memory.png`
- **관련 테스트 스위트**: `tests/dlg_gnn/`
- **작업 배경 보고서**: `docs/work_reports/_2026/dlg_gnn/000_*` ~ `024_*`

### 4.2. Paper #41: Stream MC (`docs/papers/_41_01_Stream/`)
- **최종 검증 완료 패키지**: `results/stream_mc/sci_v3_submission_r4/`
- **캘리브레이션 및 민감도 결과**: `results/stream_mc/results_sci_v2/` 및 `results/stream_mc/sci_v3_final/`
- **통합 검증 보고서**: `reports/stream_mc/round_4/DLG_StreamMC_SCI_Integrated_Verification_Report.md`
- **핵심 도표(Figures)**: `figures/stream_mc/` (`calibration_plot.png`, `mc_sensitivity_plot.png`, `leakage_utility_plot.png`)
- **관련 테스트 스위트**: `tests/stream_mc/`
- **작업 배경 보고서**: `docs/work_reports/stream_mc/107_*` ~ `115_*`

### 4.3. Paper #42: Benchmark (`docs/papers/_42_DLG_GNN_Benchmark/`)
- **Round 5 최종 벤치마크 결과**: `outputs/benchmark/sci_round5_final/raw/benchmark_raw.csv`
- **동결 데이터셋 매니페스트**: `outputs/benchmark/sci_round5_final/manifests/data_freeze.json`
- **국방 확장 결과(D1~D4)**: `outputs/benchmark/sci_defense_extension_real_final/`
- **관련 실행 스크립트**: `scripts/defense_extension/`, `scripts/defense_extension_real/`
- **관련 테스트 스위트**: `tests/benchmark/`
- **작업 배경 보고서**: `docs/work_reports/benchmark/208_*` ~ `211_*`, `docs/work_reports/_2026/benchmark/200_*` ~ `207_*`

### 4.4. Paper #43: Temporal Distribution Shift (TDS / GraphRAG) (`docs/papers/_43_01_TDS/`)
- **Round 7 최종 모델 지표 요약**: `results/graph_rag/main_final_v2/comparable_model_metrics_per_seed.csv`
- **출판 준비 게이트 상태**: `results/graph_rag/main_final_v2/paper_ready_gate_v8.json` (Gate M = PASS)
- **부트스트랩/랜덤화 통계 검정**: `results/graph_rag/main_final_v2/statistical_comparisons.csv`
- **시간적 슬라이스 분석**: `results/graph_rag/main_final_v2/temporal_slice_metrics.csv`
- **핵심 논문 도표(Figures 1~4)**: `figures/graph_rag/main_final/` (`causal_pipeline.png`, `temporal_prevalence_shift.png`, `reliability_comparison.png`, `mc_tradeoff.png`)
- **실험 실행 코드**: `experiments/graph_rag/round7/` (`train_comparable_models.py`, `finalize.py`, `calibration.py`)
- **관련 테스트 스위트**: `tests/graph_rag/` (`tests/graph_rag/round7/`, `tests/graph_rag/scam_revision/` 등)
- **작업 배경 보고서**: `docs/work_reports/graph_rag/310_graphRAG_dataset_round_7/`, `311_tds_paper/`

---

## 5. 파이썬 임포트 및 테스트 실행 가이드

1. **`experiments` 패키지 자동 검색 메커니즘**:
   - `experiments/__init__.py`에서 `graph_rag`, `stream_mc`, `dlg_gnn`, `benchmark` 디렉토리를 `__path__`에 자동 등록하므로, 코드 상에서 `from experiments.round7.policy import ...` 및 `from experiments.graph_rag.round7.policy import ...` 두 형식 모두 완전히 투명하게 작동합니다.
2. **`tests/conftest.py` 루트 설정**:
   - `pytest` 실행 시 별도의 `PYTHONPATH` 환경변수 설정 없이도 `pytest tests/graph_rag/round7` 또는 `pytest tests/` 명령어로 전체 테스트가 자동 검색 및 실행됩니다.
