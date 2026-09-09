# DLG-GNN Repository Multi-Project Architecture & Organization Guide

> **Document Version**: 1.0  
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

### 기존 문제점
과거에는 소스코드뿐만 아니라 실험 결과(`results/`), 보고서(`reports/`), 출력물(`outputs/`), 도표(`figures/`, `tables/`), 실험 스크립트(`experiments/`)가 루트 레벨에 혼재되어 있어, **특정 논문 집필 또는 리비전 시 어떤 데이터와 산출물을 참조해야 하는지 파악하기 어려운 문제**가 있었습니다.

### 해결 방안
공용 라이브러리(`src/`, `configs/`, `utils/`)는 공유 체계를 유지하면서, 산출물 5대 폴더(`figures/`, `outputs/`, `reports/`, `results/`, `experiments/`) 및 관련 문서 체계를 **4대 프로젝트 식별자(`dlg_gnn`, `stream_mc`, `graph_rag`, `benchmark`) 기반으로 모듈화·재배치**하였습니다. 기존 스크립트 및 테스트와의 하위 호환성을 위해 주요 경로에는 심볼릭 링크(Symlink)를 구축하였습니다.

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

```
dlg_gnn/
├── docs/
│   ├── architecture/                   # 시스템 및 리포지토리 아키텍처 문서
│   │   ├── 01_architecture.png
│   │   ├── 04_repository_project_structure.md   <-- [본 문서]
│   │   └── system_architecture.*.md
│   ├── papers/                         # 논문 원고 및 LaTeX 소스 (4개 프로젝트별 분리)
│   │   ├── _40_DLG_GNN/                # [dlg_gnn] DLG-GNN 6p 논문 (IEEE style)
│   │   ├── _41_01_Stream/              # [stream_mc] Stream MC Selective 논문
│   │   ├── _42_DLG_GNN_Benchmark/      # [benchmark] 5x5 Benchmark 논문
│   │   └── _43_01_TDS/                 # [graph_rag] Temporal Distribution Shift 논문 (IEEEtran)
│   └── work_reports/                   # 연구 작업지시서 및 리뷰 보고서 (4개 프로젝트별 분리)
│       ├── _2026/
│       │   ├── dlg_gnn/                # [dlg_gnn] Rounds 000 ~ 024
│       │   ├── stream_mc/              # [stream_mc] Rounds 100 ~ 109
│       │   ├── benchmark/              # [benchmark] Rounds 030 ~ 036, 200 ~ 207
│       │   └── graph_rag/              # [graph_rag] Rounds 040 ~ 045
│       ├── benchmark/                  # [benchmark] Rounds 208 ~ 211 (Defense Extensions D1~D4)
│       ├── graph_rag/                  # [graph_rag] Rounds 300 ~ 311 (Scam Revision & TDS Paper)
│       └── stream_mc/                  # [stream_mc] Rounds 107 ~ 115 (SCI Submissions R1~R4)
│
├── figures/                            # [NEW] 4대 프로젝트별 그림 및 시각화 자료
│   ├── dlg_gnn/                        # [dlg_gnn] 01_architecture.png, 02_dlg-gnn_results.png 등
│   ├── stream_mc/                      # [stream_mc] ablation_performance.png, calibration_plot.png 등
│   ├── graph_rag/                      # [graph_rag] scam_revision/*, main_final/*, main_final_v2/*
│   └── benchmark/                      # [benchmark] 벤치마크 및 국방 확장 시각화 플롯
│
├── outputs/                            # [NEW] 4대 프로젝트별 실험 중간 결과 및 실행 산출물
│   ├── dlg_gnn/                        # [dlg_gnn] ablation/, experiment_runs/ (EXP-001 ~ EXP-005)
│   ├── stream_mc/                      # [stream_mc] dlg_streammc_sci_evaluation/
│   ├── benchmark/                      # [benchmark] sci_round2_pilot ~ round5_final, sci_defense_extension*
│   └── graph_rag/                      # [graph_rag] GraphRAG / TDS 파이프라인 산출물
│
├── reports/                            # [NEW] 4대 프로젝트별 연구 검증 보고서
│   ├── dlg_gnn/                        # [dlg_gnn] ablation_study_report.md
│   ├── stream_mc/                      # [stream_mc] round_4/ (통합 검증 보고서 및 Evidence Index)
│   ├── graph_rag/                      # [graph_rag] round_1~4/, scam_revision*/, main_final/ (v7/v8 게이트)
│   └── benchmark/                      # [benchmark] 국방 확장 감사 및 데이터 적합성 보고서
│
├── results/                            # [NEW] 4대 프로젝트별 최종 정량 평가 수치 및 체크포인트
│   ├── dlg_gnn/                        # [dlg_gnn] multiseed/, latency/, leakage/, privacy_utility/, tables/
│   ├── stream_mc/                      # [stream_mc] results_sci_v2/, sci_v3_final/, sci_v3_submission_r1~r4/
│   ├── graph_rag/                      # [graph_rag] graphrag/, main_final/, main_final_v2/, paper_ready_gate_v8.json
│   └── benchmark/                      # [benchmark] 대규모 벤치마크 종합 결과
│
├── experiments/                        # [NEW] 4대 프로젝트별 실험 실행 스크립트 및 드라이버
│   ├── dlg_gnn/                        # [dlg_gnn] simulation/ (run_multiseed_simulation.py)
│   ├── stream_mc/                      # [stream_mc] run_all_paper_experiments.py, generate_figures.py 등
│   ├── graph_rag/                      # [graph_rag] round3/ ~ round7/, scam_revision/
│   └── benchmark/                      # [benchmark] 벤치마크 실행 스크립트
│
├── tables/                             # 4대 프로젝트별 LaTeX/CSV 표 산출물
│   ├── dlg_gnn/                        # [dlg_gnn] main_results.tex, context_baselines.tex 등
│   ├── stream_mc/                      # [stream_mc] stream_mc 관련 표
│   ├── graph_rag/                      # [graph_rag] scam_revision/, scam_revision_round2/
│   └── benchmark/                      # [benchmark] 벤치마크 비교표
│
├── src/                                # [공용] 공통 라이브러리 및 모델 코어 (수정 불필요)
│   ├── analysis/                       # 사후 분석 및 통계 유틸리티
│   ├── gog_fraud/                      # 다중 체인 Graph-of-Graphs 핵심 모델 및 어댑터
│   ├── models/                         # GNN 백본 (GIN, GraphSAGE, TGAT, TGN 등)
│   ├── profiling/                      # 하드웨어/레이턴시 측정 프로파일러
│   └── validation/                     # 누락/누출 방지 검증 모듈
└── tests/                              # [공용] 단위 테스트 및 재현성 검증 테스트 스위트
```

---

## 4. 논문 작성 시 프로젝트별 데이터 및 보고서 참조 가이드

### 4.1. Paper #40: DLG-GNN (`docs/papers/_40_DLG_GNN/`)
- **주요 실험 결과 수치**: `results/dlg_gnn/multiseed/multiseed_results.json`
- **절제 연구(Ablation) 결과**: `outputs/dlg_gnn/ablation/ablation_summary.md` 및 `results/dlg_gnn/ablation/`
- **정량 지표 표(LaTeX Tables)**: `tables/dlg_gnn/main_results.tex`, `results/dlg_gnn/tables/`
- **참조 도표**: `figures/dlg_gnn/01_architecture.png`, `02_dlg-gnn_results.png`, `03_memory.png`
- **작업 배경 보고서**: `docs/work_reports/_2026/dlg_gnn/000_*` ~ `024_*`

### 4.2. Paper #41: Stream MC (`docs/papers/_41_01_Stream/`)
- **최종 검증 완료 패키지**: `results/stream_mc/sci_v3_submission_r4/`
- **캘리브레이션 및 민감도 결과**: `results/stream_mc/results_sci_v2/` 및 `results/stream_mc/sci_v3_final/`
- **통합 검증 보고서**: `reports/stream_mc/round_4/DLG_StreamMC_SCI_Integrated_Verification_Report.md`
- **핵심 도표(Figures)**: `figures/stream_mc/` (`calibration_plot.png`, `mc_sensitivity_plot.png`, `leakage_utility_plot.png`)
- **작업 배경 보고서**: `docs/work_reports/stream_mc/107_*` ~ `115_*`

### 4.3. Paper #42: Benchmark (`docs/papers/_42_DLG_GNN_Benchmark/`)
- **Round 5 최종 벤치마크 결과**: `outputs/benchmark/sci_round5_final/raw/benchmark_raw.csv`
- **동결 데이터셋 매니페스트**: `outputs/benchmark/sci_round5_final/manifests/data_freeze.json`
- **국방 확장 결과(D1~D4)**: `outputs/benchmark/sci_defense_extension_real_final/`
- **관련 실행 스크립트**: `scripts/defense_extension/`, `scripts/defense_extension_real/`
- **작업 배경 보고서**: `docs/work_reports/benchmark/208_*` ~ `211_*`, `docs/work_reports/_2026/benchmark/200_*` ~ `207_*`

### 4.4. Paper #43: Temporal Distribution Shift (TDS / GraphRAG) (`docs/papers/_43_01_TDS/`)
- **Round 7 최종 모델 지표 요약**: `results/graph_rag/main_final_v2/comparable_model_metrics_per_seed.csv`
- **출판 준비 게이트 상태**: `results/graph_rag/main_final_v2/paper_ready_gate_v8.json` (Gate M = PASS)
- **부트스트랩/랜덤화 통계 검정**: `results/graph_rag/main_final_v2/statistical_comparisons.csv`
- **시간적 슬라이스 분석**: `results/graph_rag/main_final_v2/temporal_slice_metrics.csv`
- **핵심 논문 도표(Figures 1~4)**: `figures/graph_rag/main_final/` (`causal_pipeline.png`, `temporal_prevalence_shift.png`, `reliability_comparison.png`, `mc_tradeoff.png`)
- **실험 실행 코드**: `experiments/graph_rag/round7/` (`train_comparable_models.py`, `finalize.py`, `calibration.py`)
- **작업 배경 보고서**: `docs/work_reports/graph_rag/310_graphRAG_dataset_round_7/`, `311_tds_paper/`

---

## 5. 하위 호환성(Backward Compatibility) 및 유지보수 규칙

1. **루트 심볼릭 링크 유지**:
   - `results/main_final_v2` $\to$ `results/graph_rag/main_final_v2`
   - `results/results_sci_v2` $\to$ `results/stream_mc/results_sci_v2`
   - `outputs/sci_round5_final` $\to$ `outputs/benchmark/sci_round5_final`
   - `experiments/round7` $\to$ `experiments/graph_rag/round7`
   - `experiments/run_all_paper_experiments.py` $\to$ `experiments/stream_mc/run_all_paper_experiments.py`
   - 기존 파이프라인과 회귀 테스트(`pytest tests/round7`)가 경로 변경에 구애받지 않고 항상 $100\%$ 통과하도록 심볼릭 링크가 관리됩니다.

2. **향후 신규 작업 시 파일 저장 규칙**:
   - **새로운 실험 스크립트 작성 시**: `experiments/<project_name>/` 안에 배치합니다.
   - **새로운 평가 결과 생성 시**: `results/<project_name>/` 또는 `outputs/<project_name>/` 안에 배치합니다.
   - **새로운 분석 보고서 작성 시**: `reports/<project_name>/` 또는 `docs/work_reports/<project_name>/` 안에 배치합니다.
   - **새로운 도표 생성 시**: `figures/<project_name>/` 안에 배치합니다.
   - 루트 디렉토리에 프로젝트 특정 산출물을 임의로 누적하는 것을 금지합니다.
