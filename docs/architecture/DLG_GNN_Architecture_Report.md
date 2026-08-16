# DLG-GNN (Decoupled Layered Graph GNN) 아키텍처 리포트

> **목적**: SCI급 저널 투고를 위한 코드베이스 정밀화 및 AI Agent 인수인계용 기술 문서  
> **대상 독자**: 개발에 새롭게 참여하는 AI Agent 및 연구 협력자  
> **작성 기준 소스**: `/mnt/d/_Work/goat_bank/dlg_gnn` (WSL2 Ubuntu 24.04, Python 3.12.13)

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [핵심 설계 원칙](#2-핵심-설계-원칙)
3. [최상위 디렉토리 구조](#3-최상위-디렉토리-구조)
4. [핵심 패키지: src/gog_fraud/](#4-핵심-패키지-srcgog_fraud)
5. [2계층 감지 시스템 상세](#5-2계층-감지-시스템-상세)
6. [확장 모듈: nGNN 및 MC-Dropout](#6-확장-모듈-ngnn-및-mc-dropout)
7. [벤치마크 및 분석 파이프라인](#7-벤치마크-및-분석-파이프라인)
8. [설정 시스템 (YAML)](#8-설정-시스템-yaml)
9. [데이터 흐름 다이어그램](#9-데이터-흐름-다이어그램)
10. [현재 구현의 한계 및 개선 방향 (SCI 강화)](#10-현재-구현의-한계-및-개선-방향-sci-강화)
11. [의존성 및 환경](#11-의존성-및-환경)

---

## 1. 프로젝트 개요

**DLG-GNN** (Decoupled Layered Graph GNN)은 블록체인 스마트 컨트랙트의 **이상 거래(Fraud) 탐지**를 위해 설계된 2계층 계층적 그래프 신경망 프레임워크입니다.

### 핵심 아이디어
- **Level 1**: 개별 스마트 컨트랙트의 거래 내역을 서브그래프(Subgraph)로 구성하고 내부 구조적 특성을 학습
- **Level 2**: Level 1 임베딩으로 구축된 관계 메타-그래프(Relation Graph)에서 컨트랙트 간 상호작용을 학습
- **Fusion**: Level 1과 Level 2의 예측 점수를 결합하여 최종 사기 점수 생성

### 대상 데이터
- Blockchain 네트워크: **Polygon, Ethereum, BSC (Binance Smart Chain)**
- 입력 단위: 스마트 컨트랙트 주소별 거래 내역 → 거래 그래프
- 레이블: 정상(0) / 사기(1) 이진 분류

### 공개 벤치마크 통합
`scripts/benchmark_8x10_pipeline.py`를 통해 다음 공개 데이터셋 8종 × 10개 모델에 대한 통합 벤치마크를 수행합니다:
- **Citation Networks**: Cora, CiteSeer, Amazon Photo
- **Social Networks**: Reddit, Weibo
- **Financial**: Amazon-Fraud, YelpChi, T-Finance
- **Blockchain** (proprietary): GoatBank 자체 데이터

---

## 2. 핵심 설계 원칙

| 원칙 | 내용 |
|------|------|
| **계층적 분리 (Decoupled)** | Level 1 (서브그래프 내부) / Level 2 (관계 그래프) 를 물리적으로 독립된 모듈로 구현 |
| **인터페이스 표준화** | 모든 모델 출력은 `Level1Output` / `Level2Output` 데이터클래스로 통일 |
| **설정 기반 실험 (Config-Driven)** | YAML 설정으로 모델 하이퍼파라미터·데이터·평가 방식을 동적으로 제어 |
| **Config 별칭 처리** | `from_config()` 팩토리 메서드가 다양한 키 별칭(alias)을 흡수하여 이전 버전 호환성 유지 |
| **NaN-Safe 평가** | 평가 모듈 전체에서 NaN/inf 방어 로직 적용 |
| **Legacy 호환** | `adapters/legacy_adapter.py`를 통해 PyGOD 기반 baseline 모델(DOMINANT, DONE 등)을 동일 파이프라인에서 실행 |

---

## 3. 최상위 디렉토리 구조

```text
dlg_gnn/
├── configs/                    # YAML 설정 파일 모음
│   ├── benchmark/              # 벤치마크 실험 설정 (full_system, strict 등)
│   ├── experiments/            # 실험 변형 설정
│   ├── legacy/                 # Legacy 모델 하이퍼파라미터 (best_params 포함)
│   ├── mc/                     # MC-Dropout 관련 설정
│   ├── ngnn/                   # nGNN 확장 설정
│   ├── ngnn_mc/                # nGNN + MC 복합 설정
│   └── ngnn_mc_legacy/         # nGNN + MC + Legacy 복합 설정
│
├── data/                       # 원시 데이터 저장 위치 (gitignore)
│
├── docs/
│   ├── architecture/           # 내부 아키텍처 문서 및 다이어그램
│   └── work_reports/           # 실험 분석 보고서
│       └── 36-domain_wise_ranking/   # 공개 벤치마크 분석 결과
│
├── outputs/                    # 실험 결과 출력 (CSV, JSON 등)
├── results/                    # 벤치마크 결과 저장
│
├── scripts/                    # 최상위 실행 스크립트
│   ├── benchmark_8x10_pipeline.py   # 8 데이터셋 × 10 모델 벤치마크 (핵심)
│   ├── benchmark_8x10_results.py    # 결과 집계
│   └── generate_plots.py           # 시각화 생성
│
├── src/                        # 핵심 소스 코드
│   ├── gog_fraud/              # 메인 패키지 (아래 §4에서 상세 설명)
│   └── analysis/               # 벤치마크 후처리 분석
│       ├── add_benchmark_analysis.py   # 도메인별 순위, 통계 검정
│       ├── plot_benchmark_analysis.py  # 시각화
│       ├── compute_homophily.py        # 그래프 동질성 지표 계산
│       └── utils.py                    # MODEL_FAMILY_MAP, 도메인 매핑
│
├── tests/                      # 단위 테스트
├── utils/                      # 보조 유틸리티
├── architecture_skeleton.md    # 최상위 아키텍처 개요
└── README.md
```

---

## 4. 핵심 패키지: `src/gog_fraud/`

```text
src/gog_fraud/
├── common/          § 4.1  공용 타입 정의
├── data/            § 4.2  데이터 파이프라인
├── models/          § 4.3  모델 아키텍처
├── training/        § 4.4  학습 루프
├── evaluation/      § 4.5  평가 모듈
├── pipelines/       § 4.6  엔드-투-엔드 파이프라인
└── adapters/        § 4.7  Legacy 모델 어댑터
```

---

### 4.1 공용 타입 계층 (`common/`)

**파일**: `common/types.py`

모든 모델이 반환해야 하는 표준 출력 컨테이너를 정의합니다.

```python
@dataclass
class Level1Output:
    graph_id:  torch.Tensor          # 그래프 식별자
    embedding: torch.Tensor          # 그래프 임베딩 벡터
    logits:    torch.Tensor          # 원시 로짓
    score:     torch.Tensor          # sigmoid(logits) in [0, 1]
    label:     Optional[torch.Tensor]  # 정답 레이블
    aux:       Dict[str, Any]          # 보조 정보 딕셔너리

@dataclass
class Level1EmbeddingBundle:         # Level 1 → Level 2 전달용 번들
    graph_id, embedding, logits, score, label, metadata
```

`Level2Output`은 `models/level2/model.py`에 정의되며, Level 1 구조와 동일하지만 **node-level** 및 **graph-level** 출력을 모두 `aux` 딕셔너리에 포함합니다.

---

### 4.2 데이터 파이프라인 (`data/`)

```text
data/
├── io/              # 원시 데이터 로딩 (FraudDataset 퍼사드)
│   ├── dataset.py           ← FraudDataset (핵심 퍼사드)
│   ├── transaction_loader.py ← 거래 그래프 직렬화 로더
│   ├── label_loader.py       ← CSV 레이블 로더
│   ├── global_graph_loader.py← 전체 네트워크 그래프 로더
│   ├── cache_store.py        ← 청크 캐시 (chunk_*.pkl)
│   └── streaming_dataset.py  ← 스트리밍 재생용 데이터셋
│
├── level1/          # Level 1 전처리
│   ├── builder.py           ← PyG Data 변환 (L1 서브그래프 생성)
│   └── dataset.py           ← L1 학습용 PyG Dataset 래퍼
│
├── level2/          # Level 2 전처리
│   ├── relation_builder.py  ← Level 1 임베딩 → L2 관계 그래프 생성 (핵심)
│   └── dataset.py           ← L2 학습용 PyG Dataset 래퍼
│
├── preprocessing/   # 피처 정규화 및 그래프 전처리
│   ├── normalizer.py        ← 노드 피처 정규화
│   ├── graph_builder.py     ← 그래프 구조 생성 보조
│   └── ngnn/                ← nGNN 전처리 (루팅 서브그래프 추출)
│       ├── subgraph_extraction.py
│       ├── precompute_rooted_subgraphs.py
│       └── serialization.py
│
├── scripts/         # 데이터 변환 스크립트
└── transforms/      # PyG 데이터 변환
```

#### `FraudDataset` (io/dataset.py) — 핵심 퍼사드

`FraudDataset`은 세 개의 독립적인 데이터 소스를 통합합니다:

| 소스 | 로더 | 내용 |
|------|------|------|
| 거래 그래프 | `TransactionLoader` | 컨트랙트별 청크 피클(.pkl) 캐시 |
| 레이블 | `LabelLoader` | CSV (Chain, Contract, Category 컬럼) |
| 전체 그래프 | `GlobalGraphLoader` | 네트워크 전체 연결 구조 |

**분할 전략 (`_auto_split`)**:
- 정상/사기 클래스별로 **Stratified Split** 수행
- 기본: Train 70% / Valid 15% / Test 15%
- 시드: `split_seed=42` (재현성 보장)

#### `RelationBuilder` (level2/relation_builder.py) — L2 그래프 생성

Level 1 임베딩 번들에서 Level 2 관계 그래프를 동적으로 구성합니다:

| 엣지 생성 모드 | 설명 |
|--------------|------|
| `embedding_knn` | 임베딩 코사인 유사도 기반 KNN (기본값, k=5) |
| `temporal_window` | 타임스탬프 기반 시간 윈도우 내 연결 |
| `shared_entity` | 사전 구축된 개체 공유 인접 행렬 사용 |

---

### 4.3 모델 아키텍처 (`models/`)

```text
models/
├── level1/            § 5.1  Level 1 모델
│   ├── model.py         ← Level1Model (GIN 기반, 주요 구현체)
│   └── level1_gnn.py    ← Level1GNN (SAGE/GCN/GAT 선택 가능, 대안 구현체)
│
├── level2/            § 5.2  Level 2 모델
│   └── model.py         ← Level2Model (GATv2 기반)
│
├── fusion/            § 5.3  Fusion 모듈 (현재 비어있음 → pipelines/fusion.py로 이동됨)
│
└── extensions/        § 6    확장 모듈
    ├── ngnn/            ← Nested GNN (nGNN) 확장
    │   ├── level1_ngnn.py      ← Level1nGNN (드롭인 대체)
    │   ├── nested_encoder.py   ← StandardNestedEncoder (서브그래프 내 GNN)
    │   ├── readout.py          ← StandardNestedReadout
    │   └── interfaces.py       ← NestedEncoder ABC
    └── mc/              ← Monte Carlo Dropout 확장
        ├── mc_dropout.py       ← MCDropoutWrapper
        ├── config.py           ← MCConfig
        ├── interfaces.py       ← MCModel ABC
        └── utils.py
```

---

### 4.4 학습 루프 (`training/`)

```text
training/
└── loops/
    ├── level1.py   ← Level1Trainer (GIN/nGNN 학습)
    ├── level2.py   ← Level2Trainer (GATv2 학습)
    └── test_loader.py
```

**Level1Trainer** (`loops/level1.py`):
- `TransactionGraph` 또는 `torch_geometric.data.Data` 모두 처리 (`_to_pyg_data` 헬퍼)
- AMP (Automatic Mixed Precision) 지원 (`torch.amp.GradScaler`)
- 클래스 불균형 대응: `pos_weight` 기반 BCEWithLogitsLoss 가중치 조정

**Level2Trainer** (`loops/level2.py`):
- Level 1 임베딩 번들을 입력으로 받아 관계 그래프 재구성 후 학습
- 노드-레벨 손실 함수 (개별 L1 서브그래프 단위 예측)

---

### 4.5 평가 모듈 (`evaluation/`)

```text
evaluation/
├── benchmark.py       ← BenchmarkResult, BenchmarkTable, evaluate_benchmark()
├── fraud_metrics.py   ← 저수준 사기 탐지 지표 (NaN-safe)
├── mc_metrics.py      ← MC-Dropout 불확실성 지표
└── evaluator.py       ← 고수준 평가 래퍼
```

#### `BenchmarkResult` 데이터클래스 (`benchmark.py`)

각 모델 × 각 세팅의 평가 결과를 담는 표준 컨테이너:

| 필드 | 설명 |
|------|------|
| `roc_auc` | ROC-AUC (주요 지표) |
| `pr_auc` | PR-AUC (불균형 데이터셋 핵심 지표) |
| `best_f1` | PR 커브 스위프로 구한 최적 F1 |
| `best_threshold` | 최적 F1을 달성하는 임계값 |
| `f1_at_05` | 임계값 0.5에서의 F1 |
| `p_at_k`, `r_at_k` | Top-K Precision/Recall |
| `ci_roc_auc`, `ci_pr_auc` | Bootstrap 신뢰구간 |
| `num_samples`, `num_pos`, `num_neg` | 샘플 통계 |

#### `fraud_metrics.py` 핵심 함수

```python
binary_classification_metrics(y_true, y_score, threshold=0.5)  # 혼동행렬 기반 모든 지표
find_best_f1_threshold(y_true, y_score)                          # 최적 임계값 탐색
topk_metrics(y_true, y_score, k)                                 # Top-K Precision/Recall
```

---

### 4.6 파이프라인 스크립트 (`pipelines/`)

| 스크립트 | 역할 |
|----------|------|
| `run_fraud_benchmark.py` | **메인 벤치마크** (54 KB): Legacy + L1 + L1+L2 + Full 4가지 실험 실행 |
| `fusion.py` | **Fusion 전략** (32 KB): 4가지 스코어 결합 방법 구현 |
| `run_sci_evaluation.py` | SCI 논문용 엄격한 평가 프로토콜 실행 |
| `run_streaming_replay.py` | 스트리밍 시나리오 재현 |
| `run_ablation_study.py` | 구성요소별 기여도 분석 |
| `run_mc_benchmark.py` | MC-Dropout 불확실성 정량화 실험 |
| `run_tuning_workflow.py` | 하이퍼파라미터 탐색 |
| `train_level1.py` | L1 모델 단독 학습 |
| `train_level2.py` | L2 모델 단독 학습 |
| `export_level1_embeddings.py` | L1 임베딩 내보내기 |
| `run_baseline_benchmark.py` | Baseline 모델 단독 실행 |
| `search_legacy_params.py` | Legacy 모델 하이퍼파라미터 탐색 |

---

### 4.7 Legacy 어댑터 (`adapters/`)

**파일**: `adapters/legacy_adapter.py` (46 KB — 가장 큰 단일 파일)

[PyGOD](https://github.com/pygod-team/pygod) 라이브러리의 Graph Anomaly Detection 모델들을 동일한 벤치마크 파이프라인에 통합합니다.

#### 지원 모델 (PyGOD 기반)

| 모델 | 계열 | 방법론 |
|------|------|--------|
| **DOMINANT** | Reconstruction | GNN 오토인코더 (구조+특성 재구성 오차) |
| **CONAD** | Contrastive | 대조 학습 기반 이상 탐지 |
| **DONE** | Reconstruction | 이중 오토인코더 |
| **AnomalyDAE** | Reconstruction | Dual-Autoencoder |
| **CoLA** | Contrastive | 서브그래프 대조 학습 |
| **GAAN** | Generative | 적대적 생성 네트워크 |
| **GUIDE** | Distance | 구조 기반 안내 탐지 |
| **GAE** | Reconstruction | 그래프 오토인코더 |

#### 대형 그래프 처리 전략 (`LegacyAdapterConfig`)

```yaml
max_nodes: 4096          # 이 이상이면 분할 처리
large_graph_mode: "partition"  # "skip" | "partition"
partition_size: 4096
aggregation_method: "max"      # "max" | "topk_mean"
```

---

## 5. 2계층 감지 시스템 상세

### 5.1 Level 1: 서브그래프 내부 인코더

**주요 구현체**: `models/level1/model.py` → `Level1Model`

#### 아키텍처 구성

```
입력: TransactionGraph (단일 컨트랙트의 거래 서브그래프)
  │
[선택적] StructuralEncoder (그래프 통계적 특성)
  │
Encoder Backend (택1):
  ├── GNN 모드: Level1GNNEncoder (GIN 기반)
  │     └── GINConv × num_layers → GraphReadout (mean/max/meanmax)
  └── nGNN 모드: Level1nGNN (§6 참조)
  │
Level1FraudHead (MLP: Linear → ReLU → Dropout → Linear)
  │
출력: Level1Output (graph_id, embedding, logits, score, label)
```

#### `Level1GNNEncoder` 세부사항
- **GINConv** (Graph Isomorphism Network) 기반 다층 메시지 패싱
- 각 레이어: `MLPBlock(in_dim, hidden_dim, out_dim)` 내장
- 활성화: ReLU + Dropout

#### `GraphReadout` 모드

| 모드 | 출력 차원 | 설명 |
|------|----------|------|
| `mean` | `hidden_dim` | 전역 평균 풀링 |
| `max` | `hidden_dim` | 전역 최대 풀링 |
| `meanmax` | `hidden_dim × 2` | 평균+최대 연결 (기본값) |

#### `Level1ModelConfig` 기본값

```python
in_dim: int = 16            # 노드 피처 차원
hidden_dim: int = 128       # 인코더 은닉 차원
num_layers: int = 3         # GIN 레이어 수
dropout: float = 0.2        # 드롭아웃 비율
readout: str = "meanmax"    # 그래프 리드아웃 방식
struct_dim: int = 0         # 그래프 통계 피처 차원 (0=비활성)
encoder_backend: str = "gnn" # "gnn" | "ngnn"
```

**대안 구현체**: `models/level1/level1_gnn.py` → `Level1GNN`
- SAGE/GCN/GAT 컨볼루션 타입 선택 가능
- LazyLinear 지원 (입력 차원 자동 추론)

---

### 5.2 Level 2: 관계 메타-그래프 인코더

**구현체**: `models/level2/model.py` → `Level2Model`

#### 아키텍처 구성

```
입력: L2 관계 그래프
  (노드 = L1 서브그래프, 엣지 = KNN/시간/공유 엔티티 관계)
  │
LayerNorm + Clamp (입력 안정화)
  │
Level2GATEncoder (GATv2 기반)
  ├── GATv2Conv × num_layers (동적 어텐션)
  └── LayerNorm + ELU + Dropout
  │
  ├── 노드-레벨 헤드 (주요) ────────────────────────────
  │   Level2FraudHead (MLP) → node_logits → node_scores
  │
  └── 그래프-레벨 헤드 (보조) ─────────────────────────
      GraphReadout → Level2FraudHead → graph_logits → graph_score
  │
출력: Level2Output
  ├── score = node_scores   (PRIMARY: L1 서브그래프별 점수)
  ├── logits = node_logits
  └── aux["graph_score"]    (AUXILIARY: 그래프 전체 점수)
```

#### `Level2GATEncoder` 세부사항
- **GATv2Conv** (Dynamic Graph Attention v2) 다층 인코더
- GAT 대비 GATv2 선택 이유: 동적 어텐션 강도 (표현력 우수)
- 각 레이어 후 `LayerNorm` 적용 (학습 안정성)
- 활성화: ELU

#### `Level2ModelConfig` 기본값

```python
in_dim: int = 16       # L1 임베딩 + 1 (score 연결)
hidden_dim: int = 128
num_layers: int = 2
num_heads: int = 4     # GAT 멀티헤드 수
dropout: float = 0.2
edge_dim: int = 0      # 엣지 피처 차원 (0=없음)
readout: str = "meanmax"
```

**안정화 처리**:
- `input_norm = LayerNorm` (입력 정규화)
- `torch.clamp(-10, 10)` (입력 클리핑)
- `torch.nan_to_num` (NaN/inf 방어)

---

### 5.3 Fusion 계층

**구현체**: `pipelines/fusion.py`

Level 1과 Level 2 점수를 결합하는 4가지 전략:

#### 전략 1: `WeightedSumFusion`
```
최종 logit = w₁ × logit(L1_score) + w₂ × logit(L2_score)
기본 가중치: L1=0.4, L2=0.6
```

#### 전략 2: `CalibratedFusion`
- Temperature Scaling으로 L1/L2 점수를 보정 후 가중합
- 잘 보정된(calibrated) 확률 추정 지원

#### 전략 3: `LearnedFusion`
- MLP 기반 학습형 결합
- L1/L2 로짓 → 2층 MLP → 최종 로짓
- 훈련 데이터에서 최적 결합 가중치를 학습

#### 전략 4: `FusionEnsemble`
- 여러 Fusion 전략의 평균 또는 다수결 앙상블

```python
# Fusion 입출력 표준 컨테이너
FusionInput(level1_score, level2_score, level1_logits?, level2_logits?, label?, graph_id?)
FusionOutput(score, logits, level1_score, level2_score, label?, graph_id?, metadata)
```

---

## 6. 확장 모듈: nGNN 및 MC-Dropout

### 6.1 Nested GNN (nGNN) 확장

**위치**: `models/extensions/ngnn/`

표준 GNN이 단일 홉 이웃만 보는 것과 달리, **nGNN은 각 노드 주변의 루팅 서브그래프를 추출하여 2단계 인코딩**을 수행합니다.

```
원본 그래프
  │ [precompute_rooted_subgraphs.py]
각 노드 n에 대해 k-hop 루팅 서브그래프 추출
  │
StandardNestedEncoder (서브그래프별 GNN 인코딩)
  └── 서브그래프 풀링: "main_root" | "mean" | "max" | "sum"
  │
StandardNestedReadout (서브그래프 임베딩 → 그래프 임베딩)
  │
MLPHead → 분류
```

**`Level1nGNN`**: `Level1GNN`의 드롭인 대체체로, `encoder_backend="ngnn"` 설정 시 자동으로 선택됩니다.

**전처리**: `data/preprocessing/ngnn/precompute_rooted_subgraphs.py`
- 학습 전 루팅 서브그래프를 사전 계산하여 저장
- 학습 시 캐시에서 로드하여 속도 향상

### 6.2 Monte Carlo Dropout (MC-Dropout) 확장

**위치**: `models/extensions/mc/`

**불확실성 정량화**: 예측 시 드롭아웃을 활성 상태로 유지하고 T회 반복 추론하여 예측 분산을 계산합니다.

```python
# MCConfig
num_mc_samples: int = 30    # MC 샘플링 횟수
mc_dropout_p: float = 0.2   # 드롭아웃 확률
```

**출력 지표** (`mc_metrics.py`):
- `mean_score`: 평균 예측 점수
- `std_score`: 예측 표준편차 (불확실성 척도)
- `epistemic_uncertainty`: 인식론적 불확실성

---

## 7. 벤치마크 및 분석 파이프라인

### 7.1 공개 벤치마크 (`scripts/benchmark_8x10_pipeline.py`)

8개 공개 데이터셋 × 10개 모델에 대한 비교 실험 파이프라인:

**데이터셋 (8종)**:

| 도메인 | 데이터셋 |
|--------|----------|
| Citation | Cora, CiteSeer, Amazon Photo |
| Social | Reddit, Weibo |
| Financial | Amazon-Fraud, YelpChi, T-Finance |

**모델 (10종)**:

| 계열 | 모델 |
|------|------|
| Reconstruction | DOMINANT, DONE, AnomalyDAE, GAE |
| Contrastive | CONAD, CoLA |
| GNN (DLG) | DLG (본 연구 모델) |
| 기타 | GAAN, GUIDE, 추가 모델 |

### 7.2 분석 파이프라인 (`src/analysis/`)

```
add_benchmark_analysis.py
  ├── Step 7: 도메인별 모델 순위 (Citation/Social/Financial/Blockchain)
  ├── Step 7.5: ROC-AUC 상세 점수/순위 테이블
  └── Step 8: Friedman + Nemenyi 통계 검정 (Critical Distance)

plot_benchmark_analysis.py
  ├── Metric-wise 비교 차트
  └── Edge Homophily vs. F1/PR-AUC 산점도 (x축: [0.9, 1.01])

compute_homophily.py
  └── 각 데이터셋의 Edge Homophily / Anomaly Ratio / Avg Degree 계산

utils.py
  └── MODEL_FAMILY_MAP: 모델별 계열 분류 (Reconstruction/Contrastive/Decoupled 등)
```

**성능 평가 지표 체계**:
1. **ROC-AUC** (단독 순위) — 가장 일반적인 기준
2. **복합 가중 점수** = ROC 33% + PR-AUC 33% + Best-F1 33%
3. **통계적 유의성**: Friedman 검정 + Nemenyi post-hoc

---

## 8. 설정 시스템 (YAML)

### 설정 파일 계층

```
configs/benchmark/full_system.yaml  ← 4가지 실험 동시 실행 (핵심)
configs/benchmark/strict.yaml       ← 엄격한 검증 설정
configs/benchmark/bsc_full.yaml     ← BSC 체인 전용
configs/benchmark/ethereum_full.yaml ← Ethereum 체인 전용
configs/benchmark/polygon_full.yaml  ← Polygon 체인 전용
```

### `full_system.yaml` 주요 구조

```yaml
setting: "full_system"

dataset:
  transactions_root: "../_data/dataset/transactions"
  labels_path: "../_data/dataset/labels.csv"
  chain: "polygon"            # polygon | ethereum | bsc
  auto_split: true
  train_ratio: 0.70
  val_ratio: 0.15

run_legacy: true              # PyGOD baseline 모델 실행
run_revision_l1: true         # Level1 only
run_revision_l1_l2: true      # Level1 + Level2
run_revision_full: true       # Full (L1 + L2 + Fusion)

legacy:
  models: ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"]
  epoch: 50
  hid_dim: 16

level1:
  encoder_backend: "gnn"      # "gnn" | "ngnn"
  hidden_dim: 16
  num_layers: 3
  conv_type: sage             # sage | gcn | gat
  dropout: 0.1

level2:
  eval_chunk_size: 16
  train_chunk_size: 16

fusion:
  # 기본값(WeightedSumFusion) 사용
```

### Config 팩토리 패턴

모든 주요 클래스는 `from_config(cfg)` 팩토리 메서드를 제공하며, 다양한 키 별칭을 자동으로 정규화합니다:

```python
Level1ModelConfig.from_config(cfg)   # in_dim <- input_dim, num_node_features 등
Level2ModelConfig.from_config(cfg)   # hidden_dim <- hid_dim, embed_dim 등
FraudDataset.from_config(cfg)        # chain <- blockchain, network 등
Level1GNN.from_config(cfg)           # 중첩 cfg 구조도 자동 탐색
```

---

## 9. 데이터 흐름 다이어그램

### 전체 시스템 데이터 흐름

```
[원시 데이터]
  ├── transactions/*.pkl  (컨트랙트별 거래 그래프)
  ├── labels.csv          (이진 레이블)
  └── global_graph/       (전체 네트워크)
        │ FraudDataset.load()
        ▼
[FraudDataset]
  ├── train_graphs: List[TransactionGraph]
  ├── valid_graphs: List[TransactionGraph]
  └── test_graphs:  List[TransactionGraph]
        │ Level1Trainer (level1.py)
        ▼
[Level 1 학습/추론]
  Level1Model (GIN/nGNN)
  → Level1Output {embedding, score, logits}
        │ export_level1_embeddings.py
        ▼
[Level1EmbeddingBundle] (.pt 파일로 디스크 저장)
        │ RelationBuilder (relation_builder.py)
        ▼
[L2 관계 그래프]
  노드=L1 서브그래프, 엣지=KNN/시간/공유 엔티티 관계
        │ Level2Trainer (level2.py)
        ▼
[Level 2 학습/추론]
  Level2Model (GATv2)
  → Level2Output {node_scores, node_logits}
        │ FusionInput 생성
        ▼
[Fusion]
  FusionStrategy.fuse(FusionInput)
  → FusionOutput {score, logits}
        │ evaluate_benchmark()
        ▼
[BenchmarkResult]
  {roc_auc, pr_auc, best_f1, ...}
        │
        ▼
[BenchmarkTable] → CSV / JSON 저장
```

### Level 1 내부 데이터 흐름

```
TransactionGraph
  │ _to_pyg_data()
  ▼
PyG Data {x: [N, 16], edge_index: [2, E], y: [1]}
  │ PyGDataLoader (batch_size=32)
  ▼
PyG Batch {x: [N*32, 16], batch: [N*32], edge_index}
  │ Level1GNNEncoder (GIN × 3 layers)
  ▼
Node embeddings: [N*32, 128]
  │ GraphReadout (meanmax)
  ▼
Graph embeddings: [32, 256]   # 256 = 128*2 (mean + max)
  │ [Optional] StructuralEncoder 결합
  │ Level1FraudHead (256 → 128 → 1)
  ▼
Logits: [32, 1]
  │ sigmoid
  ▼
Score: [32, 1] in [0, 1]
```

---

## 10. 현재 구현의 한계 및 개선 방향 (SCI 강화)

SCI급 저널 심사에서 제기될 수 있는 주요 한계와 권장 개선 방향입니다.

### 10.1 모델 아키텍처 관련

| 한계 | 현황 | 권장 개선 |
|------|------|----------|
| **GIN vs SAGE 중복** | `Level1Model`(GIN)과 `Level1GNN`(SAGE/GCN/GAT) 두 구현체가 공존, 기준 불명확 | 단일 통합 인코더 인터페이스로 정리 |
| **Level 2 엣지 타입** | KNN/시간/공유 엔티티 모드가 있으나 실험에서 단일 모드만 사용 | 다중 관계 타입 동시 학습 (Heterogeneous GNN) |
| **Fusion 학습** | LearnedFusion이 구현되어 있으나 현재 주로 WeightedSum 사용 | LearnedFusion 파라미터 최적화 실험 필요 |
| **시간적 모델링** | 현재 temporal_window는 단순 슬라이딩 윈도우 | TGNN(Temporal GNN) 또는 TGN 적용 고려 |

### 10.2 학습 프로세스 관련

| 한계 | 현황 | 권장 개선 |
|------|------|----------|
| **클래스 불균형** | `pos_weight` 가중치만 적용 | Focal Loss 또는 오버샘플링 (GraphSMOTE) 추가 |
| **하이퍼파라미터 탐색** | 수동 그리드 서치 (`run_tuning_workflow.py`) | Optuna/Ray Tune 기반 체계적 탐색 |
| **학습 스케줄러** | 명시적 LR 스케줄러 미적용 (일부 파이프라인) | CosineAnnealingLR 또는 ReduceLROnPlateau 적용 |

### 10.3 실험 엄밀성 관련

| 한계 | 현황 | 권장 개선 |
|------|------|----------|
| **반복 실험** | 단일 시드(seed=42) 실험 | 5-fold CV 또는 10회 반복 실험으로 통계적 안정성 확보 |
| **데이터 누수** | 자동 분할 시 시간적 순서 미반영 | Temporal Split 적용 (과거→미래 방향) |
| **공정 비교** | Legacy 모델과 epoch 수 차이 가능성 | 공정 계산 예산 (FLOPs 또는 학습 시간) 통제 |

### 10.4 코드 아키텍처 관련

| 한계 | 현황 | 권장 개선 |
|------|------|----------|
| **훈련 루프 분산** | `run_fraud_benchmark.py`에 학습 로직 혼재 | Trainer/Evaluator 완전 분리, 모듈화 |
| **하드코딩 경로** | 일부 설정에서 절대 경로 하드코딩 | 환경 변수 또는 Hydra 기반 경로 관리 |
| **Level 2 의존성** | L2 학습은 L1 임베딩 파일이 먼저 필요 | 통합 파이프라인에서 순서 자동 관리 |

---

## 11. 의존성 및 환경

### 실행 환경
- **OS**: WSL2 Ubuntu 24.04 LTS
- **Python**: 3.12.13
- **가상환경**: `.venv` (프로젝트 루트 기준)

### 핵심 라이브러리

| 라이브러리 | 역할 |
|------------|------|
| `torch` | 딥러닝 백엔드 |
| `torch_geometric` | GNN 레이어 (GINConv, GATv2Conv, SAGEConv, GCNConv 등) |
| `pygod` | Graph Anomaly Detection baseline 모델 (DOMINANT, DONE 등) |
| `scikit-learn` | 평가 지표 (roc_auc_score, average_precision_score 등) |
| `scipy` | Friedman 검정, Nemenyi post-hoc |
| `numpy`, `pandas` | 데이터 처리 |
| `yaml` | 설정 파일 파싱 |
| `psutil` | 메모리/CPU 모니터링 |
| `matplotlib`, `seaborn` | 시각화 |

### 실행 예시

```bash
# WSL2 Ubuntu에서
cd /mnt/d/_Work/goat_bank/dlg_gnn
source .venv/bin/activate

# 전체 벤치마크 실행
python -m gog_fraud.pipelines.run_fraud_benchmark \
    --config configs/benchmark/full_system.yaml

# 공개 데이터셋 벤치마크
python scripts/benchmark_8x10_pipeline.py

# 결과 분석
python scripts/benchmark_8x10_results.py
python src/analysis/add_benchmark_analysis.py
python src/analysis/plot_benchmark_analysis.py
```

---

## 부록: 주요 클래스 관계도

```
run_fraud_benchmark.py
  ├── FraudDataset.from_config()
  │     ├── TransactionLoader → TransactionGraph[]
  │     ├── LabelLoader → {contract_id: label}
  │     └── GlobalGraphLoader → GlobalGraphData
  │
  ├── [run_legacy] LegacyBatchRunner
  │     └── PyGOD Detectors (DOMINANT, DONE, ...)
  │
  ├── [run_revision_l1] Level1Trainer
  │     └── Level1Model / Level1GNN
  │           ├── Level1GNNEncoder (GIN)
  │           │     └── GINConv × n
  │           ├── [optional] Level1nGNN (nGNN)
  │           │     ├── StandardNestedEncoder
  │           │     └── StandardNestedReadout
  │           └── Level1FraudHead
  │
  ├── [run_revision_l1_l2] Level2Trainer
  │     └── Level2Model
  │           ├── Level2GATEncoder (GATv2)
  │           ├── Level2FraudHead (node-level)
  │           └── Level2GraphReadout (graph-level)
  │
  └── [run_revision_full] Fusion
        ├── WeightedSumFusion
        ├── CalibratedFusion
        ├── LearnedFusion
        └── FusionEnsemble

evaluate_benchmark(y_true, y_score)
  → BenchmarkResult
      └── BenchmarkTable.add() → CSV/JSON
```

---

*리포트 생성일: 2026-07-22*  
*기준 소스 경로: `/mnt/d/_Work/goat_bank/dlg_gnn`*  
*Python 환경: WSL2 Ubuntu 24.04, Python 3.12.13, 가상환경 `.venv`*
