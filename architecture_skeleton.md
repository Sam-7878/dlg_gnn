# DLG-GNN Project Architecture Skeleton

이 문서는 **DLG-GNN** (*Decoupled Local-to-Global Graph Neural Network*) 및 관련 4대 서브프로젝트의 최신 아키텍처 골격을 설명하는 기본 안내 문서입니다.

세부 모듈별 아키텍처와 다른 프로젝트에서의 재사용 방법은 **[`docs/architecture/`](docs/architecture/)** 디렉터리의 상세 기술 문서를 참조하십시오.

---

## 1. 핵심 아키텍처 원칙 (Modern Architecture Principles)

1. **국소-대역 분리 학습 (Decoupled Local-to-Global Learning):**
   - $k$-hop 에고넷 중심의 국소 이상 탐지(Level 1)와 그래프 전체의 구조적 관계 학습(Level 2)을 분리하고, 학습 가능한 게이팅 파라미터($\alpha$)를 통해 융합.
2. **희소 행렬 완전 역전파 (Exact Sparse Reconstruction Engine):**
   - 기존 모델(DOMINANT 등)의 $O(N^2)$ 밀집 인접 행렬 생성 한계를 극복.
   - Gram 행렬 닫힌 형식 계산(Closed-form Gram Reduction)을 적용하여 $O(N^2)$ 메모리 할당 없이 $O(|E|d + Nd^2)$ 연산량만으로 정확한 Frobenius 인접 행렬 복원 손실을 계산.
3. **4대 서브프로젝트 대칭 구조화:**
   - 테스트(`tests/`), 리포트(`reports/`), 결과물(`results/`)이 `dlg_gnn`, `benchmark`, `stream_mc`, `tds`의 4대 서브프로젝트 단위로 일관되게 대칭 구성.
4. **엄격한 과학적 재현성 (Dual-Mode Reproduction):**
   - Mode 1: 동결된 원천 데이터로부터 0.4초 만에 논문 전체 표와 통계치를 복원하는 즉시 재현 파이프라인.
   - Mode 2: 10개 벤치마크 및 8개 탐지기 모델의 전체 재학습 파이프라인.

---

## 2. 주요 패키지 트리 및 모듈 구조

```text
dlg_gnn/
 ├── src/
 │    ├── gog_fraud/           # 핵심 DLG-GNN 모델 및 스트리밍 AML 라이브러리
 │    │    ├── models/         # DLG, DLG-Base, exact_reconstruction, Level 1/2
 │    │    ├── streaming/      # Bounded-state 스트리밍 엔진, 서브그래프 저장소, 캐시
 │    │    ├── selection/      # AML 트리아지 라우터
 │    │    ├── pipelines/      # 벤치마크, 튜닝, 스트리밍 리플레이 파이프라인
 │    │    └── data/           # 데이터 어댑터, 분할기, 합성 이상치 주입(-Syn)
 │    └── analysis/            # Friedman 순위 검정, Wilcoxon-Holm 사후 검정, 호모필리 분석
 ├── configs/                  # 모델, 데이터셋, 하이퍼파라미터 선언적 YAML
 ├── experiments/              # 벤치마크 실행 엔트리포인트 (run_sci_round5_final.py 등)
 ├── scripts/                  # Mode 1 재현 스크립트, 환경 검증 도구
 ├── tests/                    # 4대 서브프로젝트 대칭 테스트 스위트 (dlg_gnn, benchmark, stream_mc, tds)
 ├── reports/                  # 감사 및 실행 리포트 (dlg_gnn, benchmark, stream_mc, tds)
 ├── results/                  # 실험 결과 수치 (dlg_gnn, benchmark, stream_mc, tds)
 └── docs/architecture/        # 상세 아키텍처 및 타 프로젝트 재사용 가이드
```

---

## 3. 상세 아키텍처 문서 내비게이션 (`docs/architecture/`)

| 문서 | 핵심 주제 |
|---|---|
| [**README.md**](docs/architecture/README.md) | 전체 아키텍처 개요 및 내비게이션 맵 |
| [**01_system_overview.md**](docs/architecture/01_system_overview.md) | 상위 시스템 아키텍처 및 4대 서브프로젝트 역할 정의 |
| [**02_directory_and_source_structure.md**](docs/architecture/02_directory_and_source_structure.md) | 상세 소스 트리 가이드, 계층별 의존성 및 패키지 책임 |
| [**03_dlg_gnn_model_architecture.md**](docs/architecture/03_dlg_gnn_model_architecture.md) | 2단계 분리 학습 수학 공식 및 **Exact Sparse Reconstruction** 증명 |
| [**04_benchmark_and_evaluation_engine.md**](docs/architecture/04_benchmark_and_evaluation_engine.md) | 10개 데이터셋 포트폴리오, Mode 1/2 재현 및 Friedman/Wilcoxon 검정 엔진 |
| [**05_streaming_mc_engine.md**](docs/architecture/05_streaming_mc_engine.md) | 실시간 AML 스트리밍 엔진, Bounded-State 슬라이딩 윈도우 및 라우터 |
| [**06_tds_transaction_decomposition.md**](docs/architecture/06_tds_transaction_decomposition.md) | 트랜잭션 분해 시스템, 이분 그래프 변환 및 불확실성 인지 Micro-RAG |
| [**07_cross_project_integration_guide.md**](docs/architecture/07_cross_project_integration_guide.md) | **타 프로젝트에서 DLG 컴포넌트 재사용 가이드 (코드 예제 수록)** |

---

## 4. 과거 히스토리 참조 (Historical Documents)
프로젝트 초기 단계(Level 1/2 분리 초기 기획, nGNN 사전연산 등)의 설계 문서는 다음 폴더에 보존되어 있습니다:
- `docs/architecture/level1_and_2/`: 초기 Level 1/2 분리 기획안
- `docs/architecture/mc/`: 초기 몬테카를로 전략 개요
- `docs/architecture/ngnn/`: 초기 이웃 GNN 파이프라인
- `docs/architecture/ngnn_precompute/`: 초기 사전연산 데이터플로우
