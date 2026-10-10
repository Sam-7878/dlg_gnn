# Pipeline and Data Handling Architecture

이 문서는 기초 이상 거래 탐지 데이터 세트부터 모델들의 단계적 평가, 훈련 및 융합에 이르는 End-to-end 실험 파이프라인의 과정을 구조화합니다.

## `run_fraud_benchmark.py` (핵심 실행 파이프라인)
- **위치:** `src/gog_fraud/pipelines/run_fraud_benchmark.py`
- 모델 파라미터 최적화, 불확실성/사기 징후(MC-nGNN 등급) 비교 평가를 모두 조율합니다.
  
### 평가 및 실행 단계 (Evaluation Phases)
1. **Legacy**
   과거의 벤치마크 모형(`DOMINANT`, `DONE`, `GAE`)을 어댑터 패키지를 통해 레거시 검증 절차에 맞춰 동작시킵니다.
2. **Revision L1**
   레거시에 대응할 최신 `Level1Model` (인터페이스: `Level1Output`을 명확히 반환)를 이용해 Subgraph 단위만의 Fraud Score 및 개별 AUC/AP(매크로 정확도)를 도출하여, 관계(Relation)가 포함되지 않은 구조의 한계 및 성능 기준선을 보여줍니다.
3. **Revision L1+L2 (단독 모델)**
   `Level1Trainer`가 산출한 고도화 임베딩(representation)을 추출 및 고정(Freeze)하여, Level 2 구조망(`Level2Trainer`)의 노드 입력 Feature로 주입해 시간적 흐름/사기 연관 전파(Graph Attention Network 의존 관계)를 실험합니다. 
4. **Revision Full (Fusion)**
   모든 Level 1 점수(Score 및 Embedding)와 Level 2에서 확산/보정된 Context Embedding 값들을 최종 결합(Fusion Network)하거나 관절 학습시켜 도출된 최고 수준의 탐지 능력을 정량 평가합니다.

## `Level1GraphDataset` / `Level2GraphDataset`
- 이질적인 (개별 트랜잭션, 노드 메타 통계 / 묶음 Subgraph 연관도) 두 개의 그래프 유형을 혼동 없이 다루기 위한 분리.
- **Level 1 Edge:** 한 지갑의 송수신 내역 집합 (Transaction Network). 송신자, 상호작용 지표 등이 명시됨.
- **Level 2 Edge:** 서로 상관없는 (혹은 연관 가능성이 존재하는) 시간순 서브그래프들의 Time Edge, Temporal Continuity Meta-Edge입니다.

## 설정 관리 (YAML Configs)
- `configs/benchmark/` 디렉토리에 명시된 설정 파일들을 기반으로 모델 하이퍼파라미터(`lr`, `hid_dim`, `dropout` 등)가 조절되며 `_build_level1_model`, `_build_level1_trainer` 등의 팩토리 함수가 객체들을 적응할 수 있게 준비합니다. (이때의 인자와 도메인 언어가 통일되도록 인터페이스를 통합.)

## Benchmark 데이터셋 전환 및 무결성 확보 (2026-10-09 Update)

### 1. 레거시 `*_hybrid_graph.pt` 폐기 및 제거
과거 GoG 프로젝트에서 생성되었던 `*_hybrid_graph.pt` (Ethereum, BSC, Polygon)는 다음과 같은 연구적 한계가 존재했습니다:
- **라벨 누수(Label Leakage) 위험:** k-NN($k=5$) 외에 추가된 엣지들이 라벨 기반(intra-class $k=3$)으로 샘플링되어 라벨 독립적 토폴로지 구성 원칙에 위배됨.
- **재현성 결여:** 과거 생성 스크립트 부재로 인한 외부 감사 지적(`DATASET_CONSTRUCTION_AUDIT.md`).

이에 따라 **과거의 `*_hybrid_graph.pt`, `*_knn_graph.pt`, `*_label_graph.pt` 아티팩트는 디스크에서 전면 영구 삭제**되었습니다.

### 2. `relation_builder.py` 기반 클린 Level 2 메타 그래프(`*_level2_graph.pt`) 전면 대체
`scripts/build_clean_level2_graphs.py`는 `relation_builder.py`의 `embedding_knn`, k=5, cosine 모드로 기존 tensor의 embedding에서 새 edge를 생성했습니다. 현재 활성 설정에는 temporal_window가 포함되지 않습니다.

1. **확인된 edge 속성:** embedding을 고정한 CPU 반사실 검사에서 label을 전부 0 또는 1로 바꿔도 edge·weight·x가 동일했습니다. 이 함수의 직접 label 의존성이 제거됐다는 범위의 검증입니다.
2. **Feature 출처:** 새 embedding은 이전 8차원 값과 동일하며, 양수인 column 3만 변합니다. 정규화 벡터가 모두 같아 cosine 유사도가 전부 1입니다. 원천 데이터에서 feature를 만드는 규칙과 tie 처리의 의미를 추가로 확정해야 합니다.
3. **Target 계약:** 기본 builder의 y는 graph-level scalar입니다. 새 artifact의 labels는 N개지만 A03 loader는 scalar y를 사용하는 Data를 그대로 반환합니다. 노드별 평가 adapter가 필요합니다. 생성기 재실행에서도 N개 labels를 명시적으로 선택해야 합니다.
4. **재현 범위:** 생성기는 기존 graph.pt에서 embedding을 가져옵니다. 원천 CSV/JSON만으로 재생성하는 경로와 반복 생성 계약은 아직 검증을 통과하지 않았습니다.
5. **논문 결과:** 보존된 A05/A07 수치와 canonical manifest는 이전 hybrid input에 연결됩니다. 새 graph에서 평가한 결과로 교체되었다고 해석할 수 없습니다. 새 version manifest, 재평가, 표·통계·원고·PDF 반영 후 과학 감사를 진행해야 합니다.

### 3. 독립 감사 상태

**2026-10-09: 최종 과학 감사 미통과.** 세 chain의 새 파일 존재와 옛 9개 파일 삭제는 확인됐습니다. 위 계약·feature·재현·결과 계보 문제가 남아 있습니다. 상세 근거는 `projects/benchmark/reports/astra_revision/RelationBuilder_Migration_Final_Audit.md`와 `projects/benchmark/evidence/astra_revision/relation_builder_migration_audit.json`에 있습니다. 기존 frozen 기록은 과거 실행의 증거로 유지합니다.
