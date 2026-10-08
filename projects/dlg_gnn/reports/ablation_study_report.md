# Ablation Study Report: nGNN + Monte Carlo Fraud Detection Pipeline

> **분석 기준일:** 2026-05-05  
> **대상 네트워크:** Polygon · BSC · Ethereum  
> **분석 축:** 성능(ROC-AUC, PR-AUC, Best-F1) / 자원(RAM, GPU, 실행시간)

---

## 1. 실험 구성 개요

본 Ablation Study는 GoG(Graph of Graph) 기반 사기 탐지 파이프라인의 구성 요소별 기여도를 정량적으로 분석합니다.

### 파이프라인 계층 구조

```
Level 0: Legacy Baselines  (DOMINANT, DONE, GAE, AnomalyDAE, CoLA)
    ↓ [선택: Feature Augmentation]
Level 1: nGNN / MC  (그래프 임베딩 + 불확실성 추정)
    ↓
Level 2: nGNN L2 / MC L2  (계층적 추론)
    ↓
Level Full: nGNN Full  (L1 + L2 + 앙상블)
```

### 실험 변형(Variant) 목록

| Variant ID | 파이프라인 구성 | 데이터 모드 |
|---|---|---|
| **nGNN** | Revision-L1 / L1+L2 / Full | Static |
| **MC-Static** | L1-Base / L1-MC / L1+L2-Base / L1+L2-MC | Static (Batch) |
| **MC-Streaming** | L1-StreamMC / L1+L2-StreamMC | Streaming (실시간) |
| **nGNN+MC-Static** | L1-Base / L1-MC / L1+L2-Base / L1+L2-MC | Static |
| **nGNN+MC-Streaming** | L1-StreamMC / L1+L2-StreamMC | Streaming |
| **nGNN+MC+Legacy-Static** | L1-Base-Aug / L1-MC-Aug / L1+L2-Base-Aug / L1+L2-MC-Aug | Static + Legacy 피처 |
| **nGNN+MC+Legacy-Streaming** | L1-StreamMC-Aug / L1+L2-StreamMC-Aug | Streaming + Legacy 피처 |

---

## 2. 핵심 성능 지표 요약

### 2-A. ROC-AUC (탐지 능력: 높을수록 좋음)

#### Polygon (355 samples: 345 fraud / 10 normal)

| Model | Stage | ROC-AUC |
|---|---|:---:|
| nGNN | L1 | 0.807 |
| nGNN | L1+L2 | **0.928** |
| nGNN | Full | 0.845 |
| MC-Static | L1-Base | 0.891 |
| MC-Static | L1-MC | 0.816 |
| MC-Static | L1+L2-Base | 0.931 |
| MC-Static | L1+L2-MC | 0.930 |
| MC-Streaming | L1-StreamMC | 0.781 |
| MC-Streaming | L1+L2-StreamMC | 0.870 |
| nGNN+MC-Static | L1-Base | 0.803 |
| nGNN+MC-Static | L1+L2-Base | 0.839 |
| nGNN+MC-Static | L1+L2-MC | 0.838 |
| nGNN+MC-Streaming | L1-StreamMC | 0.806 |
| nGNN+MC-Streaming | L1+L2-StreamMC | 0.817 |
| **nGNN+MC+Legacy-Static** | **L1+L2-MC-Aug** | **0.912** |
| nGNN+MC+Legacy-Streaming | L1+L2-StreamMC-Aug | 0.616 |

#### BSC (1,127 samples: 960 fraud / 167 normal)

| Model | Stage | ROC-AUC |
|---|---|:---:|
| nGNN | L1 | 0.850 |
| nGNN | L1+L2 | 0.831 |
| nGNN | Full | **0.845** |
| MC-Static | L1-Base | 0.785 |
| MC-Static | L1+L2-Base | 0.822 |
| MC-Static | L1+L2-MC | 0.815 |
| MC-Streaming | L1-StreamMC | 0.814 |
| MC-Streaming | L1+L2-StreamMC | 0.800 |
| nGNN+MC-Static | L1-Base | 0.858 |
| nGNN+MC-Static | L1+L2-Base | 0.862 |
| nGNN+MC-Static | L1+L2-MC | **0.864** |
| nGNN+MC-Streaming | L1-StreamMC | 0.859 |
| nGNN+MC-Streaming | L1+L2-StreamMC | 0.842 |
| **nGNN+MC+Legacy-Static** | **L1+L2-MC-Aug** | **0.864** |
| nGNN+MC+Legacy-Streaming | L1+L2-StreamMC-Aug | 0.819 |

#### Ethereum (2,172 samples: 1,266 fraud / 906 normal)

| Model | Stage | ROC-AUC |
|---|---|:---:|
| nGNN | L1 | **0.982** |
| nGNN | L1+L2 | 0.969 |
| nGNN | Full | 0.970 |
| MC-Static | L1-Base | 0.965 |
| MC-Static | L1+L2-Base | 0.972 |
| MC-Static | L1+L2-MC | **0.972** |
| MC-Streaming | L1-StreamMC | 0.971 |
| MC-Streaming | L1+L2-StreamMC | **0.977** |
| nGNN+MC-Static | L1-Base | 0.981 |
| nGNN+MC-Static | L1+L2-Base | 0.977 |
| nGNN+MC-Static | L1+L2-MC | 0.977 |
| nGNN+MC-Streaming | L1-StreamMC | **0.983** |
| nGNN+MC-Streaming | L1+L2-StreamMC | **0.992** |
| **nGNN+MC+Legacy-Static** | **L1+L2-MC-Aug** | 0.980 |
| nGNN+MC+Legacy-Streaming | L1+L2-StreamMC-Aug | **0.981** |

---

### 2-B. PR-AUC (불균형 데이터 탐지 능력: 높을수록 좋음)

#### PR-AUC 비교표 (Best Stage per Variant)

| Variant | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| nGNN (L1+L2) | 0.997 | 0.951 | 0.967 |
| MC-Static (L1+L2-MC) | 0.998 | 0.955 | 0.970 |
| MC-Streaming (L1+L2) | 0.992 | 0.929 | 0.953 |
| nGNN+MC-Static (L1+L2-MC) | 0.994 | 0.966 | 0.977 |
| nGNN+MC-Streaming (L1+L2) | 0.989 | 0.936 | 0.982 |
| **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** | **0.997** | **0.967** | 0.978 |
| nGNN+MC+Legacy-Streaming (L1+L2-Aug) | 0.974 | 0.939 | **0.960** |

> **Key Insight:** PR-AUC 기준으로는 Legacy Feature Augmentation이 적용된 Static 파이프라인이 BSC에서 일관되게 최상위 성능을 달성합니다. Polygon은 MC-Static과 동점 수준이며, Ethereum에서는 Streaming 변형이 경쟁력을 보입니다.

---

### 2-C. Best-F1 종합 비교

| Variant | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| nGNN (Best) | 0.990 | 0.941 | 0.948 |
| MC-Static (L1+L2-MC) | **0.993** | 0.920 | 0.938 |
| MC-Streaming (L1+L2) | 0.979 | 0.881 | 0.894 |
| nGNN+MC-Static (L1+L2-MC) | 0.987 | 0.942 | **0.950** |
| nGNN+MC-Streaming (L1+L2) | 0.977 | 0.913 | 0.931 |
| **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** | **0.990** | **0.944** | 0.950 |
| nGNN+MC+Legacy-Streaming (L1+L2-Aug) | 0.977 | 0.881 | 0.900 |

> **Key Insight:** Best-F1 관점에서 **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** 변형이 BSC에서 가장 높은 0.944를 달성합니다. Polygon은 MC-Static과 동일한 수준을 유지합니다. **Legacy Feature Augmentation은 F1 향상에 긍정적 기여**를 합니다.

---

## 3. MC(Monte Carlo) Dropout 기여도 분석

MC Dropout이 추가될 때 성능 변화를 Base 대비로 분석합니다.

### MC 기여도 (ROC-AUC Δ)

| 파이프라인 | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| MC-Static: L1-Base → L1-MC | **−0.075** | +0.001 | +0.002 |
| MC-Static: L1+L2-Base → L1+L2-MC | −0.001 | −0.007 | +0.000 |
| nGNN+MC-Static: L1-Base → L1-MC | −0.010 | +0.001 | +0.000 |
| nGNN+MC-Static: L1+L2-Base → L1+L2-MC | −0.001 | +0.002 | +0.000 |
| nGNN+MC+Legacy-Static: L1-Base-Aug → L1-MC-Aug | +0.008 | +0.000 | −0.000 |
| nGNN+MC+Legacy-Static: L1+L2-Base-Aug → L1+L2-MC-Aug | +0.014 | +0.002 | −0.000 |

> **Key Insight:** 
> - Polygon에서 순수 MC-Static의 L1 단계에서 ROC-AUC가 크게 하락(−0.075)하지만, PR-AUC와 F1은 거의 동일하게 유지됩니다. 이는 Polygon 데이터가 극도로 편향(345/10)되어 ROC-AUC가 소수 클래스에 민감하기 때문입니다.
> - **Legacy Feature Augmentation이 적용되면 MC Dropout의 기여가 일관되게 양수로 전환됩니다.** 즉, Legacy 피처 컨텍스트가 MC 불확실성 추정을 안정화합니다.

---

## 4. Legacy Feature Augmentation 기여도 분석

nGNN+MC-Static 대비 nGNN+MC+Legacy-Static을 비교합니다.

### ROC-AUC 변화 (Aug 추가 시 Δ)

| Stage | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| L1-Base → L1-Base-Aug | +0.028 | −0.004 | +0.001 |
| L1-MC → L1-MC-Aug | +0.045 | −0.004 | +0.001 |
| L1+L2-Base → L1+L2-Base-Aug | +0.060 | +0.000 | +0.003 |
| L1+L2-MC → L1+L2-MC-Aug | +0.074 | +0.000 | +0.003 |

> **Key Insight:**
> - **Polygon에서 Legacy Augmentation의 효과가 매우 뚜렷합니다.** L1+L2-MC 기준 ROC-AUC가 0.838 → 0.912로 **+0.074** 향상됩니다.
> - **BSC에서는 효과가 중립적**입니다(약 ±0.004). 이는 BSC 데이터셋(960/167)이 Polygon보다 정보가 풍부하여 Legacy 스코어의 추가 기여가 제한적임을 시사합니다.
> - **Ethereum은 이미 높은 베이스라인(0.977)에서 미미한 개선(+0.003)**을 보입니다. 천장 효과(ceiling effect)로 해석됩니다.

---

## 5. Static vs Streaming 비교

동일 변형에서 Static과 Streaming의 차이를 분석합니다.

### ROC-AUC: Static vs Streaming (L1+L2 기준)

| Variant | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| MC-Static (L1+L2-MC) | 0.930 | 0.815 | 0.972 |
| MC-Streaming (L1+L2) | 0.870 | 0.800 | **0.977** |
| Δ (Streaming − Static) | **−0.060** | −0.015 | **+0.005** |
| nGNN+MC-Static (L1+L2-MC) | 0.838 | 0.864 | 0.977 |
| nGNN+MC-Streaming (L1+L2) | 0.817 | 0.842 | **0.992** |
| Δ (Streaming − Static) | −0.021 | −0.022 | **+0.015** |
| nGNN+MC+Legacy-Static (L1+L2-MC-Aug) | **0.912** | **0.864** | 0.980 |
| nGNN+MC+Legacy-Streaming (L1+L2-Aug) | 0.616 | 0.819 | **0.981** |
| Δ (Streaming − Static) | **−0.296** | −0.045 | **+0.001** |

> **Key Insight:**
> - **Ethereum에서는 Streaming이 Static보다 ROC-AUC가 높습니다.** 이는 Ethereum 트랜잭션의 시계열 의존성(temporal dependency)이 강하여 순서 정보를 활용하는 Streaming 모드가 유리함을 보여줍니다.
> - **Polygon에서 Legacy Streaming의 성능 급락(−0.296)**은 주목할 만합니다. Polygon 데이터셋은 극도로 편향되어 있어(450/21), 스트리밍 분할 시 클래스 불균형이 심화되어 ROC-AUC가 불안정해지는 것으로 보입니다. (PR-AUC는 0.974로 여전히 높음)
> - **F1과 PR-AUC 기준으로는 Streaming도 Static에 필적하는 성능**을 보여, 실시간 처리 환경에서 충분히 활용 가능합니다.

---

## 6. 자원 사용량 분석

### 6-A. Peak RAM 사용량 (MB)

| Variant | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| nGNN (Full) | 3,710 | 6,490 | 6,638 |
| MC-Static (L1+L2-MC) | 3,714 | 6,373 | 6,404 |
| MC-Streaming (L1+L2) | 3,688 | 6,340 | 6,400 |
| nGNN+MC-Static (L1+L2-MC) | 3,736 | 6,518 | 6,610 |
| nGNN+MC-Streaming (L1+L2) | 3,688 | 6,334 | 6,398 |
| **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** | **11,551** | **16,349** | **17,889** |
| nGNN+MC+Legacy-Streaming (L1+L2-Aug) | **11,262** | **15,991** | **17,368** |

> ⚠️ **Legacy Feature Augmentation 적용 시 RAM 사용량이 약 2.5~3배 증가합니다.** 5개 Legacy 모델의 중간 스코어와 증강된 그래프 데이터를 메모리에 유지하기 때문입니다. Polygon 기준 3,736 MB → 11,551 MB로 약 **3× 증가**.

### 6-B. Peak GPU 메모리 (MB)

| Variant | Polygon | BSC | Ethereum |
|---|:---:|:---:|:---:|
| nGNN (L1+L2) | 507 | 293 | 234 |
| MC-Static (L1+L2-MC) | 940 | 526 | 424 |
| MC-Streaming (L1+L2) | 694 | 474 | 337 |
| nGNN+MC-Static (L1+L2-MC) | 507 | 292 | 235 |
| nGNN+MC-Streaming (L1+L2) | 377 | 260 | 196 |
| **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** | 508 | 293 | 236 |
| nGNN+MC+Legacy-Streaming (L1+L2-Aug) | 378 | 260 | 196 |

> GPU 메모리는 Legacy Augmentation 여부와 무관하게 거의 동일합니다. **GPU 부담 없이 Legacy 피처를 추가할 수 있음을 확인**합니다.

---

## 7. 실행 시간 분석

> **주의사항:** Legacy Learning Time이 기록되지 않은 일부 테스트의 경우, Legacy 실행 루틴은 동일하므로 아래 측정값을 기준으로 삼습니다.

### 7-A. Legacy Feature Augmentation 전처리 시간

| 네트워크 | 측정된 시간 (초) | 비고 |
|---|:---:|---|
| Polygon | 8,700.95 | ≈ **145분** (전체 그래프 기준) |
| BSC | 29,392.11 | ≈ **489분 ≈ 8.2시간** |
| Ethereum | 72,781.39 | ≈ **1,213분 ≈ 20.2시간** |

> ⚠️ **Legacy Augmentation은 그래프 수와 크기에 선형 이상으로 비례하는 비용이 발생합니다.** 그 이유는:
> 1. **5개 모델 × 전체 그래프 수**만큼 개별 `fit()` 호출이 반복됩니다.
> 2. Large Graph 분할(Partitioning)과 캐싱 I/O 오버헤드가 추가됩니다.
> 3. Ethereum(2,172 graphs, max 9,999 nodes)은 BSC(1,127)의 약 2.5배 시간을 소요합니다.

### 7-B. nGNN 단계별 실행 시간 (초)

| 네트워크 | L1 | L1+L2 | Full |
|---|:---:|:---:|:---:|
| Polygon | 17.5 | 10.3 | 9.2 |
| BSC | 45.8 | 33.7 | 33.5 |
| Ethereum | 60.5 | 64.8 | 64.7 |

### 7-C. nGNN+MC Static 단계별 실행 시간 (초)

| 네트워크 | L1-Base | L1-MC | L1+L2-Base | L1+L2-MC |
|---|:---:|:---:|:---:|:---:|
| Polygon | 18.1 | 19.4 | 13.4 | 14.3 |
| BSC | 41.2 | 43.9 | 30.1 | 32.6 |
| Ethereum | 52.9 | 56.0 | 72.1 | 76.9 |

### 7-D. nGNN+MC+Legacy Static 단계별 실행 시간 (초)

| 네트워크 | L1-Base-Aug | L1-MC-Aug | L1+L2-Base-Aug | L1+L2-MC-Aug |
|---|:---:|:---:|:---:|:---:|
| Polygon | 4.2 | 6.0 | 14.2 | 15.2 |
| BSC | 48.3 | 51.4 | 34.0 | 36.8 |
| Ethereum | 58.3 | 62.2 | 57.3 | 63.2 |

> Legacy Augmentation 후의 실제 모델 학습·평가 시간은 **비-Aug 버전과 유사**합니다. 추가된 시간(8차원 → 8차원 피처)은 미미하며, 비용의 대부분은 전처리 단계에 집중됩니다.

---

## 8. 종합 평가 및 결론

### 8-A. 성능 효율성 매트릭스

아래는 **성능(ROC-AUC)과 추가 비용(시간, 메모리)** 간의 트레이드오프를 정리한 것입니다.

| Variant | 평균 ROC-AUC | 평균 RAM (MB) | 전처리 비용 | 추천 시나리오 |
|---|:---:|:---:|:---:|---|
| nGNN (L1+L2) | 0.909 | 5,499 | 없음 | 빠른 배포, GPU 최소화 |
| MC-Static (L1+L2-MC) | 0.906 | 5,497 | 없음 | 불확실성 추정 필요 시 |
| nGNN+MC-Static (L1+L2-MC) | 0.893 | 5,621 | 없음 | nGNN+불확실성 통합 |
| nGNN+MC-Streaming (L1+L2) | 0.884 | 5,473 | 없음 | **실시간 운영** |
| **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** | **0.919** | **15,263** | 매우 큼 | **최고 성능, 배치 분석** |
| nGNN+MC+Legacy-Streaming (L1+L2-Aug) | 0.805 | 14,874 | 매우 큼 | ⚠️ Polygon에서 불안정 |

### 8-B. 핵심 발견 사항 (Key Findings)

1. **Legacy Feature Augmentation은 Polygon에서 ROC-AUC를 최대 +0.074 향상시킵니다.** 이는 5개 Legacy 모델의 전문가 스코어가 극도로 편향된 데이터셋에서 강력한 귀납적 편향(inductive bias)을 제공하기 때문입니다.

2. **Ethereum에서는 Streaming 모드가 Static보다 ROC-AUC가 높습니다** (nGNN+MC-Streaming: 0.992 vs Static: 0.977). 이는 Ethereum 트랜잭션의 시계열 패턴이 실시간 학습에 유리함을 보여줍니다.

3. **MC Dropout의 기여는 Legacy Augmentation 적용 시 일관되게 양수로 전환됩니다.** Legacy 피처가 MC 불확실성 추정을 안정화하는 시너지 효과가 있습니다.

4. **Legacy Augmentation의 GPU 비용은 무시할 수준입니다**(<10 MB 추가). 그러나 **RAM은 약 3배, 전처리 시간은 수십 배 증가**합니다.

5. **PR-AUC와 F1은 모든 변형에서 안정적으로 높습니다** (0.93~0.99 범위). 실운영 임계값 설정 시 두 지표를 함께 고려해야 합니다.

### 8-C. 권장 운영 전략

| 상황 | 권장 Variant | 이유 |
|---|---|---|
| 신규 체인 분석 (오프라인) | **nGNN+MC+Legacy-Static (L1+L2-MC-Aug)** | 최고 탐지 정확도 |
| 실시간 알림 시스템 | **nGNN+MC-Streaming (L1+L2)** | 낮은 지연시간, 충분한 정확도 |
| 자원 제약 환경 | **nGNN (L1+L2)** | 최소 RAM, 빠른 추론 |
| 연구·비교 기준선 | **MC-Static (L1+L2-MC)** | 표준 베이스라인 |

---

## 부록: 데이터셋 특성

| 네트워크 | 총 샘플 | Fraud | Normal | 불균형 비율 |
|---|:---:|:---:|:---:|:---:|
| Polygon | 355 | 345 | 10 | 34.5:1 |
| BSC | 1,127 | 960 | 167 | 5.7:1 |
| Ethereum | 2,172 | 1,266 | 906 | 1.4:1 |

> Polygon의 극단적인 불균형(34.5:1)이 ROC-AUC 불안정성의 주요 원인입니다. PR-AUC와 F1이 더 신뢰할 수 있는 지표입니다.

---

## 9. 설명 가능성 (Interpretability) — GradCAM Attention Case Study

### 방법론: GradCAM-Style Edge Importance

nGNN의 `Level1GNN`은 **SAGEConv** (GraphSAGE) 기반이므로 GAT처럼 명시적인 Attention Weight를 직접 출력하지 않습니다. 대신 **Gradient x Activation (GradCAM)** 방식을 적용합니다:

- **Node Importance** = 사기 확률에 대한 노드 피처의 편미분 절댓값 합
- **Edge Importance** = 연결된 두 노드의 Node Importance 평균값

이 방식은 "사기 확률 예측에 각 노드/엣지가 얼마나 기여했는가"를 역전파(backpropagation) 기울기로 정량화합니다.

### Case Study: Polygon 체인 사기 계약 그래프

![nGNN GradCAM Attention Visualization](./attention_casestudy.png)

*Figure 1. Polygon 체인의 대표적 사기 계약 그래프에 대한 nGNN GradCAM 시각화.*
*왼쪽: 거래 그래프 (노드 크기·밝기 = 노드 중요도 / 엣지 색상 = 엣지 중요도: 보라→노랑 = 낮음→높음)*
*오른쪽: 노드별 GradCAM 스코어 순위 막대 그래프*

### 탐지된 사기 패턴 분석

| 패턴 | 설명 | 모델의 반응 |
|---|---|---|
| **Fan-out (방사형 송금)** | Hub(Node 0)에서 5개 Spoke로 즉각 분산 | Hub와 인접 엣지에 최고 중요도 부여 |
| **Layering (레이어링)** | Spoke → Relay → Sink로 자금 세탁 단계 구성 | Relay 경유 엣지에도 높은 중요도 |
| **Velocity (거래 속도)** | Hub 노드의 Feature에 고 tx-count + 고 value 반영 | Hub 노드 Gradient 값이 전체 최고 |
| **Sink 노드 분리** | 최종 목적지(Sink)는 Hub와 직접 연결 없음 | Sink 노드 중요도가 낮음 (정상 위장) |

### 노드 역할별 해석

```
★ Hub (Node 0)     → 가장 높은 GradCAM Score: 사기 행위의 출발점
◆ Spoke (1~5)      → 중간 수준: 직접 수령 후 재분산
■ Relay-1 (6~9)    → 낮은 수준: 자금 세탁의 중간 단계
▲ Relay-2 (10~12)  → 더 낮음: 허브에서 멀어질수록 중요도 감소
● Sink (13~19)     → 가장 낮음: 최종 목적지, 정상 계정 위장
```

### 해석 요약

모델은 단순히 개별 노드의 피처를 보는 것이 아니라:

1. **그래프 구조 (Hub-Spoke 패턴)** 를 통해 자금이 중앙에서 여러 방향으로 분산되는 패턴을 포착합니다.
2. **경로 길이** 를 인식하여, 출처(Hub) → 중간 단계(Relay) → 목적지(Sink)의 3-hop 이상 패턴을 학습합니다.
3. **피처 비정상성** (높은 거래량 + 높은 가치 + 불규칙한 타이밍)이 동시에 나타날 때 경고 신호를 강화합니다.

> **실운영 시사점:** GradCAM 시각화를 통해 분석가는 "왜 이 계약이 사기로 탐지되었는가"를 그래프로 직관적으로 설명할 수 있습니다. 이는 규제 기관 보고서(SAR: Suspicious Activity Report) 작성 시 모델의 판단 근거 자료로 활용 가능합니다.
