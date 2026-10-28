A18의 핵심 수치 복구는 상당 부분 확인됐습니다. 공개된 210개 시드별 경보 예산 레코드에서 수학적 불변식 위반이 없었고, F1도 455개 시드 값의 집계가 최종 표와 일치합니다.

다만 **“수치가 집계상 맞다”와 “원시 점수부터 독립 재계산됐다”는 다릅니다.** A18의 경보 예산 검증 스크립트는 `scores.npz`의 해시를 확인하지만, TP@K는 `metrics.json`에서 읽습니다. 또 F1 455건 중 420건은 임계값 정보가 비어 있어 “검증셋에서만 선택한 임계값을 테스트셋에 적용했다”는 표 설명을 현재 증거만으로는 확인할 수 없습니다. 마지막 제출용 정리는 이 두 가지 **증거 수준과 설명의 일치**에 집중해야 합니다.


A18 실행 보고서와 업로드하신 `243_A18.zip`, `submission.zip`을 검토했습니다. 또한 공개 GitHub 저장소의 A18 검증 스크립트와 결과 CSV를 이용해 **경보 예산 지표 210건, F1 지표 455건의 수치 계산을 독립적으로 확인**했습니다. Preprints.org용 21페이지 PDF와 Applied Sciences용 18페이지 PDF도 확인했습니다.

## 1. 최종 판정

**A18의 핵심 과학적 오류 수리는 상당 부분 성공했습니다. 다만 `PREPRINT READY`를 최종 승인하기 전에 작은 규모의 증거·재현성 정리가 필요합니다.**

이번 검토에서 가장 긍정적인 점은 이전 A17의 치명적 문제였던 Table 3의 Recall@K 오류가 실제로 해결됐다는 것입니다.

| 검토 항목                                  | 독립 검토 결과   |
| -------------------------------------- | ---------- |
| Table 3 Precision@K / Recall@K 수학적 정합성 | ✅ PASS     |
| F1 455개 값의 평균·표준편차 계산                  | ✅ PASS     |
| Figure 1의 DLG 구조 수정                    | ✅ 대체로 PASS |
| Preprints/MDPI TeX·PDF 생성              | ✅ PASS     |
| 원시 점수에서 Table 3 직접 재계산 여부              | ⚠️ 추가 확인   |
| F1의 validation-only threshold 적용 여부    | ⚠️ 추가 확인   |
| 평가 분모 및 실행 출처 문서                       | ❌ 불일치 발견   |
| A18 공개 릴리스와 제출 파일 해시                   | ⚠️ 수정 필요   |

### Table 3은 실제로 개선됐습니다

공개된 `alert_budget_per_seed_verified.csv`를 독립 계산한 결과, 210개 레코드에서 다음 관계가 모두 성립했습니다.

$$
P@K=\frac{TP@K}{K},\qquad
R@K=\frac{TP@K}{M_{\mathrm{test}}}
$$

위반은 **0건**입니다.

또한 최종 논문의 Ethereum DOMINANT 결과인 P@1% = 0.4897, R@1% = 0.0121이 시드별 데이터의 실제 집계와 일치했습니다.

F1도 마찬가지입니다. 455개 시드별 F1을 다시 집계한 결과, 예를 들어 Elliptic DLG-Aug의 **0.2239 ± 0.0164**, BitcoinOTC DOMINANT의 **0.9329 ± 0.0060**이 최종 표와 일치했습니다.

이것은 A18의 중요한 성과입니다.

## 2. 다만 두 가지 과학적 검증은 아직 완전히 끝났다고 볼 수 없습니다

**첫째, Table 3의 검증은 현재 코드상 `metrics.json`의 수학적 정합성을 검사합니다.**

A18 스크립트는 `scores.npz`가 존재하는지 확인하고 해시를 계산하지만, 실제 점수 배열을 로딩해 상위 K개 노드를 다시 선정하지는 않습니다.

따라서 현재 입증된 것은 **기록된 TP·Precision·Recall의 계산이 정확하다는 사실**입니다. 원시 점수와 평가 마스크에서 해당 TP가 실제로 재현되는지까지는 별도로 확인해야 합니다.

다행히 암호화폐 105개 실행의 원시 점수는 보존됐다고 하셨으므로, GPU 재학습 없이 확인할 수 있는 문제입니다.

**둘째, F1의 임계값 선택 과정은 증거가 부족합니다.**

A18의 `f1_per_seed_verified.csv`를 확인해 보니 455개 레코드 가운데 **420개에서 threshold 정보가 비어 있습니다.** 나머지 35개 LANL 레코드에는 threshold가 기록돼 있습니다.

현재 논문 Appendix Table A2에는 다음과 같이 기재돼 있습니다.

> Validation-thresholded Test F1 Score. Thresholds selected solely on validation splits.

그러나 A18 검증 스크립트는 `validation_f1`, `f1`, `test_f1` 등 여러 필드 중 발견되는 값을 읽습니다. 따라서 모든 데이터셋에서 **validation에서 선택한 임계값을 test에 적용했다는 사실**은 이 스크립트만으로 확인되지 않습니다.

이것이 곧 데이터 누출이나 F1 계산 오류라는 뜻은 아닙니다. 원래 실행 코드에서 임계값 선택 과정을 확인하거나, 확인할 수 없는 캠페인에 대해서는 표 설명을 정확하게 제한하면 됩니다.

## 3. 재현성 문서의 불일치도 발견했습니다

`a18_authoritative_source_map.md`에 기재된 암호화폐 평가 분모가 실제 A18 CSV와 다릅니다.

| 데이터셋     | A18 Source Map | 실제 검증 CSV           |
| -------- | -------------- | ------------------- |
| Ethereum | N=1,428, M=584 | N=2,893, 평균 M=1,183 |
| BSC      | N=1,500, M=221 | N=1,500, 평균 M=221   |
| Polygon  | N=4,641, M=120 | N=471, 평균 M=12.2    |

또한 Source Map에는 동결된 13+1 데이터셋 계약과 맞지 않는 미지원 모델 설명도 일부 남아 있습니다.

**최종 논문의 Table 3 수치는 올바른 평가 집단을 사용하고 있습니다.** 따라서 이것은 새로운 실험을 수행할 문제가 아니라, 잘못 생성된 출처 설명 문서를 수정해야 하는 문제입니다.

공개 릴리스와 해시에도 정리할 부분이 있습니다.

현재 논문이 참조하는 `benchmark-a16-submission-2026-10-28` 태그에는 A18에서 수정한 검증 CSV가 없습니다. 해당 CSV는 GitHub `main`에서 확인됩니다.

또한 실제 업로드된 제출 파일의 SHA-256은 A18의 별도 submission gate 보고서와는 일치하지만, **A18 Execution Report에 기록된 해시와는 다릅니다.** 제출 폴더의 기본 게이트 보고서에도 과거 A13 내용이 남아 있습니다.

이 부분은 하나의 최종 버전과 하나의 해시 원장으로 통일해야 합니다.

## 4. A19 최소 작업지시서를 작성했습니다

이번에도 **새로운 GNN 학습이나 대규모 벤치마크 재실험은 권하지 않습니다.**

필요한 추가 작업은 기존 원시 점수와 실행 코드로 평가 근거를 최종 확인하고, 잘못된 출처 문서 및 릴리스 메타데이터를 정리하는 것입니다.

**[A18 Independent Review 다운로드](dlg_gnn/docs/work_reports/benchmark/244_A19/DLG_Benchmark_A18_Independent_Review_2026-10-28.md)**

**[A19 Final Evidence and Release Closure Work Order 다운로드](dlg_gnn/docs/work_reports/benchmark/244_A19/DLG_Benchmark_A19_Final_Evidence_And_Release_Closure_Work_Order_2026-10-28.md)**

[**A19 SHA-256 파일](sandbox:/mnt/data/DLG_Benchmark_A19_Final_Evidence_And_Release_Closure_Work_Order_2026-10-28.md.sha256)**

A19에는 여섯 개의 검증 게이트를 두었습니다. Table 3의 원시 점수 기반 검증, F1 평가 의미 확인, 평가 분모 Source Map 수정, 최종 Git 태그 발행, PDF·ZIP·보고서 해시 일치, 최종 제출 승인입니다.

**작업량은 CPU 기반 검증과 문서 수정으로 제한했습니다.**

## 5. Applied Sciences 투고 가능성

연구 주제 자체는 여전히 적합합니다.

해당 특집호 *Machine Learning and Its Application for Anomaly Detection*는 금융·사이버보안 분야의 비지도 이상탐지와 효율적인 머신러닝 응용을 모집하며, 제출 마감은 **2026년 12월 31일**입니다. ([MDPI][1])

제 판단으로는 A18에서 논문의 과학적 완성도가 상당히 높아졌습니다. 특히 잘못된 Recall@K 수치를 바로잡고, F1을 실행 원본에 연결했으며, DLG 구조 그림을 실제 코드에 맞게 수정한 것은 심사 대응상 의미 있는 개선입니다.

다만 **수치 계산이 맞는 것과 평가 절차까지 재현됐다는 주장은 구분해야 합니다.** A19는 바로 이 차이를 해소하기 위한 마지막 증거 정리 단계입니다.

A19에서 해당 문제를 해결하거나 증거 범위를 정직하게 명시한다면, 더 이상의 실험 확장 없이 **Preprints.org 등록 → Benchmark DOI 확보 → Applied Sciences 투고**로 진행하는 것이 합리적입니다.

[1]: https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt?utm_source=chatgpt.com "Machine Learning and Its Application for Anomaly Detection | Applied Sciences | MDPI"
