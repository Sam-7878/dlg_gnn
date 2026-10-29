# 316 작업 결과

**과학적 증거 동결·원고 보강·두 플랫폼 빌드·로컬 재현 검증은 완료했습니다.**
Preprints.org 게시와 Applied Sciences 제출은 저자·권리·공개 릴리스·실제 포털 확인이
남아 각각 HOLD입니다. 접수/게시/DOI 발급/push는 수행하지 않았습니다.

## 원고와 패키지

| 산출물 | 위치 |
|---|---|
| 공통 비공개 과학 본문 | `projects/tds/paper/submission_316/shared/sections/manuscript_body.tex` |
| Preprints.org PDF | `projects/tds/paper/submission_316/preprints/preprint_article.pdf` |
| MDPI PDF | `projects/tds/paper/submission_316/mdpi_applsci_anomaly/applsci_article.pdf` |
| Preprint 독립 소스 ZIP | `projects/tds/submission/preprints/316/preprint_source.zip` |
| MDPI 독립 소스 ZIP | `projects/tds/submission/applsci/316/mdpi_source.zip` |
| MDPI cover letter/metadata | 같은 폴더의 `cover_letter_anomaly.md`, `metadata_applsci_anomaly.json` |
| 숫자 공개 후보 | `projects/tds/publication/316/TDS_316_numeric_release_candidate.zip` |
| 수치·해시·상태 이력 | `projects/tds/results/316_dual_submission/` |

기존 제목을 두 버전에서 동일하게 유지했습니다. Abstract·Introduction·Related Work를
감독학습 기반 금융 계약 그래프의 이상/사기 위험 선별로 명확히 하고, top1%/5%의
37/183건 조사 예산을 별도 적용 소절로 설명했습니다. 비지도 탐지·사기 발생 시점 예측·
실시간 배포·XAI·프라이버시·고유 GNN 구조 우월성 주장은 추가하지 않았습니다.

Benchmark의 참고문헌 목록을 검토해 **GADBench2023**과 **Qiao 등2025 survey** 두 편만
원출처 확인 후 추가했습니다. 인용 문헌은19→21편이며, 기존 서지 항목은 변경하지 않았습니다.
GADBench의 정적 노드 탐지 결과를 이 계약 그래프의 성능 순위로 전용하지 않았습니다.

## 보존 및 검증

- 현재 R315 입력542개와 리뷰 원본315 PDF/숫자 ZIP을 구분해 동결했습니다.
- packed graph, primary artifact88개, primary checkpoint20개 및 보완 checkpoint35개를
  확인했고, 원본 예측65개의3648건/107양성과 식별자 정렬을 재검증했습니다.
- 65개 숫자 점수 벡터는 리뷰된315 원본 및 이동 후315 후보와 바이트가 같습니다.
  표·그림·AP/CI/Holm·operating metrics·strict-time 결과는 바꾸지 않았습니다.
- 로컬 숫자 ZIP 신규 추출에서 AP 평균/표준편차와 대표 비교의 세 가지10000회 통계가
  재현되었으며, 최종 패키지 테스트18개(과학8+기존게이트3+316게이트7)가 통과했습니다.
- 두 형식의 과학 본문·부록·서지·표/그림 입력은 같습니다. 원고는 각각14쪽, 부록4쪽이며,
  총36쪽의 clean-build 텍스트/렌더링 픽셀이 일치합니다. 최종 컴파일 경고는 없습니다.
- R315 MDPI의13쪽에서316은 응용 설명과 두 인용 추가로14쪽이 되었습니다.
  글꼴/표를 억지로 줄여 이전 쪽수를 맞추지 않았으며 수치 변경은 없습니다.
- 원고 소스·제출 ZIP·저자 기록은 Git 비공개이고, 숫자 증거·코드·보고서는 추적 가능합니다.

## GPU와 공식 양식

RTX3090은24576MiB, 기존 RTX4070 Laptop은8188MiB로 확인했습니다. 장치 메모리는
하나의32GB 공간으로 자동 합쳐지지 않습니다. **316 지시서는 신규 학습을 명시적으로
제외**하므로 이번에는 GPU 용량만 기록하고 재학습/성능 향상 실험은 하지 않았습니다.
과거 메모리 제약 실험 재개는 별도 과학적 amendment가 있을 때 결정할 사항입니다.

공식 MDPI 템플릿 ZIP은 현재 HTTP200으로 내려받았고 기존 클래스와 해시가 동일합니다.
새 특집호는 `Machine Learning and Its Application for Anomaly Detection`입니다.
작업지시서에 보고된 마감 **2026-12-31**을 출처와 함께 반영했습니다. 다만 이번 세션의
공식 SI 페이지 직접 접근은403/429여서 Guest Editor·실제 마감/포털 선택 항목의 독립
확인은 남았습니다. 이전 특집호의 제목·URL·마감은 현재 cover letter/metadata에 없습니다.

## 실제 게시/투고 전에 남은 항목

두 저자의 신원/교신저자와 각각의 최종 PDF 해시 승인, 공식 과제 증빙, 실제 COI·
funder-role·originality·권리 확인, preprint 공개 동의 및 라이선스 승인이 필요합니다.
TDS 숫자 자료의 실제 immutable release와 익명 다운로드/재현 또는 허용된 예외도
확인해야 합니다. A16 문구 적용 권한을 새316 최종 승인으로 간주하지 않았습니다.

현재 local/remote HEAD가 존재하고 일부315 변경이 commit된 사실은 확인했지만,
이는 검증된 TDS 공개 릴리스와 다릅니다. Preprint DOI/URL/버전은 null이며,
접수·게시·저널 제출 flag는 모두 false입니다. 실제 공개 후 발급된 식별자를 그때
MDPI 자료에 반영하도록 분리했습니다. 상세 판정은 submission_readiness_316.json입니다.
