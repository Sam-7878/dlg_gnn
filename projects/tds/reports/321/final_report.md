# TDS R321 검토 및 투고 준비

2026-11-06. R320의 제출 후보와 증거는 비교 기준으로 삼았다. Applied Sciences 원고는 R320부터 MDPI 클래스와 참고문헌 스타일을 사용했으며 Preprints 원고는 별도 중립 템플릿을 사용한다. Grok이 본 파일의 형식과 두 제출 후보의 형식을 구분했다.

| 작업 | 결과 |
|---|---|
| P0-1 | 두 원고에서 저자가 직접 확인한 Claude Opus 5.5, Gemini 4 Argon, ChatGPT 6.1 Sol을 유지하고 Methods와 Acknowledgments를 정리했다. Data Availability는 저장소, GoG 출처 및 조건부 접근 범위를 적었다. 실제 DOI 확인 후 교체한다. |
| P0-2, P0-8 | RQ1–RQ3, Discussion, Conclusions 및 초록을 표본 시각, 크기와 활동, 평가 감사 중심으로 맞췄다. |
| P0-3, P0-4 | 7일 이상 테스트 1,929건 중 양성 379건. 노드 LR AP 0.4175, ROC-AUC 0.5926, lift +0.2210. 조건부 학습은 학습 계약 15,763건과 양성 1,428건을 사용했다. 원 코호트 AP는 노드 0.2354, 엣지 0.0660, 네 카운트 0.4438이다. |
| P0-5 | 0.6358과 0.6418은 동일 split, seed ID, GIN mean–max 및 weighted BCE를 쓴 별도 학습 결과다. 수치 차이를 특정 구현 요인으로 단정하지 않는다. 새 family는 4,000 swap으로 Holm 최소값 4/4001, 이전 family는 10,000회로 분리했다. |
| P0-6 | Figure 1을 절차만 담은 PDF/SVG로 교체했다. 분기·체인 해석, 세 수명 구간, 다중 실행 AP SD와 단일 실행 표시를 추가했다. 체크리스트를 Supplement S10으로 옮겼다. |
| P0-7 | 두 Supplement의 S1–S10을 정리했다. 코드와 집계 수치의 로컬 Zenodo 후보를 구성했다. DOI 발급과 공개 다운로드 확인은 저자 승인 대기다. |
| P1-1 | 3090으로 PyGOD 기반 계약 단위 변형 3종을 두 코호트에 실행했다. 단일 seed, 5 epoch, 폭 32, 계약당 평균 노드 점수의 탐색 대조군이다. CoLA corruption과 OCGNN 중심 추정은 계약 분할에 맞춰 구현했다. 표준 PyGOD 노드 검출기 결과와 구분한다. |
| P1-2 | AP lift·ratio 수치와 목표 유병률 2.93%, 10%, 30%에서의 200회 고정 점수 민감도를 추가했다. 새 1·3·14·30일 재구성 및 재학습은 후속 작업이다. |
| P1-3, P1-4 | 실제 LightGBM 튜닝 차이를 명시했다. 균등 튜닝과 actor-grouped split 재학습은 미완료다. 식별된 배포자 집합이 없는 상황에서 계약 ID를 actor ID로 사용하지 않았다. |
| P1-5 | 설치형 audit CLI와 합성 예제를 작성했다. ordering, cutoff, counts, ap 명령을 로컬 설치본에서 확인했다. |
| P2 | 방법론 연결이 약한 자기인용을 제거하고 BSC 차이를 해석했다. 외부 데이터 전이 및 discovery timestamp 수집은 미완료다. |

R320 기록의 연구비 증빙 상태를 승계했다. R321 원고와 소스 ZIP의 해시는 새 승인서에 기록한다. R320의 권리 매트릭스는 코드와 집계 수치를 공개 가능 범위로 분류한다. 후보 ZIP에 원자료, 체크포인트, 주소, 계약별 예측 및 원고 소스를 포함하지 않았다. 전체 점수 재생에는 별도 승인된 입력이 필요하다.

게시 순서는 양 저자의 대상 파일 및 데이터 공개 범위 승인 → Zenodo 버전 DOI와 다운로드 검증 → DOI 반영 후 Preprints.org 게시 → preprint DOI를 투고 원고와 cover letter에 반영 → Applied Sciences 제출이다. MDPI의 [공식 특집호 목록](https://www.mdpi.com/journal/applsci/special_issues?page_no=33)은 해당 특집호의 마감을 2026-12-31로 표시한다. 포털 접수 상태는 실제 제출 시 확인한다. [특집호 페이지](https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt) · [Zenodo DOI 안내](https://help.zenodo.org/docs/deposit/describe-records/reserve-doi/) · [PyGOD API](https://docs.pygod.org/en/latest/pygod.detector.html)

상태는 **LOCAL_REVIEW_READY / HOLD_AUTHOR_ACTION**이다.
