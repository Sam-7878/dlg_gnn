# TDS R322 결과 및 투고 상태

2026-11-06. R321 원고, 증거 및 파일 해시를 비교 기준으로 사용했다. R322 후보의 상태는 **LOCAL REVIEW READY / HOLD AUTHOR ACTION**이다.

| 항목 | 반영 결과 |
|---|---|
| P0-1 | Applied Sciences는 기존 MDPI applsci Article 클래스를 사용한다. 특집호명과 비교 수치가 포함된 단일 문단 초록을 확인했다. Preprints는 별도 중립 템플릿으로 빌드했다. |
| P0-2, P0-3 | 점수 배열, 실행 해시, 분할 정의, CLI 및 재생 코드를 하나의 로컬 후보 ZIP에 넣었다. 내부 경로를 없애고 본문, Data Availability, Supplement가 공통 참조를 사용하도록 했다. Zenodo DOI와 공개 버전 링크는 승인 및 공개 확인 대기다. |
| P0-4 | RTX 3090에서 두 코호트 × 세 검출기 × 다섯 seed로 30회 실행했다. AP, AP lift, ROC-AUC, top-k 평균과 표준편차를 표에 넣었다. 무작위 top-100 기대값은 7일 테스트 56.16건, 원 테스트 2.93건이다. ROC-AUC와 최상위 예산 순위는 별도로 해석했다. |
| P0-5 | 원 코호트의 Full-training Isolation Forest는 학습 라벨을 무시한다. 7일 Negative-only Isolation Forest는 학습 음성 선택에 라벨을 사용하며 one-class, label-informed로 명시했다. 두 점수 방향은 사전에 고정됐다. |
| P0-6 | 두 저자의 기존 대학원 컴퓨터공학과 표기, 이메일, ORCID, 교신저자 별표를 유지했다. 김기형 교수는 아주대 일반대학원 컴퓨터공학과와 학부 사이버보안학과 공식 명단에 각각 등재돼 있다. 박성수 저자의 소속과 ORCID는 사용자가 제공한 저자 기록을 따른다. |
| P1-7 | 양쪽 LightGBM에 동일한 6개 설정을 적용했다. 원 코호트의 기존 단일 설정은 AP 0.4015, 새 validation 선택 모델은 0.3887다. |
| P1-8 | 원 코호트 four-count LR 0.4438과 weighted-BCE GIN 0.4524를 초록, RQ2, 결론에 반영했다. |
| P1-9 | 공유 상위 거래 상대방 기준 3,137개 그룹을 구축하고 2,000회 클러스터 부트스트랩을 수행했다. GIN-size AP 구간은 0.9357–0.9521이다. 배포자 신원 및 train/test 간 actor 독립성은 미측정이다. |
| P1-10 | 본문과 완전히 같은 Supplement 표 세 개와 LR 반복 확인 행을 제거했다. |
| P2 | 지정한 참고문헌 세 건의 원 출처를 대조했다. AI 모델명을 유지하고 제공사·용도·저자 책임을 명시했다. 특집호와 Guest Editor를 적은 cover letter를 작성했다. |

| 코호트 | 검출기 | 실행 수 | 평균 AP | 평균 ROC-AUC | 평균 top-100 포착 |
|---|---|---:|---:|---:|---:|
| horizon7 | CoLA | 5 | 0.6529 | 0.7078 | 16.20 |
| horizon7 | DOMINANT | 5 | 0.4399 | 0.3173 | 33.80 |
| horizon7 | OCGNN | 5 | 0.6024 | 0.5916 | 62.40 |
| horizon7 | if_negative_only | 1 | 0.6068 | 0.6268 | 56.00 |
| original | CoLA | 5 | 0.0963 | 0.7653 | 9.80 |
| original | DOMINANT | 5 | 0.0157 | 0.1088 | 0.00 |
| original | OCGNN | 5 | 0.0404 | 0.5754 | 4.60 |
| original | if_full_train | 5 | 0.0170 | 0.2047 | 0.00 |

동일 신경망 예산 검증은 GIN-size와 MLP 각각에 두 코호트에서 12개 설정과 5 epoch를 적용했다. validation AP로 설정·epoch를 고르고 선택 결과만 test에서 평가했다. 결과는 Supplement의 Matched Neural Tuning Budget 표에 있다.

1·3·14일 입력을 원자료에서 재구성하고 node LR, MLP, GIN-size를 고정 seed 7로 비교했다. 기존 7일 seed 7을 포함한 네 점과 각 테스트의 양성 수·표본수는 본문 그림과 Supplement에 있다. 다른 관측 기간 간 AP는 서로 다른 적격 모집단의 기술 통계다.

원 코호트의 5-seed GIN-size AP는 0.4793, GIN-sum AP는 0.4438다. 7일에서 관측된 큰 개선과 달리 원 코호트 효과는 제한적이었다.


두 소스 ZIP에서 논문과 Supplement를 각각 독립 컴파일했다. 공통 과학 본문, 표, 그림, 선언, 참고문헌의 바이트 일치와 내부 경로 제거를 검사했다. 리뷰가 지정한 참고문헌은 [Dilini et al.](https://www.mdpi.com/2076-3417/16/4/1906), [Zhou et al. preprint](https://arxiv.org/abs/2605.07133), [Heidrich et al.](https://www.mdpi.com/2076-3417/16/5/2624)이다. 전체 참고문헌의 DOI 일괄 재검증과 외부 데이터셋 전이는 이번 라운드에서 수행하지 않았다.

[아주대 일반대학원 컴퓨터공학과 명단](https://www.ajou.ac.kr/gs/research/faculty37.do)과 [아주대 사이버보안학과 교원 명단](https://www.ajou.ac.kr/researcher/sub01/university.do?deptCd=%24cms%24CIZQDAzGYExgjEA&mode=selectList&uniCd=h)을 확인했다. [MDPI 공식 특집호 목록](https://www.mdpi.com/journal/applsci/special_issues?page_no=33)의 해당 항목은 2026-12-31 마감과 Submission Open을 표시한다. 저널 로그인 포털의 접수 상태는 별도 확인 대상이다.

수치 후보 ZIP은 계약 ID, 주소, 타임스탬프를 제외한다. 정확한 top-k 재생을 위해 보존한 배열 순서는 별도 공개 분할과 연결될 수 있다. 두 저자의 대상 파일 승인과 점수 배열 공개 범위 승인이 필요하다. 승인 후 Zenodo의 실제 버전 DOI와 다운로드를 검증하고 동일 참조를 세 곳에 반영한다. Preprints 게시 후 실제 preprint DOI를 MDPI 파일에 반영한다.
