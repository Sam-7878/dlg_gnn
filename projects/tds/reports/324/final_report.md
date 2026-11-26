# TDS R324 작업보고서 — 논문 수정(쉬운 서술 + R323 리뷰 반영)

- 작업일: 2026-11-26
- 범위: 논문 원고·Supplement·표·그림만 수정. 새 실험은 없음. 저자 승인, 라이선스 결정, Zenodo/GitHub 공개는 랩미팅 이후 저자 확인 단계로 남김.
- 상태: **REVIEW BUILD READY**. DOI 자리는 빨간색 자리표시(`[DOI-PLACEHOLDER: versioned Zenodo DOI]`)로 표시됨. DOI를 받기 전에는 `--final` 빌드가 거부됨.

## 1. 산출물

| 구분 | 경로 | 비고 |
|---|---|---|
| Preprints.org 원고 | `projects/tds/submission/preprints/324/preprint_article.pdf` | 25쪽 (R323 23쪽) |
| Applied Sciences 원고 | `projects/tds/submission/applsci/324/applsci_article.pdf` | 23쪽 (R323 21쪽) |
| Supplement (공통) | `projects/tds/submission/{preprints,applsci}/324/supplement.pdf` | 8쪽, 두 venue에서 바이트 동일 |
| 소스 ZIP | `preprint_source.zip`, `mdpi_source.zip` | 압축을 푼 깨끗한 폴더에서 단독 컴파일 확인 |
| Cover letter 초안 | `projects/tds/submission/applsci/324/cover_letter.md` | Preprint DOI·Zenodo DOI·추천 심사자는 자리표시로 둠 |
| 원고 소스 | `projects/tds/paper/submission_324/common/` (공통), `venues/` (템플릿) | paper/는 git 제외 영역 |
| 빌드·QA 스크립트 | `projects/tds/scripts/revision324/{build.py,qa.py,README.md}` | |
| 빌드 영수증·QA | `projects/tds/reports/324/{build_receipt.json,qa.json}` | QA 35개 항목 전부 PASS |

재빌드: `python scripts/revision324/build.py` → `python scripts/revision324/qa.py` (`.venv_cuda` 사용). 공통 파일(본문, 표, 그림, 참고문헌, Supplement)은 두 venue에서 바이트 단위로 동일한지 빌드가 검사함(P2-4).

## 2. 쉬운 서술로 바꾼 내용 (사용자 요청)

목표는 GNN·사기탐지·통계 중 한 분야만 아는 심사자도 따라올 수 있게 하는 것이었다. 내용과 수치는 바꾸지 않았다.

1. **Abstract 전면 재작성 (195단어)**: "무엇이 문제인가 → 무엇을 했나 → 무엇을 발견했나 → 무엇을 제안하나" 순서로 썼다. 전문용어는 처음 나올 때 풀어 썼다(AP, GIN 등). 특별호 각도로 label-free anomaly detector 결과(DOMINANT ROC-AUC 0.1088, CoLA AP lift +0.0670/+0.0913)와 7일 코호트 prevalence 56.16%를 넣었다.
2. **Keywords**: `anomaly detection`, `graph neural networks`, `blockchain fraud detection`, `prevalence shift`를 앞에 배치.
3. **Introduction 재구성(P1-1)**: 동기(사기 컨트랙트 탐지 필요) → 공백(어느 시점으로 정렬하는가, 크기 효과) → RQ 3개 → RQ별 기여 3개(1:1) → 논문 구성.
4. **새 Figure 1 (연구 설계 개념도)**: (a) 같은 컨트랙트를 첫/마지막 거래로 정렬하면 test set이 어떻게 달라지는지, (b) 7일 코호트의 cutoff, (c) 감사 4단계. 이전 그림은 4개 상자뿐이었고, 캡션은 그림에 없는 내용(체인, Cohort A/B)을 설명하고 있었다.
5. **핵심 용어표(Table 1) 신설**: contract, f/l, lifetime, cutoff, archived/seven-day cohort, prevalence, AP·AP lift, ROC-AUC, top-k capture, count-only control. 코드명(`original`/`horizon7`) 대응도 여기에 적었다(P0-4 매핑 각주 역할).
6. **Methods**: 한 문단에 정보를 몰아넣은 문장을 짧게 나눴다. 굵은 소제목(Supervised reference panel, Training and seeds 등)을 달고, 각 장치의 목적("왜 하는가")을 첫 문장에 썼다.
7. **Results**: 소제목을 결론형으로 바꿨다(예: "Seven-Day Cohort: Graph Size Explains Most of the GIN Gap"). 표 앞에 해석 문장을 먼저 두고, `data/*.tex`에 흩어져 있던 서술 조각을 본문으로 옮겼다.
8. **Discussion (P1-4)**: 단락마다 굵은 핵심 문장으로 시작하게 했다. 8단락, 인용 8건(GADBench, Shchur, Errica, Dwivedi, Heidrich, Moreno-Torres, Kapoor, Zhou).
9. **Conclusions**: "아카이브를 어떻게 자르느냐가 모델만큼 중요하다"는 메시지로 시작하게 했다.

**제목 제안(미반영, 저자 결정 사항)**: 현재 제목은 저자 기록 및 이전 버전과의 연속성 때문에 유지했다. 더 쉬운 대안은 다음과 같다.
- *When the Test Set Changes: Observation Time and Graph Size in Blockchain Fraud Screening*
- *How Data Splits and Graph Size Shape Graph-Based Fraud Screening of Blockchain Contracts*

## 3. R324 작업지시서 항목별 처리

| 항목 | 처리 | 근거·위치 |
|---|---|---|
| P0-1 승인 대기 문구 | **원고 부분 완료.** "pending", "approval", "subject to review" 문구 0건(QA). DOI는 `\TDSDataDOI` 매크로 하나로 모았고, 저자 승인 2건은 저자 단계로 남김 | `common/replay_reference.tex` |
| P0-2 라이선스 | **원고 부분 완료.** DAS를 최종 형태로 다시 썼다(코드 MIT, score `\TDSScoreLicense`=CC BY-NC-SA 4.0). 최종 결정은 저자 단계 | `declarations.tex` |
| P0-3 Run count | 완료. 결정적 적합은 "1†"로 표기하고 동일 각주 매크로 `\TDSDeterministicNote`를 Table 4(구 T3)·Table 7(구 T8)·S8(구 S7)·S1·S2·S12에서 문자열 단위로 공유. `± 0.0000` 0건. lightgbm_fixed 5개 복제 배열은 "repeated with other seeds, the scores were identical"로 명시 | 데이터: `score_summary.csv` runs |
| P0-4 코호트 명칭 | 완료. archived / seven-day 두 이름만 사용. `original`·`horizon7`은 용어표와 Supplement 머리말의 코드명 매핑에만 남음(QA) | |
| P0-5 오탈자·서식 | 완료. SD는 `2.1×10⁻⁵`로 표기. 지적된 공백 누락 두 문장은 새로 쓴 문장으로 대체했고 조판 결과를 육안으로 확인 | |
| P0-6 S8 체크리스트 | 완료. "Measured consequence in this study" 열을 추가하고 7행(실제 행 수) 모두 수치로 채움 | 현 Table S9 |
| P0-7 GIN-sum/4-count | 완료. Table 6 캡션에 0.44379 / 0.44375 명시 | |
| P0-8 Seed | 완료. 본문에 seed 문장, Supplement에 seed 표(S15) 신설. 321의 유래(첫 탐색 실행)를 명시 | `scripts/revision321/unsupervised.py` SEED=321 |
| P1-1 Introduction | 완료 (2절 참조) | |
| P1-2 Abstract | 완료. 195단어, 수치 3개 이상, 특별호 키워드 포함 | |
| P1-3 Table 6/8 중복 | 완료. 구 Table 6(7일 첫 패널)을 Supplement S2로 이동하고 SD는 보존. 모든 표 앞에 본문 문장이 오도록 배치 | |
| P1-4 Discussion | 완료 (2절 참조). 수치 0.0631/0.6755/0.7122/16.45/16.06 포함 | |
| P1-5 Table 14 SD | 완료. "3,618 of the 3,648 (106 of 107 positives)" 각주 추가. 부록 질문 3(107 vs 106)의 원인: 목표 2.93%가 실제 2.933%보다 약간 낮게 반올림되어 `floor(3541·0.0293/0.9707)=106` | `check_diagnostics.py` |
| P1-6 버전 | 완료. 본문과 Supplement S16. 출처는 `results/317/environment.json`과 `.venv_cuda`. 기존 패널(controlled reference, archival)은 해시로 식별되는 보존 산출물이라고 명시 | |
| P1-7 UTC-day 구간 | 완료. Table 8(구 T9)에 UTC-day 구간 열 추가 | `results/319/new_statistics_319.csv` |
| P1-8 단일 아카이브 | 완료. Limitations 첫 단락 | |
| P1-9 2026 참조 | 완료. arXiv 2605.07133 페이지와 Crossref(app16041906, app16052624)로 제목·저자·권호 일치 확인 | |
| N1 | P0-3으로 처리 | |
| N2 mlp seed 7 | 완료. **재사용이 아니라 같은 seed·설정·epoch(3)에서 나온 결정적 결과**임을 확인하고 S13 캡션에 명시(refit best_epoch=3 = short search epoch 3) | `refit_manifest.csv` |
| N3 | P0-8로 처리 | |
| N4 refit_manifest cohort 공란 | **미처리.** 공개 패키지 작업이라 저자 단계에서 처리 | |
| N5 공개용 permutation 패키지 | **원고만 반영.** DAS에 "공개본은 공통 permutation과 재생성 checksum을 사용한다"고 적음. 패키지 생성은 Zenodo 전에 저자 단계에서 해야 함 | |
| N6 LightGBM 트리 수 | 완료. 3.5절에 99 vs 28 트리, 7일 코호트 validation AP 0.9932 조기 정점 | |
| P2-2 Cover letter | 초안 갱신. 추천 심사자는 저자 결정 사항 | |
| P2-3 Supplement 표지·S1 행 | 완료. MDPI식 표지(저자, 소속, 교신), 목차, S1에 7일 GIN-0/GIN-size/GIN-sum/LightGBM 행 추가 | `results/319/seven_day_matched_models.csv` |
| P2-4 venue 동일성 | 완료. 빌드가 공통 파일의 바이트 동일성을 검사 | |
| P2-1 두 번째 데이터셋 | 범위 밖(지시서 권고대로) | |

표 번호가 바뀌었다(용어표가 Table 1이 됨). 현재 본문 표: 1 용어, 2 정렬, 3 lifetime, 4 archived 주 비교, 5 count control, 6 archived readout, 7 7일 비교, 8 size-readout 통계, 9 activity strata, 10 archived calibration, 11 budget, 12 7일 체인, 13 7일 calibration, 14 matched prevalence, 15 detector. Supplement의 표 번호는 빌드 시 `.aux`에서 자동 생성되므로 본문에 하드코딩된 S번호가 없다(QA).

## 4. 남은 저자 단계 (랩미팅 이후, 순서 고정)

1. 교수님 리뷰 반영. 수정 시 `build.py`의 `SOURCE_DATE_EPOCH`를 그날로 바꾸고 재빌드 → QA.
2. 저자 2인 서면 승인 (P0-1).
3. score 재배포 라이선스 확정 (P0-2). 결정이 바뀌면 `replay_reference.tex`의 `\TDSScoreLicense` 한 줄만 고치면 된다. score를 배포하지 않기로 하면 DAS 문장도 함께 고쳐야 한다.
4. 공개용 permutation 패키지 생성 + checksum 재생성 + `refit_manifest.csv` cohort 열 채우기 (N4, N5).
5. GitHub tag → Zenodo DOI → `\TDSDataDOI`에 DOI 입력 → `build.py --final`.
6. Preprints.org 게재 → Preprint DOI를 cover letter에 넣고 MDPI 투고 (마감 2026-12-31).

## 5. 확인 필요 사항

- 원고가 2~3쪽 늘었다(용어표, Figure 1, Discussion 확장). MDPI Article에는 쪽수 제한이 없다.
- Figure 1은 개략도(not to scale)다. 수치 라벨(2.93%, 56.99%, 56.16%, 19 h)은 본문 표와 같다.
- "One plausible reading, which we did not test" 문장(Discussion, detector 단락)은 검증하지 않은 해석임을 명시한 문장이다. 원치 않으면 삭제해도 논증에는 영향이 없다.
