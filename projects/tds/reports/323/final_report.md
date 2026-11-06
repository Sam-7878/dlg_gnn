# TDS R323 결과 및 제출 상태

R322의 파일 해시를 검증하고 그 후보를 R323의 과학 및 조판 비교 기준으로 사용했다. 현재 WSL의 Ubuntu 배포판은 26.04.1 LTS이며 설치된 GPU 가상환경은 `.venv_cuda` (Python 3.14.4)다. WSL 등록 목록에 24.04가 없어서 추가 실험은 이 환경의 RTX 3090에서 실행했다.

| 코호트 | DOMINANT 점수 | 5시드 평균 AP | 평균 ROC-AUC | 평균 top-100 포착 |
|---|---|---:|---:|---:|
| original | DOMINANT | 0.0157 | 0.1088 | 0.00 |
| original | DOMINANT_low_error_diagnostic | 0.2329 | 0.8912 | 32.80 |
| horizon7 | DOMINANT | 0.4399 | 0.3173 | 33.80 |
| horizon7 | DOMINANT_low_error_diagnostic | 0.6642 | 0.6827 | 82.00 |

PyGOD는 재구성 손실을 그대로 score로 내보내고 높은 score를 outlier 쪽으로 분류한다. 계약 어댑터도 양의 재구성 오차를 평균한다. 낮은 오차를 양성 쪽에 두면 inherited label 순위가 개선되지만, 이는 테스트 확인 후 얻은 역방향 민감도다. P0-1의 수치 재채점은 두 부호 모두 수행했고 primary anomaly 계약에는 원 방향을 적용했다. [PyGOD DOMINANT 구현](https://docs.pygod.org/en/latest/_modules/pygod/detector/dominant.html)

| 지시 | 처리 |
|---|---|
| P0-2 LightGBM | 원 코호트 고정 구성 AP 0.4015를 이전 5개 점수 배열에서 재생했다. validation 6셀 최적 모델 AP 0.3887와 분리했다. |
| P0-3 DOI·라이선스 | MIT 코드를 점수의 제안 라이선스 CC BY-NC-SA 4.0과 분리했다. 순서 보존 배열은 원본 분할에 연결될 수 있어 검토용 ZIP에만 포함했다. 두 저자의 해당 범위 승인과 상류 약관 확인이 필요하다. DOI·공개 태그 상태는 대기다. |
| P0-4 상호참조 | 유병률 분석은 S6, 거래 상대방은 S9다. 원 코호트 GIN–MLP 블록은 UTC-hour, 별도 7일 크기 비교의 블록은 UTC-day로 실제 코드를 따라 구분했다. |
| P0-5 CoLA | 전 순위 AP와 상위 100건 포착을 구분해 해석했다. |
| P0-6 부분집합 | 동일 순위를 반복한 재적합 행을 삭제했다. 기존 노드 LR의 활동 구간별 값은 마스크와 점수 배열에서 재생한다. |
| P0-7 저자·형식 | 사용자 지정 소속과 교신저자 표기를 적용하고 저작권 단어 분절 및 떠 있는 소제목을 수정했다. 공저자 최종 서면 승인 대기다. |
| P1-1~3 | 동기·기여·구성 문단을 추가했다. 2.93%의 공유 재표집과 top-1% 포착을 본문 표에 제시했다. |
| P1-4 | 1·3·7·14일의 신경망 두 종을 각각 5 seed로 비교했다. 7일 및 각 창 seed 7은 보존 점수를 사용했다. 결정적 LR은 창마다 1회다. 단기 validation 탐색의 선택 설정을 3 seed, 최대 60 epoch, patience 5로 확인했다. 전체 12셀을 긴 예산으로 재탐색하지 않았다. |
| P1-5~7 | 코호트별 AP와 prevalence 그림, 관측창 눈금, 약어를 정리했다. GoG DOI와 Related Work를 갱신했다. |
| P1-8 | 두 IF 설정을 7일 코호트에 각 5 seed로 정리했다. 원래 저장된 negative-only seed 7을 포함한다. 모델명과 제공사 표기는 기존 저자 확인을 따른다. |

S1 전체 지표, S2 보존 비교, S3 통계 비교, S4 체인, S5 예산, S6 유병률 재표집, S7 LightGBM, S8 체크리스트, S9 endpoint, S10 단기 신경망 탐색, S11 관측창, S12 긴 refit, S13 방향 진단. S6의 목표 비율은 실제 코드와 표를 따라 약 2.93%로 기록했다. 지시서 P1-2의 2.3%와 차이가 있다.

논문 PDF와 Supplement는 각 출판사 후보에 별도 제공한다. 수치 재현 검토 ZIP과 비공개 원고는 별도 경로에 저장했다. 양 저자의 정확한 파일 및 점수 배포 범위 승인 후 Zenodo 버전 DOI를 검증해 세 참조 위치를 갱신한다. 이후 실제 Preprints DOI를 받은 뒤 MDPI 원고에 반영한다.

GoG DOI는 [NeurIPS 원문](https://papers.nips.cc/paper_files/paper/2024/hash/3205b048f9cc54b9f7963db0b0f52d53-Abstract-Datasets_and_Benchmarks_Track.html)에서 확인했다. 2026년 세 인용은 [Dilini et al.](https://www.mdpi.com/2076-3417/16/4/1906), [Zhou et al. preprint](https://arxiv.org/abs/2605.07133), [Heidrich et al.](https://www.mdpi.com/2076-3417/16/5/2624)의 출처와 대조했다. 추가 외부 데이터셋, 추천 심사위원 및 전체 참고문헌 DOI 일괄 대조는 후속 범위다. 기존의 별도 Preprints와 Applied Sciences 템플릿을 사용했다. Supplement는 두 버전에 공통인 별도 문서다.

MDPI 공식 특집호의 표시 마감일은 2026-12-31이다. 공식 APC는 CHF 2,400이며, 기관별 할인 적용 여부는 미확인이다. [특집호](https://www.mdpi.com/journal/applsci/special_issues/C67H3U9QGT) · [APC](https://www.mdpi.com/journal/applsci/apc)

상태 **LOCAL REVIEW READY / HOLD AUTHOR ACTION**.
