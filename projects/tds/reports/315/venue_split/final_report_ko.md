# TDS 원고 이원화 및 결과 데이터 분리 완료 보고

기준일: 2026-10-29. 이번 사용자 지시에 따라 이전 `projects/results/tds/` 규칙을
대체하고, TDS 결과를 모두 `projects/tds/results/<작업>/`에 관리하도록 정리했습니다.

## 두 원고 형식

| 용도 | 소스·PDF 위치 | 제출용 소스 ZIP 위치 |
|---|---|---|
| Preprints.org | `projects/tds/paper/preprints/submission_315/` | `projects/tds/submission/preprints/315/TDS_315_preprints_sources.zip` |
| Applied Sciences | `projects/tds/paper/applsci/submission_315/` | `projects/tds/submission/applsci/315/TDS_315_applsci_sources.zip` |

원고 공통 본문·서지·표/그림 소스는 비공개 `paper/common/315/`에 있습니다.
Preprints.org용 원고는 14쪽, MDPI용 원고는 13쪽이며 부록은 각각 4쪽입니다.
두 형식의 과학 본문과 서지는 바이트 수준으로 동일합니다. Preprint에서는
저널 로고·헤더를 제거하고 중립적인 형식을 사용했습니다. AI의 원고 작성 보조
범위도 Methods에 명시했습니다. 지표·실험·과학적 결론은 변경하지 않았습니다.

투고 순서는 **Preprints.org 게시 후 Applied Sciences**입니다. 새 특집호 목표는
[Machine Learning and Its Application for Anomaly Detection](https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt)로
설정했습니다. 기존 특집호의 제목과 마감일을 현재 메타데이터에 전용하지 않았습니다.
새 페이지의 직접 접근이 HTTP 429여서 마감일·실제 포털 접수 상태는 미확인입니다.
Preprint DOI/URL은 아직 없으며, 실제 게시 후 저널 cover letter와 포털에 기입합니다.

## 데이터 및 비공개 자료 위치

- `results/main_final_v2/`: 기존 primary panel, 변경 없음.
- `results/314_applsci/`: 35개 보완 실험, 학습 이력, 통계·분석 CSV 등.
- `results/315_applsci/`: 숫자 공개 후보 예측 65개, AP 재현, 입력 해시·계보 등.
- `results/historical_reviewed_draft/data/`: 과거 원고에 남아 있던 그림 입력 CSV 3개.
- `reports/315/`: 공개용 감사 보고서. 이번 현재 보고서는 `venue_split/`에 있습니다.
- `submission/author_records/315/`: 저자·연구비·선언 기록, 비공개.
- `private/`: 이전 혼합 상태의 원본 ZIP·롤백 사본·원고 렌더링 QA, 비공개.

총 391개 파일을 이동할 때 SHA-256을 비교했습니다. 옛 manifest는 바이트를 보존하고
`publication/layout/path_migrations.json`으로 경로를 해석합니다. 원고 디렉터리 전체에서
결과 CSV·체크포인트·NumPy/Parquet 파일이 남아 있지 않음을 확인했습니다.
공통 원고의 metadata.json은 제목·Abstract·키워드 등 원고 소스 메타데이터입니다.

숫자 결과와 공개 후보 ZIP은 Git 추적 가능하며, 원고 소스·제출 ZIP·저자 기록은
제외됩니다. 식별자가 포함된 원시 예측과 체크포인트 등은 별도 비공개 규칙을 유지하고,
Git에 공개할 예측은 row ID로 축약한 65개 CSV를 사용합니다. 실제 공개 push는 하지 않았습니다.

## 검증

동결 입력 405개를 새 경로에서 검증했고, 보완 실험 35개 및 primary 예측 30개의
정렬·표본 수·양성 수·AP가 일치했습니다. 기존 과학 테스트 8개와 제출 상태 테스트
3개가 통과했습니다(총 11개). 대표 비교의 세 가지 10,000회 통계 재현도 일치합니다.
공개 숫자 ZIP만 신규 폴더에 추출한 재현과 8개 과학 테스트도 통과했습니다.

두 원고/부록의 독립 소스 ZIP clean build가 모두 통과했으며, 총 35쪽의 원본/신규
빌드 텍스트와 렌더링 픽셀이 일치합니다. 최종 컴파일 경고와 미해결 참조가 없습니다.
표·그림·첫 페이지·Methods·선언문·약어 목록을 선택적으로 시각 검토했습니다.
전체 페이지의 자동 비교와 전 페이지 수동 검토는 구분해서 기록했습니다.

추가 학습은 하지 않았습니다. 저자 최종 승인·공식 과제 증빙·남은 선언·공개 릴리스
등 기존의 실제 제출 전 확인 사항은 유지합니다. 두 새 PDF의 해시는 각각의
`approval_identity.json`에 기록하며 이전 후보의 승인 해시를 재사용하지 않습니다.
게시·투고는 수행하지 않았습니다.

## 앞으로의 실행

공통 원고를 수정한 다음 저장소 루트에서 실행합니다.

```bash
source ../.venv_cuda/bin/activate
python projects/tds/scripts/revision315/build.py
python projects/tds/scripts/revision315/claims.py
python projects/tds/scripts/revision315/verify_layout.py
python projects/tds/scripts/revision315/close.py
```

데이터/공개 후보를 갱신할 때는 `audit.py`, `stage.py`, `verify_candidate.py`를
순서대로 실행합니다. 모든 향후 TDS 결과는 해당 작업의 results 하위 폴더에 생성하고
paper에 결과 파일이나 감사 패키지를 다시 복사하지 않습니다.
