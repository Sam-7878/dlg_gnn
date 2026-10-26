
A15 Submission Evidence Closure 작업지시서([DLG_Benchmark_A15_Submission_Evidence_Closure_Work_Order_2026-10-24.md](file:///d:/_Work/goat_bank/dlg_gnn/docs/work_reports/benchmark/240_A15/DLG_Benchmark_A15_Submission_Evidence_Closure_Work_Order_2026-10-24.md))에 명시된 모든 요구사항과 검증 게이트(A15-G1 ~ A15-G8)를 완벽히 통과하고, Preprints.org 및 MDPI Applied Sciences Special Issue 제출 패키지 생성을 완료했습니다.

---

### 1. 핵심 해결 사항 및 작업 요약

1. **평가 양성률(Test Prevalence) 검증 및 분모 정합성 해결 (A15-G1: HARD GATE 통과)**
   - **문제 원인:** 이전 초안 Table 2의 암호화폐 데이터셋 AP 기준선에 과거 초안 수치($0.0195 / 0.0210 / 0.0052$)가 잔존하여 Table 1의 전체 그래프 양성률과 불일치하는 출판 차단(Publication Blocker) 결함이 존재했습니다.
   - **해결:** 모델 평가 시 사용된 5개 시드(42~46)의 $20\%$ 테스트 마스크 분모와 실제 양성 노드 수를 전수 조사하여 [evaluation_population_by_seed.csv](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/evaluation_population_by_seed.csv) (70개 실행 전수)를 구축했습니다.
     - **Ethereum:** 테스트 분모 $2,893$개 중 평균 양성 $1,183.0$개 $\to$ 평가 양성률 **$\pi_{\text{test}} = 0.4089$** (범위 $0.3951 \sim 0.4307$)
     - **BSC:** 테스트 분모 $1,500$개 중 평균 양성 $221.0$개 $\to$ 평가 양성률 **$\pi_{\text{test}} = 0.1473$** (범위 $0.1353 \sim 0.1600$)
     - **Polygon:** 테스트 분모 $471$개 중 평균 양성 $12.2$개 $\to$ 평가 양성률 **$\pi_{\text{test}} = 0.0259$** (범위 $0.0149 \sim 0.0382$)
   - Table 1(전체 그래프 특성)과 Table 2(평가 테스트셋 양성률 $\pi_{\text{test}}$)의 표기 및 각주를 완전히 분리하여 논리적 정합성을 확립했습니다.

2. **미지원 모델 분류 및 모델 지원 수 정정 (A15-G2: HARD GATE 통과)**
   - 미지원 모델–데이터셋 쌍 7개의 구체적 원인을 원천 로그에 근거하여 명시했습니다:
     - **GADNR (5개 OOM):** Elliptic, DGraphFin, Yelp-Syn, Flickr-Syn, Reddit-Syn
     - **AnomalyDAE (2개 OOM):** Yelp-Syn, Reddit-Syn ($N \times N$ dense 교차 어텐션 메모리 초과)
   - 본문 오탈자("GADNR fails on 7 graphs")를 정정하여 총 91개 1차 모델–데이터셋 쌍 중 **84개 지원 ($84 \times 5 = 420$회) + 외부 LANL 7개 지원 ($7 \times 5 = 35$회) = 총 455회 실행** 완료 수치를 일관되게 동기화했습니다.

3. **실행 보고서 오류 표 교체 및 비인과적 관찰 서술 정정 (A15-G3 통과)**
   - **Section 2.4 오류 표 교체:** A14 보고서에 잘못 삽입되었던 비계약 데이터셋(Tolokers, Questions, T-Finance, T-Social, YelpChi)을 완전히 삭제하고, [conditional_ap_14.csv](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/conditional_ap_14.csv)에 기반한 **13+1 정규 데이터셋 Delta 표**로 전면 교체했습니다.
   - Table 4의 지나치게 단정적인 인과적 서술(예: "허브 편향 극복", "국소 유사도가 사기 탐지를 향상시킴")을 객관적인 관찰 서술("construct similarity graph 평가에서 더 높은 AP 관찰됨", "메커니즘은 해당 비교로 격리되지 않음")로 정제했습니다.

4. **클린 디렉토리 독립 압축해제 빌드 및 SHA-256 디스크 바이트 검증 (A15-G5 통과)**
   - `/tmp/a15_clean_unpack_test` 임시 디렉토리에서 `source_bundle.zip`을 독립 압축해제 후 TeX Live 빌드 검증:
     - **Preprints.org:** **19페이지** 정상 렌더링, 정의되지 않은 참조(0건) 및 인용(0건)
     - **Applied Sciences:** **16페이지** 정상 렌더링, 정의되지 않은 참조(0건) 및 인용(0건)
   - 빌드 후 생성된 최종 파일의 실제 디스크 바이트로부터 SHA-256 체크섬을 재산출하여 동기화했습니다.

5. **확장 음성 대조군(Negative Controls) 검증 (A15-G6 통과)**
   - 암호화폐 과거 양성률 주입, 부호 반전, 비계약 데이터셋 주입, GADNR 오류 개수 변조, 과거 Git 태그 주입, 인과적 금지어 주입 등 **8/8개 음성 대조군 변이 테스트**가 모두 즉각 실패(Fail-Closed)함을 입증했습니다.

6. **공개 Git 릴리스 태그 및 원격 푸시 완료 (A15-G4 통과)**
   - 불변 Git 태그 생성: **`benchmark-a15-submission-2026-10-25`**
   - 원격 저장소(`origin/main` 및 태그) 푸시 및 `git ls-remote` 정합성 확인:
     - Tag Object: `d97188e06e6349164b8556c9244581db3efa8d3c`
     - Target Commit: `a037531c790a91833887e7aad27bf60aeb87fa0e`

---

### 2. 최종 제출 패키지 및 체크섬 (True Disk SHA-256)

#### Preprints.org 제출 패키지 (`projects/benchmark/submission/preprints/`)
| 파일명 | 형식 | 크기 | SHA-256 체크섬 |
| :--- | :---: | :---: | :--- |
| [DLG-Benchmark_A15_Preprints.tex](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/DLG-Benchmark_A15_Preprints.tex) | LaTeX Source | 56,257 B | `db69f640e759d414bddbcc242cc0c666a4a020895405506c0a251ba78d3da7d7` |
| [DLG-Benchmark_A15_Preprints.pdf](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/DLG-Benchmark_A15_Preprints.pdf) | PDF (18p) | 564,060 B | `243af5be49cfc219141b57ce0b51b8b41573ffc76968e4192805acc17fa415ec` |
| [source_bundle.zip](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/source_bundle.zip) | 압축 아카이브 | 106,590 B | `85b47f0c51534980979322540252147b381940ddad8f709922a6be975a9167ce` |
| [DATA_AVAILABILITY.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/DATA_AVAILABILITY.md) | 정책 문서 | 2,185 B | `d017a21f6e16470d2d55fa450a56b37c7057eee6dfb913a733ebe4d633ee4883` |
| [PREPRINT_GATE_REPORT_A15.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/PREPRINT_GATE_REPORT_A15.md) | 게이트 보고서 | 2,642 B | `0e1df8575e0adac2392fb39a9fa1dad2c56e189ad329f66dd511065be2e2ca7f` |

#### MDPI Applied Sciences 제출 패키지 (`projects/benchmark/submission/applsci/`)
| 파일명 | 형식 | 크기 | SHA-256 체크섬 |
| :--- | :---: | :---: | :--- |
| [DLG-Benchmark_A15_AppliedSciences.tex](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/DLG-Benchmark_A15_AppliedSciences.tex) | LaTeX Source | 55,880 B | `6b4570c25c59edb8ba5c1bec23a032d4e2e7633b7f1fe93557685b74d32f5bc8` |
| [DLG-Benchmark_A15_AppliedSciences.pdf](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/DLG-Benchmark_A15_AppliedSciences.pdf) | PDF (16p) | 344,720 B | `41327e131711e9e905b5fea525b460694a3c0de25cc81c30d05f75fbecc5e1b0` |
| [source_bundle.zip](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/source_bundle.zip) | 압축 아카이브 | 1,142,571 B | `e78f6141013aad1976f7ae827c7b99eb6aaefb9cf175f5ade7c577c08f15decb` |
| [COVER_LETTER.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/COVER_LETTER.md) | 커버레터 | 4,743 B | `6e6ec59f759b4bdc07afb21a8394c44e64145bfb07c14e8886436fb08982202b` |
| [DATA_AVAILABILITY.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/DATA_AVAILABILITY.md) | 정책 문서 | 1,343 B | `c3ac6c268934d0146dea0fe54f35251a8b2a3ead0704abd437a7aab2ec24cfd5` |
| [SUBMISSION_GATE_REPORT_A15.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/applsci/SUBMISSION_GATE_REPORT_A15.md) | 게이트 보고서 | 3,456 B | `fc1a477697ea70f77b970ab629f7abeef51523238bb0e0cf81d763d0146cda53` |

---

### 3. 검증 보고서 및 산출물 파일 링크

- **A15 종합 실행 보고서:** [DLG_Benchmark_A15_Execution_Report_2026-10-25.md](file:///d:/_Work/goat_bank/dlg_gnn/docs/work_reports/benchmark/240_A15/DLG_Benchmark_A15_Execution_Report_2026-10-25.md)
- **A15 최종 서명 문서:** [A15_FINAL_SIGNOFF.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/A15_FINAL_SIGNOFF.md)
- **평가 분모 및 양성률 감사 보고서:** [evidence_denominator_audit.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/evidence_denominator_audit.md)
- **최종 제출 준비 보고서:** [final_submission_readiness.md](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/final_submission_readiness.md)
- **원격 Git 식별자 검증 결과:** [remote_git_identity_verified.json](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/remote_git_identity_verified.json)
- **릴리스 매니페스트:** [release_manifest.json](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/release_manifest.json)
- **음성 대조군 테스트 로그:** [negative_controls_a15.json](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a15/negative_controls_a15.json)

---

### 4. 최종 승인 선언 (Dual Sign-Off)

1. **Preprints.org:** **`A15 PREPRINT READY`**  
   - [source_bundle.zip](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/source_bundle.zip) 및 [DLG-Benchmark_A15_Preprints.pdf](file:///d:/_Work/goat_bank/dlg_gnn/projects/benchmark/submission/preprints/DLG-Benchmark_A15_Preprints.pdf)를 Preprints.org에 업로드하여 고유한 벤치마크 Preprint DOI를 취득할 준비가 완료되었습니다.
2. **MDPI Applied Sciences Special Issue:** **`A15 APPLIED SCIENCES PACKAGE READY`**  
   - Preprint DOI 발급 직후 Special Issue `C67H3U9QGT` (*Machine Learning and Its Application for Anomaly Detection*, 마감일: 2026-12-31)로 최종 투고할 준비가 완료되었습니다.