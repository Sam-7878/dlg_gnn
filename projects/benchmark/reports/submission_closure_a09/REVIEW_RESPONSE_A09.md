# A09 — Review Response and Submission Closure

**상태: 로컬 증거 검증·편집·PDF 검토 완료, C5 승인/공개 identity 대기. `SUBMISSION_READY=false`.**

2026-10-10. A09 작업지시서 C1–C6에 대한 실제 수행 기록이다. 신규 production training은 0회이며 A08 execution revision2의 105 corrected crypto 실행, 350 historical success 및 1 historical failure의 원래 등급을 유지한다. 초기 50success/1failure, qualification, 과거 label-informed crypto는 현재 결과 수에 더하지 않았다.

| 항목 | 판정 | 실제 조치와 증거 |
|---|---|---|
| C1 | PASS_LOCAL | 105 raw-score metric JSON 정확 일치, 351 historical member 해시, full-precision 표·200000permutation·전체18Holm byte-identical replay, 새 public checkout verify/tables exit0. 기존226payload ZIP+189file companion 인계. |
| C2 | PASS_TECHNICAL | 3chain actual input/X/E/target/split 및35run final map, providerREADME hash 일치/Category0 mapping, 14dataset origins/injection source. 원천·공개 정책의 최종 저자 확인은 C5. |
| C3 | PASS_LOCAL | 기존 absolute/허용오차 기록 보존; 누락 상대오차·mean loss는 동일 tiny fixtures만 회수. 7미지원 셀을 실제 OOM/원인 미기록으로 구분. |
| C4 | PASS | 기존 primary 값 보존; 15test-support rows/no-skill reference, 조건부 효과·경쟁모델 우세·낮은 절대품질·비우월성 설명, 개발 이력 부록 이동. |
| C5 | PENDING | 공동저자 승인, 공개 허가와 실제 immutable evidence URL/commit/release 및 다운로드 재검증이 아직 없다. Reviewer access도 저자 결정 대기. |
| C6 | PASS_LOCAL | neutral26page+MDPI25page, source-resolved1394display, 모든51rendered page 시각 검토, 두 ZIP clean compile/PDF byte-identical. C5 완료 후 availability/approval 문구 재생성·해시/검토 갱신 필요. |

## C1 — 실파일 재검산과 인계 범위

- `cpu_metric_recheck.json`: private node_ids/labels/val/test 배열은 frozen 입력과 일치; 105score 및 원 metric SHA 일치. `benchmark_metrics.evaluate_scores`로 얻은 전체 metric dict가 원본과 정확히 같다. Validation grouped-score F1 및 largest-threshold tie, test `>=`, ceil alert counts/stable-ID ties 포함.
- 공개 scalar105JSON에서는 `selected_node_ids`만 제거한다. 원본 private metric SHA를 남기고 threshold/confusion/precision/recall/F1/support/alert counts와 지표는 그대로 보존한다. Private raw-score 재현과 public scalar/table replay를 혼동하지 않는다.
- `historical_identity_recheck.json`: 보존 ZIP 안351실파일의 SHA를 독립 확인(350success/1failure). 과거에 없는 raw score/threshold/환경 증거를 새 등급으로 승격하지 않는다.
- `statistics_replay_check.json` 및 `replayed/`: NumPy default_rng(20261008)의 원 stream, B200000, batch2000, average ties, tie correction, >=statistic−1e−12, plus-one 사용. S1–S4표/전체18pairwise family/원 p·exceedance records를 byte-identical 재생성했다.
- `actual_reproduction/facade_checks.json`: 새로운 로컬 public checkout의 실제 verify/tables exit0. 외부 공개 release를 내려받은 검증으로 표시하지 않는다.

실행 명령:

```bash
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_collect_evidence.py
/mnt/d/_work/goat_bank/.venv_cuda/bin/python projects/benchmark/scripts/a09_summarize_evidence.py
# cwd: local_storage/benchmark/a09_submission_closure/clean_public_checkout
PYTHON=/mnt/d/_work/goat_bank/.venv_cuda/bin/python bash projects/benchmark/reproduce.sh verify --revision a08
PYTHON=/mnt/d/_work/goat_bank/.venv_cuda/bin/python bash projects/benchmark/reproduce.sh tables --revision a08
```

기존 A08 ZIP은 **226payload + release_manifest.json 1개 = 227ZIP실파일**이다. A09의 `existing_226_payload_inventory.json`는 manifest까지 실제227파일을 기록하며 기존 ZIP bytes/hash는 변경하지 않았다. 226라는 기존 payload 표현을 총 ZIP파일 개수로 잘못 쓰지 않는다.

## C2 — provenance와 stale 상태 마감

`impact_map_final.csv`의 Ethereum/BSC/Polygon은 실제 frozen input-manifest hash, X/E/label/split selection hash, 체인별35승인run 목록/selection SHA를 연결한다. 옛 `PENDING_A08_BUILD` map과 “execution in progress” audit heading은 과거 계획/기록으로 보존하고 `HISTORICAL_STATUS_NOTE.md`에서 현재 final map을 안내한다.

원고 Data Origins에는 Luo et al. NeurIPS2024 GoG, Weber et al. Elliptic, Huang et al. DGraph, SNAP BitcoinOTC, Kent LANL을 정식 인용한다. GoG local README SHA는 frozen providerREADME SHA와 일치하고 Category0=fraud를 명시한다. Raw token transfers/원천 global graph와 우리의8observables exactk5 feature-similarity graph를 구분한다.

`dataset_origins.csv`는14input의 원 raw/tensor identity·실제 변환·split·injection recipe를 연결한다. 8injected targets는 native fraud labels가 아니며 정확한 Npositive를 injection ratios에서 추정하지 않는다. 보존 git object loader의 raw CRLF와 LF-normalized SHA를 구분하고 canonical release-builder SHA를 loader SHA로 오인하지 않는다.

원천 LICENSE 관측은 CC BY-NC-SA4.0이다. 이는 새 배포 허가나 법적 데이터 제한의 증명이 아니다. 비공개 mappings/scores/manuscript는 저자의 공개 정책이며 reviewer 전달 범위는 C5에서 실제 결정한다.

## C3 — 수치·미지원 셀

`numerical_qualification_summary.csv`/original JSON은 기존 maximum absolute 기록을 유지한다. 기존 CUDA에 없던 tensor-relative 값을 만들어 채우지 않았다. Historical CPU 상대오차 max|a−b|/(|a|+1e−12)는 CUDA와 별개다.

`qualification_recovery.json`/CSV는 같은 frozen tiny fixtures의 실제 CPU/RTX3090 forward/loss/score/gradients/oneAdamupdate 및 Gram/GCN/row-block 측정을 회수한다. 원 comparison·atol/rtol·sampling·deterministic/CUBLAS 조건은 변경하지 않았다. Model float32 allclose atol1e−5/rtol1e−4, scalar float64 atol1e−7/rtol1e−8; floor-relative는 max|a−b|/max(|b|,atol/rtol). 51per-quantity rows 및12finite original scalar parameterizations가 있다. Maximum tolerance fraction0.01885034888982773<1. Original16arithmetic tests와 혼동하지 않는다. Detached neighbor gradients가0이라는 사실만으로 전체 모델 동등성을 주장하지 않는다.

7missing cells: Yelp/Reddit AnomalyDAE는 RTX309024GiB에서 관측된 encoder preflight OOM; FlickrGADNR는 Round5seed42/0epoch의 실제 OOM(당시 CUDA8GiB 보고); Elliptic/DGraphFin/Yelp/RedditGADNR는 승인run이 없고 원인은 미기록이다. 이를 OOM으로 추정하거나 현재 GPU에서 불가능하다고 확대하지 않았다. 과거 오류 로그의 비정상 WSL allocated telemetry는 유효한 memory 측정으로 채택하지 않았다.

최신 measured timing은 현재 corrected inputs/fresh invocation. 이전 DGraphFin resumed segment를 총 wall time으로 쓰지 않았고, allocated VRAM과 process/driver total을 구분하며 matched AnomalyDAE8/24GiB recovery 주장은 복원하지 않았다.

## C4 — 결과와 원고 수정

ETH Base/Aug AP0.4106/0.4359(+0.0253), BSC0.3390/0.3411(+0.0021), Polygon0.0854/0.0731(−0.0124). 경쟁모델은 각 체인에서 더 높은 AP mean. Actual test prevalence와 always-positiveF1은15실제 mask에서 산출하며 trained baseline이나 유한 표본 randomAP 기대값의 정확한 공식으로 주장하지 않는다. Reddit/LANL Aug 열세, S2DOMINANT/Base Holm1.0000, S3MCp0.193584와 비동등성/비우월성을 설명한다.

Abstract/Introduction의 긴 개발 이력은 provenance appendix로 이동했다. Same-source correction, feature/topology/population 동시 변화, old affectedcrypto 제외, historical/current환경·threshold tier 차이는 Methods에 유지했다. Conclusions/author statements 분리, Comparison/N 열 정리, doubled Proceedings 제거, Günnemann 움라우트 교정,20local current/historical범위 및 global receptivefield 의미를 수정했다.

## C6 — 목적별 PDF와 소스

공식 MDPI_template.zip을2026-10-10실제 다운로드했다. 클래스 header는18June2025이며 “2026class”라고 허위 표기하지 않는다. 원 class bytes는 보존하고 preamble에서 미배정 DOI·출판정보를 제거/명시했다. MDPI logo PDF는 공식 EPS를 실제 변환했다. Neutral은 journal/publisher logo·명칭 header 없이 article 형식이다. 클래스/서지/figure dependency를 포함한 private ZIP으로 clean compile했고 두 PDF가 검토본과 byte-identical이다.

최종 PDF:

| 파일 | Pages | SHA256 |
|---|---:|---|
| DLG-Benchmark_A09.pdf | 26 | `83fe56816d1c1c312e24ad5b8f14e1b258df319c958152abe0ef2ffec7b84923` |
| DLG-Benchmark_A09_MDPI.pdf | 25 | `4c034589ef8460bb6708f24ae3794ace66fd09e55049d34a2637eaaa80d0dd01` |

`Final_PDF_Review_A09.md`/`pdf_review.json`은51페이지별 관찰과 실제PNG/PDFSHA를 보유한다. 25neutral페이지는 이미 수동 확인한 render와 최종pixel이 동일함을 확인했고 수정된25쪽은 다시 열었다. MDPI최종25페이지는 모두 열어 검토했다. 숫자1394개는 원 source-resolution과 실제PDF표시/페이지를 대조했으며 text검색만으로 검사하지 않았다. Original570display perPDF는 유지된다.

## 실제 인계 파일 identity

| 파일 | Bytes | SHA256 |
|---|---:|---|
| projects/benchmark/evidence/public_numeric_evidence.zip | 21450806 | `b1caab1124404eb6ac4dcd909007c69ceab2410e02f84363d53b0f8465b0c7f5` |
| projects/benchmark/evidence/a08_public_numeric_evidence.zip | 993748 | `b1ae519a058784ed1d8306d2986748de6832e142209502e984030ff35b18e100` |
| projects/benchmark/evidence/a09_submission_closure_companion.zip | 566235 | `336c7bcdaa50c68582be78161ccea3c769cf214d1b71a5d327e98709d1de94f5` |
| local_storage/benchmark/a09_submission_closure/submission_packages/DLG-Benchmark_A09_LaTeX_submission.zip | 31762 | `b8a725cc321cd7b9be6105832434ba1a85ec77ac891069b4393fc9dc3794a055` |
| local_storage/benchmark/a09_submission_closure/submission_packages/DLG-Benchmark_A09_MDPI_LaTeX_submission.zip | 1068727 | `f01f5c17cbdb6f6fa0c4eff9118ade830c80412a819df253cfa6bf75b220ecbc` |

A09companion은189실파일이다. `evidence_inventory.csv`/`companion_payload_manifest.json`는187payload의 상대경로·bytes·SHA를 기록하며 그 두 index도 ZIP에 들어 있다. **이 response와 outer closure manifest는 별도 인계 문서이며 ZIP 안에 재귀적으로 포함되지 않는다.** 기존 ZIP에 포함된 과학 source/config/locks와 repo facades의 범위는 README에서 구분한다.

Public/private검사는 새 contract IDs/scores/arrays/checkpoints/privatewriter/TeX/Bib/PDF가 companion에 없는지 확인했다. 두private submissionZIP와 원고/writer는 gitignore로 계속 제외하며 curatedcompanionZIP만 exception으로 추가했다. Commit/push/index추가/release/논문upload/투고를 수행하지 않았다.

## 보존 및 승인 경계

374originalA08files 및57actualexecution-source hashes(설치된PyGOD namespace 포함)가 일치했다. OriginalA08acceptanceSHA: `8a6d786034a29d18de0c8042a28950d5f244d533dbb9d84ef9bd3ad57f1f1f58`. OriginalA08PDF/source/초기 실패/ZIP/G0–10은 그대로 남아 있다. RTX3090 연결은 수치 fixture 실행 허용이며 공동저자·출판 승인이 아니다.

**C5에서 필요한 실제 결정:** 공동저자 과학적 결과·원천/Category0·old-result제외·최종 원고·기여·지원금·이해상충·AI표기·공개 범위·reviewer access·Preprints/MDPI 투고 동의. 이후 scientificsource/numericevidence를 저자가 직접 게시할지 또는 agentcommit/push/versionedrelease를 허가할지 결정한다. 실제 공개immutableURL/commit 및 다운로드hash/verify/tables가 확인된 뒤 availability와 pending문구를 사실에 맞게 업데이트하고 PDF/sourceZIP/숫자/해시/페이지 검토를 갱신한다.

이번 작업에서 새 대규모 학습·추가seed·신규baseline·Stream/TDS/LLM 확장·GPU구매는 수행하지 않았다. C5까지 닫힌 후 추가 연구 없이 제출 마감한다. 게재 판단은 편집부/심사자에게 있다.

## 저자 승인 후 추가 기록 — 2026-10-10T14:44:32Z

사용자가 “공동저자 확인 완료, 승인합니다”라고 명시적으로 답했다. 검토된 두PDF와 본 보고서의 수정 결과·출처/매핑·원고·기여/지원금/이해상충/AI·공개/심사 접근·Preprints/MDPI 동의에 대한 사용자 확인을 author-local 승인 기록에 보존했다. 공동저자의 별도 서명을 직접 수신했다고 표현하지 않는다.

사용자는 “제가 직접 게시하고 URL/commit을 전달하겠습니다”라고 선택했다. 따라서 agent는 commit/push/release하지 않는다. C5의 승인 부분은 사용자 확인으로 충족되었고, 실제 author-published evidence identity와 다운로드 검증은 대기 중이다. 이전 본문에서의 pending 표현은 승인 전 검사 시점의 상태다. `PUBLICATION_HANDOFF_A09.md`가 현재 게시 인계 범위와 예상 hashes를 안내한다.

기존 reviewed PDF/소스ZIP/companion bytes는 이번 승인 기록으로 변경하지 않는다. 실제 URL/commit 수신 후 Data Availability와 approval wording을 갱신하고 두PDF/소스ZIP/해시/최종검토를 다시 연결한다. `submission_ready=false`는 이 남은 공개·재검증 단계 때문에 유지한다.
