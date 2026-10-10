# A09 — 승인된 공개 게시 인계

공동저자 확인·승인: 사용자 명시 확인 기록 완료. 게시 담당: Agent. 사용자가 검증된 scientific source/protocol/환경 lock/A08 numeric ZIP/A09 companion의 commit/push 및 versioned release 게시를 명시 승인했다.

## 공개 범위

검증된 scientific source, protocol, environment lock, 재현 facade와 numeric evidence를 같은 immutable commit/tag/release에 연결한다. 두 ZIP만 올렸다면 그 숫자를 재생성하는 scientific source/config/lock 및 verify/tables revision의 연결도 확인한다. 새 raw IDs/scores/checkpoints와 미투고 TeX/Bib/PDF/private writer는 공개 범위에서 제외한다.

기존 A08의 public checkout 검증 범위와 실제 CLI는 `README_REPRODUCE.md`에 있다. 고정 scientific source identity는 `current_execution_source_recheck.json`의57파일과 execution manifest가 연결한다. .gitignore에서 curated A09 companion ZIP만 예외로 허용하며 원고 비공개 규칙은 유지한다.

## numeric package identity

| 파일 | Bytes | SHA256 |
|---|---:|---|
| projects/benchmark/evidence/public_numeric_evidence.zip | 21450806 | `b1caab1124404eb6ac4dcd909007c69ceab2410e02f84363d53b0f8465b0c7f5` |
| projects/benchmark/evidence/a08_public_numeric_evidence.zip | 993748 | `b1ae519a058784ed1d8306d2986748de6832e142209502e984030ff35b18e100` |
| projects/benchmark/evidence/a09_submission_closure_companion.zip | 566235 | `336c7bcdaa50c68582be78161ccea3c769cf214d1b71a5d327e98709d1de94f5` |

A08 ZIP은226payload +release_manifest1개이며 총227실파일이다. A09companion은187payload +index2개로189실파일이다. 승인 후 추가 기록, current outer closure manifest 및 이 handoff는 ZIP에 사후 포함됐다고 주장하지 않는 별도 문서다. ZIP bytes는 승인 전 검토본과 동일하다.

## 게시 후 검증할 정보

- 공개 scientific repository의 정확한40자리 commit SHA 및 commit/tag URL.
- Release/tag URL과 numeric ZIP의 다운로드 URL. 고정 commit의 파일 링크도 가능하다.
- 게시한 ZIP이 위 hash와 다르면 변경 이유 및 실제 배포본 hash/manifest.

게시된 자료를 새 위치에서 다운로드하여 해당 commit과 artifact hashes를 대조하고 verify/tables를 실행한다. 이후 승인/실제 공개 위치를 원고에 반영하고 두PDF와 private LaTeX source ZIP을 재생성·검토해 C5/C6를 마감한다. 아직 agent에 플랫폼 업로드 또는 실제 submit 수행을 요청한 것은 아니다.

## 게시 및 독립 검증 완료

공개 release: https://github.com/Sam-7878/dlg_gnn/releases/tag/benchmark-a09-2026-10-10

Scientific commit: 2cc6f85afc44830eb7eee92d709c6eb8d567ba2d

GitHub immutable=true이며 인증 없는 artifact 다운로드와 tagged checkout의 verify/tables가 실제 PASS이다. 현재 public_retrieval_verification.json이 그 별도 원기록이다. Frozen ZIP을 다시 작성하지 않았다. 최종 private PDF/LaTeX 제출 패키지는 별도 C6 검토를 마친다. 플랫폼 실제 투고는 실행하지 않았다.
