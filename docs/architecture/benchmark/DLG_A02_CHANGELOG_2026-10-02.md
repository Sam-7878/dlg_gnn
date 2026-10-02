# A02 변경 기록 및 영향 범위

**Date:** 2026-10-02  
**Base:** A01 work order + historical FROZEN protocol  
**Deliverable:** A02 complete work order, versioned amendment, consolidated effective protocol, task porting profiles.

## 주요 변경

| ID | A01/원 FROZEN 상태 | A02 변경 | 실행 영향 |
|---|---|---|---|
| CH01 | legacy freeze만 보관 | archive/hash/original path → `.venv_old` rename → `.venv_cuda` 신규 생성 | legacy in-place upgrade 방지 |
| CH02 | `~/.venvs/dlg-bench-v2-cuda` | 저장소별 `.venv_cuda`, 미래 `.venv_rocm` | IDE/worker/CI interpreter 경로 갱신 |
| CH03 | environment 이식 위험 미기술 | venv absolute shebang/activation 비이식성, 복원·재생성 구분 | rename 후 legacy 실행 가능성 과장 방지 |
| CH04 | CUDA/ROCm version 나열 | common constraints + full task/backend hash locks + compiler/ABI manifest | 공통 API와 binary 차이 분리 |
| CH05 | original dual-backend/server plan | 현재4070 Laptop+3090 Gaming Box eGPU, 실제 host probe | AMD mandatory gate/가상 server specs 제거 |
| CH06 | 3090 도착 이후 | 이미 보유한3090에서 qualification 가능 | correctness dependency는 유지 |
| CH07 | backend portability 실험 확대 가능 | 현재 CUDA-first, 미래 ROCm job-level workers | Benchmark 종료 지연 방지 |
| CH08 | paper별 평가 코드 혼재 위험 | 얇은 research_core + task adapters/profiles | schema/runner/검증/표 생성 재사용 |
| CH09 | 동일 stack→동일 protocol로 오해 가능 | Benchmark/Stream/TDS unit/seed/split/calibration/MC/statistics 분리 | task 의미 보존 |
| CH10 | Stream/TDS total24,316만 같음 | 서로 다른 split/positive count 및 unit의 lineage gate | label/hash/checkpoint 오용 방지 |
| CH11 | Base-70를 hidden-size control로 표현 | 첨부 LaTeX의70-global-epoch control로 정정 | 잘못된 control 설정 시 해당 cell rerun |
| CH12 | PR-AUC/validation F1 의미 불충분 | actual metric implementation ID, test F1 at validation-selected threshold | AP 정의/selection population 혼동 방지 |
| CH13 | allocator cap과 physical envelope 혼동 위험 |8 GiB bytes/fraction 및 cap 한계, physical4070 별도 | 순수 memory/속도 우열 과장 방지 |
| CH14 | 단순 per-run logs | job/attempt/evidence lane/ordered IDs/checkpoint/calibration/atomic outputs | 중복·부분 결과·task 혼입 차단 |
| CH15 | journal-ready 종료 | release freeze → Preprints.org → Applied Sciences package | 두 publication stage와 DOI 구분 |

## 보존한 핵심

13 primary + LANL,8 detectors,5 model seeds42–46, original50 epochs/Aug20+50, CONAD correctness gate, DLG exactness/ablation, selected memory tests, real6/synthetic7 분리, support-aware 통계 및 stop rule을 유지했다. 초기560은 theoretical production matrix의 상한이며 diagnostics/ablation은 별도다. unsupported cell을 강제로 성공시키기 위한 대규모 partition/HAEE나 ROCm 연구를 추가하지 않는다.

## 명시적으로 확인되지 않은 것

별도 A01 focus amendment와 Grok review 원문, 실제 repository/model source, dataset artifacts, checkpoints와 raw scores는 제공되지 않았다. 문서에 적힌 interface·script·manifest 이름은 구현 지시이며 존재/PASS를 보증하지 않는다. 현재 GPU installation, workload execution 및 future AMD support는 이번 문서 편집 중 검증한 것이 아니다.

Stream/TDS의 원고상 차이는 audit 대상으로 기록했으며, 어느 쪽이 맞는지 임의로 결정하거나 PDF 숫자를 수정하지 않았다. TDS input dimension/temporal memory/Monte Carlo randomization 용어는 code 검증 후 정정 여부를 판단하도록 했다.

## 실행 전 연구책임자 승인

A02 adoption record에 prior v2 results inspected 여부, 변경 이유, affected runs와 rerun 범위를 적는다. 미확인 값을 false로 바꾸지 않는다. 원 FROZEN을 동일 파일명으로 덮어써 역사적 freeze 날짜를 바꾸지 않는다.
