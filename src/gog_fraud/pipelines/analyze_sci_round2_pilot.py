"""Aggregate Round-2 pilot artifacts and issue a conservative readiness gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _fmt(value: float, digits: int = 4) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"


def analyze(output_root: Path, report_path: Path) -> dict:
    tables = output_root / "tables"; tables.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(output_root / "multiseed/raw_results.csv")
    raw = _numeric(raw, ["roc_auc", "pr_auc", "validation_f1", "topk_f1", "validation_threshold",
                         "validation_threshold_percentile", "score_min_test", "score_max_test",
                         "train_time_sec", "inference_time_sec", "peak_ram_mb", "peak_vram_mb",
                         "validation_positive", "validation_negative", "test_positive", "test_negative",
                         "pr_gain_ratio", "pr_lift"])
    success = raw.loc[raw.status.eq("success")].copy()

    integrity = pd.read_csv(tables / "table_r2_a_dataset_integrity_preflight.csv")
    support = success.groupby("dataset", as_index=False).agg(
        validation_positive=("validation_positive", "min"), validation_negative=("validation_negative", "min"),
        test_positive=("test_positive", "min"), test_negative=("test_negative", "min"),
        threshold_unstable_warning=("threshold_unstable_warning", "max"),
        metric_low_support_warning=("metric_low_support_warning", "max"))
    fidelity = pd.read_csv(output_root / "partition/partition_fidelity.csv")
    table_a = integrity.merge(support, on="dataset", how="left")
    table_a["partitioned"] = table_a.dataset.isin(fidelity.dataset)
    table_a = table_a.merge(fidelity[["dataset", "partition_size", "num_partitions", "edge_retention",
                                      "non_self_edge_retention", "cross_partition_edges"]], on="dataset", how="left")
    table_a.to_csv(tables / "table_r2_a_dataset_integrity.csv", index=False)

    contracts = pd.read_csv(output_root / "score_semantics/detector_contracts.csv")
    observed = success.groupby("model", as_index=False).agg(
        score_min=("score_min_test", "min"), score_max=("score_max_test", "max"),
        successful_runs=("status", "size"), orientation_warnings=("orientation_warning", "sum"))
    table_b = contracts.merge(observed, left_on="paper_name", right_on="model", how="left")
    table_b.to_csv(tables / "table_r2_b_detector_score_semantics.csv", index=False)

    table_c = success.groupby(["dataset", "model"], as_index=False).agg(
        successful_seeds=("seed", "nunique"), roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        validation_f1_mean=("validation_f1", "mean"), validation_f1_std=("validation_f1", "std"),
        topk_f1_mean=("topk_f1", "mean"), pr_gain_ratio_mean=("pr_gain_ratio", "mean"),
        pr_lift_mean=("pr_lift", "mean"))
    table_c.to_csv(tables / "table_r2_c_representative_performance.csv", index=False)

    contribution = pd.read_csv(output_root / "ablation/stage_contribution_matrix.csv")
    contribution.to_csv(tables / "table_r2_d_dlg_stage_contribution.csv", index=False)
    runtime = success.groupby("dataset", as_index=False).agg(
        runtime_sec=("train_time_sec", "sum"), peak_ram_mb=("peak_ram_mb", "max"), peak_vram_mb=("peak_vram_mb", "max"))
    table_e = fidelity.merge(runtime, on="dataset", how="left")
    table_e.to_csv(tables / "table_r2_e_partition_fidelity.csv", index=False)

    thresholds = success.groupby(["dataset", "model"], as_index=False).agg(
        validation_threshold_mean=("validation_threshold", "mean"),
        validation_threshold_std=("validation_threshold", "std"),
        validation_threshold_percentile_mean=("validation_threshold_percentile", "mean"),
        validation_threshold_percentile_std=("validation_threshold_percentile", "std"))
    thresholds["validation_threshold_cv"] = thresholds.validation_threshold_std / thresholds.validation_threshold_mean.abs().replace(0, np.nan)
    thresholds.to_csv(tables / "validation_threshold_stability.csv", index=False)

    matrix = raw[["dataset", "model", "seed", "status", "error_type", "error_message"]].sort_values(["dataset", "model", "seed"])
    matrix.to_csv(tables / "pilot_success_failure_matrix.csv", index=False)
    resources = success.groupby("model", as_index=False).agg(
        successful_runs=("status", "size"), train_time_sec_sum=("train_time_sec", "sum"),
        train_time_sec_median=("train_time_sec", "median"), inference_time_sec_sum=("inference_time_sec", "sum"),
        process_peak_rss_mb=("peak_ram_mb", "max"), cuda_peak_allocated_mb=("peak_vram_mb", "max"))
    resources.to_csv(output_root / "resources/model_resource_summary.csv", index=False)

    partition_ok = bool((fidelity.non_self_edge_retention >= 0.80).all())
    attempt_ok = len(raw) == 80 and raw[["dataset", "model", "seed"]].drop_duplicates().shape[0] == 80
    all_success = bool(raw.status.eq("success").all())
    orientation_ok = not bool(success.orientation_warning.fillna(False).astype(bool).any())
    fixed_ok = bool(success.fixed_05_applicable.fillna(False).eq(False).all() and success.f1_at_05.isna().all())
    gpu = json.loads((output_root / "resources/gpu_readiness.json").read_text(encoding="utf-8"))
    gates = {
        "architecture_identity_documented": True,
        "score_semantics_documented": True,
        "fixed_05_excluded_for_unbounded_scores": fixed_ok,
        "representative_matrix_attempted_80_of_80": attempt_ok,
        "all_pilot_runs_successful": all_success,
        "score_orientation_clear": orientation_ok,
        "partition_non_self_edge_retention_at_least_0_80": partition_ok,
        "cuda_usable": bool(gpu.get("cuda_usable")),
        "component_pilot_has_all_32_successes": int(pd.read_csv(output_root / "ablation/ablation_raw.csv").status.eq("success").sum()) == 32,
    }
    critical_failures = [name for name in ("partition_non_self_edge_retention_at_least_0_80", "all_pilot_runs_successful", "score_orientation_clear") if not gates[name]]
    readiness = "NOT_READY" if critical_failures else ("READY_WITH_FIXES" if not all(gates.values()) else "READY_FOR_FULL_RUN")
    decision = {"readiness": readiness, "gates": gates, "critical_failures": critical_failures,
                "pilot_runs": len(raw), "successful_runs": int(raw.status.eq("success").sum()),
                "failed_runs": int(raw.status.ne("success").sum())}
    (output_root / "manifests/readiness_gate.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    total_compute = float(success.train_time_sec.sum() + success.inference_time_sec.sum())
    same_epoch_400 = total_compute * 5.0
    paper_50_epoch_rough = same_epoch_400 * 50.0
    failure_lines = "\n".join(
        f"- {r.dataset}/{r.model}/seed {int(r.seed)}: `{r.error_type}` — {str(r.error_message).splitlines()[0]}"
        for r in raw.loc[raw.status.ne("success")].itertuples())
    partition_lines = "\n".join(
        f"- {r.dataset}: all-edge {_fmt(r.edge_retention)}, non-self {_fmt(r.non_self_edge_retention)}, "
        f"cross-edge {_fmt(r.cross_partition_ratio)}, components {int(r.connected_components_before):,}→{int(r.connected_components_after):,}"
        for r in fidelity.itertuples())
    stage_lines = "\n".join(
        f"- {r.dataset}: global={_fmt(getattr(r, 'global_only_pr_auc'))}, local={_fmt(getattr(r, 'local_only_pr_auc'))}, "
        f"augmented={_fmt(getattr(r, 'local_augmented_global_pr_auc'))}, fusion={_fmt(getattr(r, 'local_global_fusion_pr_auc'))}"
        for r in contribution.itertuples())
    report = f"""# SCI Benchmark Round 2 — Representative Pilot Report

## 결론

최종 readiness는 **{readiness}** 이다. 400-run Round 3는 시작하지 않는다. DLG benchmark 정체성과 score semantics는 확정했지만, contiguous partition이 non-self topology를 거의 보존하지 못하고, 80개 execution pilot 중 9개가 실패했으며, 여러 dataset/model에서 score-orientation warning이 관측됐다.

이 pilot은 `epochs=1` validity/resource run이다. 아래 ROC-AUC·PR-AUC는 수렴 성능이나 우월성 claim에 사용할 수 없다.

## 실행 범위와 성공성

- 범위: 5 datasets × 8 models × 2 seeds = {len(raw)} unique attempts
- 성공: {int(raw.status.eq('success').sum())}; 실패: {int(raw.status.ne('success').sum())}
- CUDA: {gpu.get('gpu_model')} / torch {gpu.get('torch_version')} / CUDA {gpu.get('cuda_version')} / usable={gpu.get('cuda_usable')}
- 전체 성공 run의 기록된 train+inference 합: {total_compute/3600:.2f} GPU-hours (dataset load와 실패 run 시간 제외)
- 동일한 1 epoch 조건의 10×8×5 단순 하한 추정: {same_epoch_400/3600:.2f} GPU-hours
- 50 epoch 선형 외삽 참고값: {paper_50_epoch_rough/3600:.1f} GPU-hours. loader, failure, nonlinear overhead를 제외하므로 scheduling 근거가 아닌 경고용 수치다.

### 실패 matrix

{failure_lines}

## Architecture identity와 명칭

Historical `DLG`는 `DLGFull`이며 local GCN feature autoencoder를 pretrain한 뒤 frozen `H_local`을 `X`에 연결하여 global reconstruction detector를 학습한다. 최종 historical score는 global augmented reconstruction score이며 score fusion이 아니다. system architecture report의 transaction-subgraph Level-1 → relation/meta-graph GATv2 Level-2 → Fusion과 동일한 모델로 서술하면 안 된다.

Canonical variants는 `global_only`, `local_only`, `local_augmented_global`, `local_global_fusion`이고 historical aliases는 각각 `DLG-Base`, `DLG-Local`, `DLG`/`DLG-Aug`, `DLG-Fusion`이다.

## Score와 threshold validity

8 detector 모두 unbounded, non-probability anomaly score다. 관측 test score 전체 범위는 {float(success.score_min_test.min()):.4g}~{float(success.score_max_test.max()):.4g}이며 model별 범위는 Table R2-B에 기록했다. 따라서 `F1@0.5`는 전부 N/A이고, primary threshold-dependent metric은 validation-selected F1이다. ROC-AUC/PR-AUC는 calibration하지 않은 raw ranking score를 사용한다.

orientation audit은 silent inversion을 하지 않았다. warning은 {int(success.orientation_warning.fillna(False).astype(bool).sum())}/{len(success)} successful runs이며 특히 1-epoch Elliptic reconstruction 계열에서 체계적 역방향 ranking이 관측됐다. 이는 score contract를 자동 반전할 근거가 아니라 under-training, label semantics, loss behavior를 Round 3 전에 재검증할 근거다.

## Partition fidelity

{partition_lines}

self-loop를 포함한 retention은 왜곡될 수 있다. 예를 들어 DGraphFin all-edge retention은 0.6299지만 non-self retention은 0.0220이다. 이 partition 결과를 원 graph benchmark와 동등하다고 간주할 수 없다.

Reddit partition sensitivity에서 2,048→4,096→8,192로 바꿀 때 DOMINANT PR-AUC는 0.0760→0.0515→0.0388, DLG는 0.0926→0.0665→0.0432로 변했다. 현재 결과는 graph/model 특성보다 partition artifact를 강하게 포함한다.

## DLG stage contribution

{stage_lines}

Cora에서는 local-only가 augmented global보다 강하고 fusion이 상당 부분 회복했다. Elliptic에서는 augmentation이 소폭 개선됐고 DGraphFin에서는 차이가 작다. 따라서 “local augmentation/fusion이 항상 개선한다”는 claim은 지원되지 않는다. Yelp global-only NaN 때문에 해당 delta 일부는 N/A다.

## Synthetic injection sensitivity

Cora DLG의 dataset-seed별 PR-AUC 평균은 seed 41={_fmt(float(pd.read_csv(output_root / 'injection/injection_seed_summary.csv').iloc[0].pr_auc_mean))}, seed 42={_fmt(float(pd.read_csv(output_root / 'injection/injection_seed_summary.csv').iloc[1].pr_auc_mean))}, seed 43={_fmt(float(pd.read_csv(output_root / 'injection/injection_seed_summary.csv').iloc[2].pr_auc_mean))}. dataset/label/edge hash가 seed마다 달라져 injection variation과 model-seed variation을 분리해야 한다.

## Round 3 전에 필요한 fixes

1. contiguous node-ID partition을 graph-aware partition 또는 overlap/edge-aware inference로 교체하고 node-score reassembly와 topology fidelity를 다시 통과시킨다.
2. Yelp NaN(DOMINANT, CONAD, DLG-Base)의 최초 발생 partition/score/loss를 finite assertion으로 특정한다.
3. Cora GADNR broken pipe와 Reddit CONAD CUDA illegal memory access를 재현·격리한다.
4. converged small/medium-dataset run으로 orientation warning이 under-training 때문인지 label/score contract 문제인지 판별한다.
5. graph-aware partition 적용 뒤 component 및 partition sensitivity pilot을 재실행한다.

## 산출 표

- `table_r2_a_dataset_integrity.csv`
- `table_r2_b_detector_score_semantics.csv`
- `table_r2_c_representative_performance.csv`
- `table_r2_d_dlg_stage_contribution.csv`
- `table_r2_e_partition_fidelity.csv`
- `validation_threshold_stability.csv`
- `pilot_success_failure_matrix.csv`

## Readiness gate

```json
{json.dumps(decision, indent=2)}
```
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="outputs/sci_round2_pilot")
    parser.add_argument("--report", default="docs/work_reports/201_benchmark_round_2/05_round2_representative_pilot_report.md")
    args = parser.parse_args()
    decision = analyze(Path(args.output_root), Path(args.report))
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
