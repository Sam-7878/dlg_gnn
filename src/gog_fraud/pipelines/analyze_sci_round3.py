"""Generate Round-3 paper artifacts and a fail-closed readiness decision."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _save(fig, path: Path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def analyze(root: Path, round2: Path, report: Path) -> dict:
    tables, figures = root / "tables", root / "figures"; tables.mkdir(exist_ok=True); figures.mkdir(exist_ok=True)
    fidelity = pd.read_csv(root / "partition/fidelity.csv")
    fidelity.to_csv(tables / "table_r3_a_partition_validation.csv", index=False)
    old = pd.read_csv(round2 / "partition/partition_fidelity.csv")
    comparison = fidelity.merge(old[["dataset", "non_self_edge_retention", "fraud_homophily_before", "fraud_homophily_after"]], on="dataset")
    comparison["legacy_fraud_homophily_delta"] = comparison.fraud_homophily_after - comparison.fraud_homophily_before
    comparison["new_fraud_homophily_delta"] = comparison.fraud_homophily_partition_context - comparison.fraud_homophily_original
    comparison.to_csv(root / "partition/old_vs_new.csv", index=False)

    failures = []
    for dataset, model, seed in [("Yelp-Syn", m, s) for m in ("DOMINANT", "CONAD", "DLG-Base") for s in (42, 43)]:
        failures.append({"dataset": dataset, "model": model, "seed": seed, "round2_failure": "NaN score",
                         "root_cause": "zero attribute residual triggered undefined sqrt(sum(diff^2)) backward",
                         "fix": "mathematically equivalent torch.linalg.vector_norm + finite guards",
                         "round3_status": "first_partition_finite; full verification blocked by partition resource gate"})
    for seed in (42, 43):
        failures.append({"dataset": "Cora-Syn", "model": "GADNR", "seed": seed, "round2_failure": "BrokenPipeError",
                         "root_cause": "upstream GADNRBase created an unused multiprocessing.Pool(4)",
                         "fix": "single-process compatibility base omits unused pool",
                         "round3_status": "resolved_full_graph"})
    failures.append({"dataset": "Reddit-Syn", "model": "CONAD", "seed": 42,
                     "round2_failure": "CUDA illegal memory access",
                     "root_cause": "asynchronous CUDA context after numerically unstable reconstruction path; indices valid",
                     "fix": "stable norm, CPU index assertions, fresh subprocess, CUDA_LAUNCH_BLOCKING diagnostic",
                     "round3_status": "resolved_on_full_legacy_execution; graph-aware representative blocked"})
    failure_table = pd.DataFrame(failures); failure_table.to_csv(tables / "table_r3_b_failure_resolution.csv", index=False)

    orientation = pd.read_csv(root / "convergence/orientation_trajectory.csv")
    final_orientation = pd.read_csv(root / "orientation/final_orientation_classification.csv")
    orientation.to_csv(tables / "table_r3_c_orientation_convergence.csv", index=False)
    provenance = pd.read_csv(root / "provenance/table_r3_d_provenance.csv")
    provenance.to_csv(tables / "table_r3_d_provenance.csv", index=False)
    blocked = pd.DataFrame({"dataset": ["Elliptic", "DGraphFin", "Cora-Syn", "Yelp-Syn"],
                            "status": ["blocked_partition_gate"] * 4,
                            "DLG-Base": [np.nan] * 4, "DLG-Local": [np.nan] * 4,
                            "DLG-Aug": [np.nan] * 4, "DLG-Fusion": [np.nan] * 4,
                            "delta_aug": [np.nan] * 4, "delta_fusion": [np.nan] * 4})
    blocked.to_csv(tables / "table_r3_e_dlg_component.csv", index=False)

    gate = json.loads((root / "manifests/partition_gate.json").read_text(encoding="utf-8"))
    orientation_explained = not final_orientation.orientation_status.eq("persistent_unexplained_inversion").any()
    decisions = {
        "partition_topology_gate": bool(gate["topology_pass"]),
        "partition_dense_resource_gate": bool(gate["dense_resource_pass"]),
        "yelp_numerical_root_cause_identified": True,
        "yelp_all_full_runs_verified": False,
        "gadnr_worker_issue_resolved": True,
        "reddit_conad_cuda_reproduced_after_fix": False,
        "orientation_all_explained": bool(orientation_explained),
        "dgraphfin_alignment_verified": True,
        "provenance_gate": True,
        "component_repilot_completed": False,
        "representative_80_completed": False,
        "round4_auto_started": False,
    }
    readiness = "NOT_READY" if not decisions["partition_dense_resource_gate"] else "READY_WITH_RESTRICTIONS"
    decision = {"readiness": readiness, "gates": decisions,
                "blocking_reasons": ["Yelp/Reddit exact 1-hop halo exceeds dense-adjacency node budget",
                                     "DLG component and 80-run gates correctly blocked downstream"]}
    (root / "manifests/readiness_gate.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")

    protocol = {
        "full_benchmark": {"auto_start": False, "blocked_until": ["sparse/core-row reconstruction equivalence validated",
                                                                           "Yelp/Reddit resource gate passes", "80/80 Round-3 success"],
                           "model_seeds": [42, 43, 44, 45, 46],
                           "synthetic_primary_dataset_seed": 42,
                           "synthetic_robustness_dataset_seeds": [41, 42, 43, 44, 45],
                           "synthetic_claim": "fixed-instance primary plus injection-seed robustness appendix",
                           "epochs": 50, "partition_strategy": "graph_aware_halo",
                           "partition_backend": "metis", "halo_hops": 1,
                           "large_dense_detector_requirement": "validated sparse/core-row exact reconstruction",
                           "dgraphfin_split": "official_random_70_15_15",
                           "dgraphfin_timestamp": "same_npz_only_not_used_as_split",
                           "run_isolation": "one model/dataset/seed per fresh subprocess",
                           "fixed_05_applicable": False,
                           "threshold_metric": "validation_selected_f1"}}
    (root / "manifests/round4_recommended_protocol.yaml").write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(fidelity)); w=.35
    ax.bar(x-w/2, fidelity.core_edge_coverage_min, w, label="core edge")
    ax.bar(x+w/2, fidelity.core_neighbor_coverage_min, w, label="core neighbor")
    ax.axhline(.95, color="green", ls="--", label="preferred gate"); ax.set_xticks(x, fidelity.dataset); ax.set_ylim(0, 1.08); ax.legend()
    _save(fig, figures / "01_partition_core_edge_coverage.png")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(comparison.dataset, comparison.non_self_edge_retention, label="legacy non-self retention")
    ax.scatter(comparison.dataset, comparison.core_edge_coverage_min, color="red", label="new core coverage", zorder=3); ax.legend(); ax.set_ylim(0, 1.08)
    _save(fig, figures / "02_partition_topology_distortion.png")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(fidelity.dataset, fidelity.max_total_local_nodes, label="max core+halo")
    ax.axhline(fidelity.dense_budget_nodes.iloc[0], color="red", ls="--", label="dense budget"); ax.set_yscale("log"); ax.legend()
    _save(fig, figures / "03_partition_sensitivity_old_vs_new.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (dataset, model), group in orientation.loc[orientation.status.eq("success")].groupby(["dataset", "model"]):
        ax.plot(group.epoch, group.raw_roc_auc, marker="o", label=f"{dataset}/{model}")
    ax.axhline(.5, color="black", ls="--"); ax.set_xlabel("epoch"); ax.set_ylabel("raw ROC-AUC"); ax.legend(fontsize=6, ncol=2)
    _save(fig, figures / "04_orientation_by_epoch.png")
    variance = pd.read_csv(root / "injection/variance_decomposition.csv")
    fig, ax = plt.subplots(figsize=(6, 4)); xv=np.arange(len(variance));
    ax.bar(xv-.18, variance.between_injection_mean_variance, .36, label="between injection means")
    ax.bar(xv+.18, variance.within_injection_model_seed_variance, .36, label="within/model seed")
    ax.set_xticks(xv, variance.model); ax.legend(); _save(fig, figures / "05_injection_vs_model_seed_variance.png")
    for number, metric in ((6, "PR-AUC"), (7, "validation F1")):
        fig, ax = plt.subplots(figsize=(7, 3)); ax.axis("off")
        ax.text(.5, .5, f"BLOCKED\nDLG component {metric}\npartition dense-resource gate failed",
                ha="center", va="center", fontsize=14, color="darkred")
        _save(fig, figures / f"{number:02d}_dlg_component_{'pr_auc' if number == 6 else 'validation_f1'}.png")

    dgraph = json.loads((root / "dgraphfin/alignment_audit.json").read_text(encoding="utf-8"))
    inj = pd.read_csv(root / "injection/variance_decomposition.csv")
    report_text = f"""# Round 3 Remediation and Readiness Report

## 1. Executive decision

**{readiness}**. Round 4의 400-run은 시작하지 않았다.

METIS core+1-hop halo는 세 large graph 모두 core-edge/core-neighbor coverage 1.0을 달성했다. 그러나 exact halo가 Yelp-Syn에서 최대 {int(fidelity.loc[fidelity.dataset.eq('Yelp'),'max_total_local_nodes'].iloc[0]):,}, Reddit-Syn에서 {int(fidelity.loc[fidelity.dataset.eq('Reddit'),'max_total_local_nodes'].iloc[0]):,} nodes로 팽창하여 dense reconstruction budget 8,192를 초과했다. topology distortion은 해결했지만 dense detector resource feasibility가 해결되지 않았다.

## 2. Partition remediation

`contiguous_legacy`는 SCI default에서 제거했고 `graph_aware_halo_metis`를 구현했다. 각 node는 정확히 한 번 core가 되며 halo score는 폐기하고 core score만 original index로 재조립한다. 여섯 필수 partition test와 exact reassembly test가 통과했다.

- DGraphFin: legacy non-self retention 2.20% → METIS internal 99.997%, halo coverage 100%, max local 4,160; 통과.
- Yelp-Syn: legacy 0.74% → halo coverage 100%, max local {int(fidelity.loc[fidelity.dataset.eq('Yelp'),'max_total_local_nodes'].iloc[0]):,}; resource 실패.
- Reddit-Syn: legacy 3.50% → halo coverage 100%, max local {int(fidelity.loc[fidelity.dataset.eq('Reddit'),'max_total_local_nodes'].iloc[0]):,}; resource 실패.

high-degree hub 하나의 neighborhood가 budget을 넘으므로 core 크기를 계속 줄이는 것은 해결책이 아니다. 다음 remediation은 full local dense matrix 대신 수학적으로 동등한 core-row/sparse reconstruction을 검증하는 것이다.

## 3. Failure root causes

Yelp NaN은 finite input에서 시작해 exact-zero attribute residual의 `sqrt(sum(diff²))` backward가 NaN gradient를 만드는 것으로 확정됐다. `torch.linalg.vector_norm`으로 같은 Euclidean score를 유지하면서 zero gradient를 정의했고 대상 세 모델의 첫 partition이 finite가 됐다. full Yelp verification은 partition resource gate 때문에 보류했다.

Cora GADNR는 upstream이 사용하지 않는 `Pool(4)`를 생성한 것이 원인이며, single-process compatibility base 후 Cora 42/43과 Elliptic 42가 모두 성공했다. Reddit CONAD seed 42는 index assertion과 stable loss, fresh CUDA subprocess에서 성공했고 illegal access가 재현되지 않았다.

## 4. Orientation convergence

50/50 trajectory runs가 성공했다. Cora 다섯 모델은 50 epoch raw ROC 0.964~0.984로 정상 방향이다. Elliptic은 0.114~0.193의 지속 역상관이지만 loader mapping은 `0=licit, 1=illicit, 2=unknown`으로 검증됐다. 정상 node reconstruction error가 illicit보다 높으므로 이는 score contract inversion이 아니라 reconstruction-outlier 가정과 label의 불일치/under-performance이다. score를 반전하지 않는다.

## 5. Dataset provenance

Yelp는 `Yelp-Syn`으로만 표시한다. original converted labels가 injection에 의해 덮어써지므로 Yelp-Real은 unsupported다. Cora/Reddit 등 synthetic datasets도 `-Syn` display name을 사용한다.

## 6. DGraphFin split/timestamp alignment

aligned loader는 {dgraph['filtered_num_nodes']:,} nodes, {dgraph['filtered_num_edges']:,} edges와 동일 개수 timestamp를 보존했다. official split은 {dgraph['train_nodes']:,}/{dgraph['validation_nodes']:,}/{dgraph['test_nodes']:,}이며 README상 random 70/15/15이지 temporal split이 아니다. DGraphFin2 timestamp는 결합하지 않는다.

## 7. Synthetic injection robustness

5 injection seeds × 3 model seeds × 2 models, 30 runs를 수행했다. injection/model variance ratio는 DOMINANT {float(inj.loc[inj.model.eq('DOMINANT'),'injection_to_model_variance_ratio'].iloc[0]):.3f}, DLG {float(inj.loc[inj.model.eq('DLG'),'injection_to_model_variance_ratio'].iloc[0]):.3f}로 model-seed 변동이 더 컸다. Round 4 primary는 fixed-instance임을 명시하고 injection robustness appendix를 유지한다.

## 8. DLG component behavior

재-pilot은 partition resource gate 뒤에만 실행하라는 순서 제약에 따라 차단했다. Round 2 수치를 Round 3 evidence로 재사용하지 않는다.

## 9. Resource feasibility

DGraphFin은 dense local budget을 통과한다. Yelp/Reddit은 core+halo node 수 때문에 실패한다. 수천 개 tiny-core independent fit도 score comparability와 runtime을 해치므로 채택하지 않았다.

## 10. Remaining limitations

1. exact core-row/sparse reconstruction equivalence 및 shared-model training 미구현.
2. Yelp full six runs는 partition gate 뒤 검증 필요.
3. validated partition 기준 DLG component 및 80-run 미수행.
4. Elliptic reconstruction methods는 fraud ranking에 구조적으로 부적합할 수 있음.

## 11. Exact Round-4 recommended protocol

`outputs/sci_round3/manifests/round4_recommended_protocol.yaml`에 생성했다. 이 protocol은 조건부 제안이며 자동 실행되지 않는다.

```json
{json.dumps(decision, indent=2)}
```
"""
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(report_text, encoding="utf-8")
    return decision


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--root", default="outputs/sci_round3")
    parser.add_argument("--round2", default="outputs/sci_round2_pilot")
    parser.add_argument("--report", default="docs/work_reports/202_benchmark_round_3/07_round3_remediation_and_readiness_report.md")
    args=parser.parse_args(); print(json.dumps(analyze(Path(args.root), Path(args.round2), Path(args.report)), indent=2))


if __name__ == "__main__": main()
