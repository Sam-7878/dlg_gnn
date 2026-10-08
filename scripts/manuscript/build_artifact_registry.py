"""
Build authoritative manuscript artifact registry with SHA-256 hashes and roles.
"""

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

AUTHORITATIVE_ARTIFACTS = [
    {
        "artifact_name": "round5_benchmark_raw",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/raw/benchmark_raw.csv",
        "role": "primary_frozen_benchmark_raw_runs",
        "authoritative": True,
        "derived_from": None,
    },
    {
        "artifact_name": "round5_support_matrix",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv",
        "role": "primary_frozen_support_matrix",
        "authoritative": True,
        "derived_from": None,
    },
    {
        "artifact_name": "round5_seed_aggregated_performance",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/summary/seed_aggregated_performance.csv",
        "role": "seed_first_aggregated_metrics",
        "authoritative": True,
        "derived_from": "round5_benchmark_raw",
    },
    {
        "artifact_name": "round5_rankings",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/rankings.csv",
        "role": "complete_case_average_rankings",
        "authoritative": True,
        "derived_from": "round5_seed_aggregated_performance",
    },
    {
        "artifact_name": "round5_friedman_tests",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/friedman_tests.csv",
        "role": "omnibus_friedman_tests",
        "authoritative": True,
        "derived_from": "round5_rankings",
    },
    {
        "artifact_name": "round5_wilcoxon_holm",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/wilcoxon_holm.csv",
        "role": "pairwise_wilcoxon_signed_rank_holm_adjusted",
        "authoritative": True,
        "derived_from": "round5_seed_aggregated_performance",
    },
    {
        "artifact_name": "round5_dataset_characteristics",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/tables/01_dataset_characteristics.csv",
        "role": "primary_10_dataset_characteristics",
        "authoritative": True,
        "derived_from": None,
    },
    {
        "artifact_name": "round5_overall_performance",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/tables/02_overall_performance.csv",
        "role": "scalable_10_dataset_performance_table",
        "authoritative": True,
        "derived_from": "round5_seed_aggregated_performance",
    },
    {
        "artifact_name": "round5_fraud_oriented_performance",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/tables/03_fraud_oriented_performance.csv",
        "role": "fraud_oriented_7_dataset_performance_table",
        "authoritative": True,
        "derived_from": "round5_seed_aggregated_performance",
    },
    {
        "artifact_name": "round5_scalability_support",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/tables/04_scalability_support.csv",
        "role": "scalability_and_exact_support_accounting",
        "authoritative": True,
        "derived_from": "round5_support_matrix",
    },
    {
        "artifact_name": "round5_dlg_extended_components",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/tables/05_dlg_extended_components.csv",
        "role": "dlg_local_global_fusion_component_analysis",
        "authoritative": True,
        "derived_from": "round4b_component_freeze",
    },
    {
        "artifact_name": "round5_graph_topology",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_round5_final/topology/final_graph_topology.csv",
        "role": "graph_topological_metrics",
        "authoritative": True,
        "derived_from": None,
    },
    {
        "artifact_name": "lanl_external_validation",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_defense_extension_real_final/tables/table_d2_lanl_external_validation.csv",
        "role": "lanl_redteam_external_validation_table",
        "authoritative": True,
        "derived_from": None,
    },
    {
        "artifact_name": "lanl_ground_truth_freeze",
        "relative_path": "dlg_gnn/outputs/benchmark/sci_defense_extension_real_final/defense_validation/lanl_ground_truth_freeze.json",
        "role": "lanl_graph_and_split_freeze_manifest",
        "authoritative": True,
        "derived_from": None,
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main():
    registry = []
    for item in AUTHORITATIVE_ARTIFACTS:
        full_path = REPO_ROOT / item["relative_path"]
        if not full_path.exists():
            raise FileNotFoundError(f"Missing authoritative artifact: {full_path}")
        h = sha256_file(full_path)
        entry = {
            "artifact_name": item["artifact_name"],
            "artifact_path": item["relative_path"],
            "sha256": h,
            "role": item["role"],
            "authoritative": item["authoritative"],
            "derived_from": item["derived_from"],
        }
        registry.append(entry)
        print(f"[{item['artifact_name']}] {h} -> {item['relative_path']}")

    out_file = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/audit/manuscript_source_registry.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    print(f"\nWrote registry to {out_file} with {len(registry)} artifacts.")


if __name__ == "__main__":
    main()
