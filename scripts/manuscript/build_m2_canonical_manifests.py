"""
Build canonical source-of-truth manifests for Round M2.
Strictly validates:
- Round 5 primary benchmark raw artifacts and statistics
- D4 Final LANL graph artifact (N=16694, E=323897, F=13, positives=301)
"""

import hashlib
import json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
M2_ROOT = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m2"
AUDIT_DIR = M2_ROOT / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()


def build_lanl_canonical_manifest():
    lanl_pt = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_defense_extension_real/graphs/lanl_graph.pt"
    assert lanl_pt.exists(), f"Missing canonical LANL graph: {lanl_pt}"
    
    graph_bytes = lanl_pt.read_bytes()
    graph_sha = hashlib.sha256(graph_bytes).hexdigest()
    
    data = torch.load(lanl_pt, map_location="cpu", weights_only=False)
    
    n_nodes = int(data.num_nodes)
    n_edges = int(data.edge_index.size(1))
    n_features = int(data.x.size(1))
    n_positives = int(data.y.sum())
    
    assert n_nodes == 16694, f"Expected 16694 nodes, got {n_nodes}"
    assert n_edges == 323897, f"Expected 323897 edges, got {n_edges}"
    assert n_features == 13, f"Expected 13 features, got {n_features}"
    assert n_positives == 301, f"Expected 301 positives, got {n_positives}"
    
    x_sha = sha256_tensor(data.x)
    edge_index_sha = sha256_tensor(data.edge_index)
    y_sha = sha256_tensor(data.y)
    
    # Reference tables
    perf_table = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_defense_extension_real_final/tables/table_d2_lanl_external_validation.csv"
    gt_freeze = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_defense_extension_real_final/defense_validation/lanl_ground_truth_freeze.json"
    
    manifest = {
        "dataset_name": "LANL-RedTeam",
        "role": "external_labeled_validation",
        "canonical_graph_path": str(lanl_pt.relative_to(REPO_ROOT)),
        "canonical_graph_sha256": graph_sha,
        "x_sha256": x_sha,
        "edge_index_sha256": edge_index_sha,
        "y_sha256": y_sha,
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "num_features": n_features,
        "positive_count": n_positives,
        "negative_count": n_nodes - n_positives,
        "anomaly_rate": n_positives / n_nodes,
        "source_of_truth_artifacts": {
            "performance_table": {
                "path": str(perf_table.relative_to(REPO_ROOT)),
                "sha256": sha256_file(perf_table),
            },
            "ground_truth_freeze": {
                "path": str(gt_freeze.relative_to(REPO_ROOT)),
                "sha256": sha256_file(gt_freeze),
            }
        },
        "authoritative_results": {
            "GADNR": {"roc_auc": 0.8261, "pr_auc": 0.1806, "val_f1": 0.2475},
            "DLG-Base": {"roc_auc": 0.7923, "pr_auc": 0.1367, "val_f1": 0.2112},
            "DLG-Aug": {"roc_auc": 0.7188, "pr_auc": 0.1114, "val_f1": 0.1683},
            "DOMINANT": {"roc_auc": 0.7997, "pr_auc": 0.1348, "val_f1": 0.1740},
            "CONAD": {"roc_auc": 0.7997, "pr_auc": 0.1348, "val_f1": 0.1740},
            "AnomalyDAE": {"roc_auc": 0.5548, "pr_auc": 0.0450, "val_f1": 0.0688},
            "CoLA": {"roc_auc": 0.4663, "pr_auc": 0.0200, "val_f1": 0.0295},
            "OCGNN": {"roc_auc": 0.3762, "pr_auc": 0.0454, "val_f1": 0.0568}
        },
        "split_protocol": "stratified_node_transductive",
        "val_ratio": 0.2,
        "test_ratio": 0.2,
        "hard_gates_passed": True
    }
    
    out_path = AUDIT_DIR / "lanl_canonical_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created canonical LANL manifest at {out_path}")
    return manifest


def build_m2_source_registry():
    r5_raw = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/raw/benchmark_raw.csv"
    r5_support = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/manifests/model_dataset_support_matrix_v2.csv"
    r5_metrics = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/summary/seed_aggregated_performance.csv"
    r5_rankings = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/rankings.csv"
    r5_friedman = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/friedman_tests.csv"
    r5_wilcoxon = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_round5_final/statistics/wilcoxon_holm.csv"
    lanl_perf = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_defense_extension_real_final/tables/table_d2_lanl_external_validation.csv"
    lanl_graph = REPO_ROOT / "dlg_gnn/outputs/benchmark/sci_defense_extension_real/graphs/lanl_graph.pt"
    lanl_canonical = AUDIT_DIR / "lanl_canonical_manifest.json"

    registry = [
        {"name": "round5_benchmark_raw", "path": str(r5_raw.relative_to(REPO_ROOT)), "sha256": sha256_file(r5_raw)},
        {"name": "round5_support_matrix", "path": str(r5_support.relative_to(REPO_ROOT)), "sha256": sha256_file(r5_support)},
        {"name": "round5_seed_aggregated_performance", "path": str(r5_metrics.relative_to(REPO_ROOT)), "sha256": sha256_file(r5_metrics)},
        {"name": "round5_rankings", "path": str(r5_rankings.relative_to(REPO_ROOT)), "sha256": sha256_file(r5_rankings)},
        {"name": "round5_friedman_tests", "path": str(r5_friedman.relative_to(REPO_ROOT)), "sha256": sha256_file(r5_friedman)},
        {"name": "round5_wilcoxon_holm", "path": str(r5_wilcoxon.relative_to(REPO_ROOT)), "sha256": sha256_file(r5_wilcoxon)},
        {"name": "lanl_canonical_graph", "path": str(lanl_graph.relative_to(REPO_ROOT)), "sha256": sha256_file(lanl_graph)},
        {"name": "lanl_canonical_manifest", "path": str(lanl_canonical.relative_to(REPO_ROOT)), "sha256": sha256_file(lanl_canonical)},
        {"name": "lanl_external_validation_table", "path": str(lanl_perf.relative_to(REPO_ROOT)), "sha256": sha256_file(lanl_perf)},
    ]

    out_path = AUDIT_DIR / "m2_manuscript_source_registry.json"
    out_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Created M2 source registry at {out_path}")
    return registry


if __name__ == "__main__":
    build_lanl_canonical_manifest()
    build_m2_source_registry()
