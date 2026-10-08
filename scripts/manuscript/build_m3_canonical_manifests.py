"""
build_m3_canonical_manifests.py

Builds canonical source-of-truth manifests for Round M3:
1. control_reference_manifest.json (comparator rows matching frozen Round 5 and LANL D4)
2. lanl_canonical_manifest.json (canonical LANL graph and benchmark reference)
3. m3_manuscript_source_registry.json (authoritative file hashes and registries)

Strictly enforces:
- Elliptic: DLG-Base PR ≈ 0.0687, DLG-Aug PR ≈ 0.1037
- DGraphFin: DLG-Base PR ≈ 0.0101, DLG-Aug PR ≈ 0.0134
- LANL: DLG-Base PR ≈ 0.1367, DLG-Aug PR ≈ 0.1114, GADNR PR ≈ 0.1806
- Rejection of 0.2917/0.2798 and 0.0381/0.0267
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
M3_ROOT = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m3"
AUDIT_DIR = M3_ROOT / "audit"
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
    lanl_pt = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real" / "graphs" / "lanl_graph.pt"
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
    
    perf_table = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real_final" / "tables" / "table_d2_lanl_external_validation.csv"
    gt_freeze = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real_final" / "defense_validation" / "lanl_ground_truth_freeze.json"
    
    manifest = {
        "dataset_name": "LANL-RedTeam",
        "role": "external_labeled_validation",
        "canonical_graph_path": str(lanl_pt.relative_to(REPO_ROOT)).replace("\\", "/"),
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
                "path": str(perf_table.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(perf_table),
            },
            "ground_truth_freeze": {
                "path": str(gt_freeze.relative_to(REPO_ROOT)).replace("\\", "/"),
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


def build_control_reference_manifest():
    """Build manifest of all authoritative frozen comparator rows for capacity controls."""
    r5_raw_dir = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "raw"
    lanl_csv = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real" / "benchmark" / "benchmark_raw.csv"
    lanl_summary_csv = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real_final" / "tables" / "table_d2_lanl_external_validation.csv"

    records = []
    
    # 1. Elliptic and DGraphFin from Round 5 raw JSONs
    for ds in ["Elliptic", "DGraphFin"]:
        for model in ["DLG-Base", "DLG-Aug"]:
            for seed in [42, 43, 44, 45, 46]:
                pattern = f"{ds}__{model}__seed{seed}__*.json"
                matched = list(r5_raw_dir.glob(pattern))
                assert len(matched) == 1, f"Expected exactly 1 raw run for {pattern}, got {len(matched)}"
                raw_path = matched[0]
                data = json.loads(raw_path.read_text(encoding="utf-8"))
                
                records.append({
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "source_file": str(raw_path.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "source_sha256": sha256_file(raw_path),
                    "run_id": data.get("run_id"),
                    "cell_key": data.get("cell_key"),
                    "python_class": data.get("python_class"),
                    "config_hash": data.get("config_hash"),
                    "backend_hash": data.get("backend_hash"),
                    "configured_epochs": data.get("configured_epochs"),
                    "actual_epochs": data.get("actual_epochs"),
                    "split_strategy": data.get("split_strategy"),
                    "metric_source": "sci_round5_final_raw",
                    "roc_auc": data.get("roc_auc"),
                    "pr_auc": data.get("pr_auc"),
                    "validation_f1": data.get("validation_f1"),
                    "oracle_best_f1": data.get("oracle_best_f1"),
                    "train_time_sec": data.get("train_time_sec"),
                    "inference_time_sec": data.get("inference_time_sec"),
                    "total_wall_sec": data.get("total_wall_sec"),
                })

    # 2. LANL comparator rows from defense benchmark_raw.csv
    df_lanl = pd.read_csv(lanl_csv)
    df_lanl_sub = df_lanl[df_lanl["dataset"] == "LANL-RedTeam"]
    for model in ["DLG-Base", "DLG-Aug", "GADNR"]:
        for seed in [42, 43, 44, 45, 46]:
            row = df_lanl_sub[(df_lanl_sub["model"] == model) & (df_lanl_sub["seed"] == seed)]
            assert len(row) == 1, f"Expected 1 row for LANL/{model}/seed{seed}, got {len(row)}"
            r = row.iloc[0]
            records.append({
                "dataset": "LANL-RedTeam",
                "model": model,
                "seed": seed,
                "source_file": str(lanl_csv.relative_to(REPO_ROOT)).replace("\\", "/"),
                "source_sha256": sha256_file(lanl_csv),
                "run_id": str(r.get("run_id")),
                "cell_key": f"lanl_{model}_{seed}",
                "python_class": f"gog_fraud.models.{model}",
                "config_hash": "lanl_canonical_d4",
                "backend_hash": "exact_sparse",
                "configured_epochs": int(r.get("configured_epochs", 50)),
                "actual_epochs": int(r.get("actual_epochs", 50)),
                "split_strategy": "stratified_node_transductive",
                "metric_source": "sci_defense_extension_real_raw",
                "roc_auc": float(r["roc_auc"]),
                "pr_auc": float(r["pr_auc"]),
                "validation_f1": float(r["f1"]),
                "oracle_best_f1": None,
                "train_time_sec": float(r.get("fit_seconds", np.nan)),
                "inference_time_sec": None,
                "total_wall_sec": float(r.get("fit_seconds", np.nan)),
            })

    # Validate aggregate values against frozen primary truth
    df_rec = pd.DataFrame(records)
    summary = {}
    for (ds, m), group in df_rec.groupby(["dataset", "model"]):
        roc_m = float(group["roc_auc"].mean())
        roc_s = float(group["roc_auc"].std())
        pr_m = float(group["pr_auc"].mean())
        pr_s = float(group["pr_auc"].std())
        f1_m = float(group["validation_f1"].mean())
        f1_s = float(group["validation_f1"].std())
        summary[f"{ds}__{m}"] = {
            "dataset": ds,
            "model": m,
            "n_seeds": len(group),
            "roc_mean": roc_m, "roc_std": roc_s,
            "pr_mean": pr_m, "pr_std": pr_s,
            "f1_mean": f1_m, "f1_std": f1_s,
        }

    # Strict hard assertions matching Work Order Section 8
    # Elliptic: DLG-Base PR ≈ 0.0687, DLG-Aug PR ≈ 0.1037
    ell_base_pr = summary["Elliptic__DLG-Base"]["pr_mean"]
    ell_aug_pr = summary["Elliptic__DLG-Aug"]["pr_mean"]
    assert abs(ell_base_pr - 0.0687) < 0.001, f"Elliptic DLG-Base PR mismatch: {ell_base_pr}"
    assert abs(ell_aug_pr - 0.1037) < 0.001, f"Elliptic DLG-Aug PR mismatch: {ell_aug_pr}"
    assert not (0.29 < ell_base_pr < 0.30), "CRITICAL: BOGUS 0.2917 detected for Elliptic DLG-Base!"
    assert not (0.27 < ell_aug_pr < 0.29), "CRITICAL: BOGUS 0.2798 detected for Elliptic DLG-Aug!"

    # DGraphFin: DLG-Base PR ≈ 0.0101, DLG-Aug PR ≈ 0.0134
    dg_base_pr = summary["DGraphFin__DLG-Base"]["pr_mean"]
    dg_aug_pr = summary["DGraphFin__DLG-Aug"]["pr_mean"]
    assert abs(dg_base_pr - 0.0101) < 0.0005, f"DGraphFin DLG-Base PR mismatch: {dg_base_pr}"
    assert abs(dg_aug_pr - 0.0134) < 0.0005, f"DGraphFin DLG-Aug PR mismatch: {dg_aug_pr}"
    assert not (0.035 < dg_base_pr < 0.040), "CRITICAL: BOGUS 0.0381 detected for DGraphFin DLG-Base!"
    assert not (0.025 < dg_aug_pr < 0.030), "CRITICAL: BOGUS 0.0267 detected for DGraphFin DLG-Aug!"

    # LANL: DLG-Base PR ≈ 0.1367, DLG-Aug PR ≈ 0.1114, GADNR PR ≈ 0.1806
    lanl_base_pr = summary["LANL-RedTeam__DLG-Base"]["pr_mean"]
    lanl_aug_pr = summary["LANL-RedTeam__DLG-Aug"]["pr_mean"]
    lanl_gadnr_pr = summary["LANL-RedTeam__GADNR"]["pr_mean"]
    assert abs(lanl_base_pr - 0.1367) < 0.001, f"LANL DLG-Base PR mismatch: {lanl_base_pr}"
    assert abs(lanl_aug_pr - 0.1114) < 0.001, f"LANL DLG-Aug PR mismatch: {lanl_aug_pr}"
    assert abs(lanl_gadnr_pr - 0.1806) < 0.001, f"LANL GADNR PR mismatch: {lanl_gadnr_pr}"

    manifest = {
        "title": "Authoritative Frozen Comparator Reference Manifest (Round M3)",
        "hard_sanity_assertions_passed": True,
        "aggregates": summary,
        "records": records,
    }
    
    out_path = AUDIT_DIR / "control_reference_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Created control reference manifest at {out_path} ({len(records)} runs verified)")
    return manifest


def build_m3_source_registry():
    r5_raw = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "raw" / "benchmark_raw.csv"
    r5_support = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "manifests" / "model_dataset_support_matrix_v2.csv"
    r5_metrics = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "summary" / "seed_aggregated_performance.csv"
    r5_rankings = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "statistics" / "rankings.csv"
    r5_friedman = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "statistics" / "friedman_tests.csv"
    r5_wilcoxon = REPO_ROOT / "outputs" / "benchmark" / "sci_round5_final" / "statistics" / "wilcoxon_holm.csv"
    lanl_perf = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real_final" / "tables" / "table_d2_lanl_external_validation.csv"
    lanl_graph = REPO_ROOT / "outputs" / "benchmark" / "sci_defense_extension_real" / "graphs" / "lanl_graph.pt"
    lanl_canonical = AUDIT_DIR / "lanl_canonical_manifest.json"
    ctrl_ref = AUDIT_DIR / "control_reference_manifest.json"

    # Enforce unchanged primary hashes
    r5_raw_sha = sha256_file(r5_raw)
    r5_supp_sha = sha256_file(r5_support)
    assert r5_raw_sha == "39a497efe81a0d2630d8817e653d35b01bbb141de4a8d008a46a8c13f1c8375c", f"Round 5 raw hash changed! {r5_raw_sha}"
    assert r5_supp_sha == "c58dbca9a9e1ed14dfc025075820a3ad745f6cb70be77764c265d90af3522914", f"Round 5 support matrix hash changed! {r5_supp_sha}"

    registry = [
        {"name": "round5_benchmark_raw", "path": str(r5_raw.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": r5_raw_sha},
        {"name": "round5_support_matrix", "path": str(r5_support.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": r5_supp_sha},
        {"name": "round5_seed_aggregated_performance", "path": str(r5_metrics.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(r5_metrics)},
        {"name": "round5_rankings", "path": str(r5_rankings.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(r5_rankings)},
        {"name": "round5_friedman_tests", "path": str(r5_friedman.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(r5_friedman)},
        {"name": "round5_wilcoxon_holm", "path": str(r5_wilcoxon.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(r5_wilcoxon)},
        {"name": "lanl_canonical_graph", "path": str(lanl_graph.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(lanl_graph)},
        {"name": "lanl_canonical_manifest", "path": str(lanl_canonical.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(lanl_canonical)},
        {"name": "lanl_external_validation_table", "path": str(lanl_perf.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(lanl_perf)},
        {"name": "control_reference_manifest", "path": str(ctrl_ref.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(ctrl_ref)},
    ]

    out_path = AUDIT_DIR / "m3_manuscript_source_registry.json"
    out_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Created M3 source registry at {out_path}")
    return registry


if __name__ == "__main__":
    build_lanl_canonical_manifest()
    build_control_reference_manifest()
    build_m3_source_registry()
