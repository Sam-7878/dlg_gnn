import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
AUDIT_DIR = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "audit"

def test_canonical_lanl_manifest():
    manifest_path = AUDIT_DIR / "lanl_canonical_manifest.json"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["dataset_name"] == "LANL-RedTeam"
    assert data["canonical_graph_sha256"] == "689c2968fe3ece9494196515e6089d6db3f430530e55b8b410d116b27c920359"
    assert data["num_nodes"] == 16694
    assert data["num_edges"] == 323897
    assert data["num_features"] == 13
    assert data["positive_count"] == 301
    assert data["negative_count"] == 16393
    assert data["hard_gates_passed"] is True
    
    auth = data["authoritative_results"]
    assert auth["GADNR"]["pr_auc"] == 0.1806
    assert auth["DLG-Base"]["pr_auc"] == 0.1367
    assert auth["DLG-Aug"]["pr_auc"] == 0.1114
    assert auth["GADNR"]["roc_auc"] == 0.8261
    assert auth["DLG-Base"]["roc_auc"] == 0.7923
    assert auth["DLG-Aug"]["roc_auc"] == 0.7188

def test_m2_source_registry():
    reg_path = AUDIT_DIR / "m2_manuscript_source_registry.json"
    assert reg_path.exists(), f"Missing registry: {reg_path}"
    
    with open(reg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert isinstance(data, list)
    assert len(data) >= 8
    
    names = {item["name"] for item in data}
    assert "round5_benchmark_raw" in names
    assert "round5_support_matrix" in names
    assert "lanl_canonical_graph" in names
    assert "lanl_canonical_manifest" in names
    assert "lanl_external_validation_table" in names
    
    for item in data:
        assert "name" in item
        assert "path" in item
        assert "sha256" in item
        assert len(item["sha256"]) == 64
