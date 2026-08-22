"""Fail-closed Round D2 artifact gates."""
import json
from pathlib import Path

import pandas as pd


ROOT = Path("outputs/sci_defense_extension/d2")


def test_d2_synthetic_lineage_forces_not_paper_ready():
    lineage = json.loads((ROOT / "manifests/defense_source_lineage.json").read_text(encoding="utf-8"))
    gate = json.loads((ROOT / "manifests/paper_readiness_gate.json").read_text(encoding="utf-8"))
    assert lineage["artifact_origin"] == "deterministic_synthetic_generator"
    assert not lineage["official_raw_files_traceable"]
    assert gate["decision"] == "NOT_PAPER_READY"
    assert gate["round5_hashes_unchanged"]
    assert gate["defense_d1_raw_unchanged"]


def test_d2_feature_lineage_and_sensitivity_accounting():
    features = pd.read_csv(ROOT / "leakage/feature_lineage.csv")
    assert len(features) == 32
    assert not features["uses_ground_truth"].astype(bool).any()
    manifest = json.loads((ROOT / "sensitivity/manifest.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / "sensitivity/theia_no_total_events_raw.csv")
    assert manifest["runs"] == manifest["expected_runs"] == len(raw) == 15
    assert manifest["graph_unchanged"] and manifest["labels_unchanged"]


def test_d2_equivalence_and_statistics_gates():
    gadnr = json.loads((ROOT / "gadnr/gadnr_compatibility_equivalence.json").read_text(encoding="utf-8"))
    stats = json.loads((ROOT / "statistics/verification.json").read_text(encoding="utf-8"))
    assert gadnr["acceptance_passed"]
    assert gadnr["connected_graph"]["score_max_abs_error"] <= 1e-6
    assert gadnr["isolated_node"]["corrected_scores_finite"]
    assert stats["rank_recomputation_matches_reported"]
    assert stats["pairwise_recomputation_matches_reported"]
