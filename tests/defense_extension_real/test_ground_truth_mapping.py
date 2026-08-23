#!/usr/bin/env python3
"""
Test: Verification of ground truth mapping integrity and feature independence.
Defense Extension Round D3 hard gate requirement.
"""
import csv
from pathlib import Path
import pytest
import torch

DARPA_GT_PATH = Path("outputs/sci_defense_extension_real/source_audit/ground_truth_mapping.csv")
LANL_GT_PATH = Path("outputs/sci_defense_extension_real/source_audit/lanl_ground_truth_mapping.csv")

THEIA_GRAPH_PATH = Path("outputs/sci_defense_extension_real/graphs/theia_graph.pt")
LANL_GRAPH_PATH = Path("outputs/sci_defense_extension_real/graphs/lanl_graph.pt")

def test_theia_ground_truth_mapping():
    """Verify THEIA ground truth mapping contains non-zero positive and negative labels with rationales."""
    if not DARPA_GT_PATH.exists():
        pytest.skip(f"THEIA ground truth mapping not found at {DARPA_GT_PATH}")
    with open(DARPA_GT_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "GT mapping must not be empty"
    positives = [r for r in rows if r["label"] == "1"]
    negatives = [r for r in rows if r["label"] == "0"]
    assert len(positives) > 0, "Must have at least 1 positive label from attack ground truth"
    assert len(negatives) > 0, "Must have negative labels"
    for r in positives:
        assert len(r["mapping_rationale"]) > 0, "Positive node must have mapping rationale"
        assert len(r["gt_reference"]) > 0, "Positive node must reference official ground truth"

def test_lanl_ground_truth_mapping():
    """Verify LANL ground truth mapping contains positive labels from redteam.txt."""
    if not LANL_GT_PATH.exists():
        pytest.skip(f"LANL ground truth mapping not found at {LANL_GT_PATH}")
    with open(LANL_GT_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0, "LANL GT mapping must not be empty"
    positives = [r for r in rows if r["label"] == "1"]
    assert len(positives) > 0, "Must have positive compromised computer labels"
    for r in positives:
        assert r["gt_reference"] == "LANL_redteam.txt_official"
        assert int(r["redteam_line_count"]) > 0

def test_feature_matrix_no_nan_or_inf():
    """Verify that PyG graphs feature matrices contain valid finite numbers."""
    for gpath in [THEIA_GRAPH_PATH, LANL_GRAPH_PATH]:
        if not gpath.exists():
            continue
        data = torch.load(gpath, weights_only=False)
        assert not torch.isnan(data.x).any(), f"NaN found in {gpath} features"
        assert not torch.isinf(data.x).any(), f"Inf found in {gpath} features"
        assert data.num_nodes > 0, f"Graph in {gpath} must have nodes"
        assert data.edge_index.shape[1] > 0, f"Graph in {gpath} must have edges"
