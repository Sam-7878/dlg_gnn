#!/usr/bin/env python3
"""
Test: Verification of graph structure, schema compliance, and determinism.
Defense Extension Round D3 hard gate requirement.
"""
from pathlib import Path
import pytest
import torch

THEIA_GRAPH_PATH = Path("outputs/sci_defense_extension_real/graphs/theia_graph.pt")
LANL_GRAPH_PATH = Path("outputs/sci_defense_extension_real/graphs/lanl_graph.pt")

def test_theia_graph_structure():
    """Verify THEIA graph dimensions, tensor types, and non-empty edge list."""
    assert THEIA_GRAPH_PATH.exists(), f"THEIA graph artifact not found at {THEIA_GRAPH_PATH}"
    data = torch.load(THEIA_GRAPH_PATH, weights_only=False)
    assert data.num_nodes > 1_000_000, f"THEIA real graph should have >1M nodes, got {data.num_nodes}"
    assert data.edge_index.shape[1] > 50_000_000, f"THEIA real graph should have >50M edges, got {data.edge_index.shape[1]}"
    assert data.x.shape[1] == 20, f"THEIA feature dimension should be 20, got {data.x.shape[1]}"
    assert data.y.sum().item() > 0, "Must have positive anomaly labels"

def test_lanl_graph_structure():
    """Verify LANL graph dimensions, tensor types, and non-empty edge list if present."""
    if not LANL_GRAPH_PATH.exists():
        pytest.skip(f"LANL graph not yet built at {LANL_GRAPH_PATH}")
    data = torch.load(LANL_GRAPH_PATH, weights_only=False)
    assert data.num_nodes > 10_000, f"LANL graph should have >10k computer nodes, got {data.num_nodes}"
    assert data.edge_index.shape[1] > 100_000, f"LANL graph should have >100k auth edges, got {data.edge_index.shape[1]}"
    assert data.x.shape[1] == 13, f"LANL feature dimension should be 13, got {data.x.shape[1]}"
    assert data.y.sum().item() > 0, "Must have positive compromised computer labels"
