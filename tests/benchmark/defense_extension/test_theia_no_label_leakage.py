"""Feature/label dependency tests for the THEIA adapter."""
import inspect
import numpy as np
import torch

from gog_fraud.extensions.defense.defense_registry import load_defense_dataset
from gog_fraud.extensions.defense.darpa_theia_adapter import (
    DarpaTheiaGraphBuilder,
    NODE_TYPE_FILE,
    NODE_TYPE_PROCESS,
)


def test_theia_feature_builder_has_no_ground_truth_dependency():
    assert "ground_truth" not in inspect.signature(DarpaTheiaGraphBuilder.extract_features).parameters
    assert "malicious" not in inspect.signature(DarpaTheiaGraphBuilder.add_event).parameters

    builder = DarpaTheiaGraphBuilder()
    builder.add_event("p1", "f1", NODE_TYPE_PROCESS, NODE_TYPE_FILE, "EVENT_READ", 1)
    before = builder.extract_features().copy()
    builder.mark_ground_truth_entity("p1")
    after = builder.extract_features().copy()
    np.testing.assert_array_equal(before, after)
    assert builder.build_labels().tolist() == [1, 0]


def test_theia_feature_label_correlation_is_diagnostic_only():
    data = load_defense_dataset("DARPA-TC-THEIA")
    x = data.x.detach().cpu().numpy().astype(np.float64)
    y = data.y.detach().cpu().numpy().astype(np.float64)

    assert torch.isfinite(data.x).all(), "THEIA feature matrix contains NaN or Inf"
    assert len(x) == len(y), "Mismatch between feature rows and label length"

    # Correlation is retained as a diagnostic, not as proof of non-leakage.
    correlations = []
    for col_idx in range(x.shape[1]):
        feat = x[:, col_idx]
        if np.std(feat) > 1e-8 and np.std(y) > 1e-8:
            correlations.append(abs(float(np.corrcoef(feat, y)[0, 1])))
    assert correlations and all(np.isfinite(correlations))


def test_theia_graph_structure():
    data = load_defense_dataset("DARPA-TC-THEIA")
    n = data.num_nodes
    e = data.edge_index

    assert e.size(0) == 2
    assert e.min() >= 0
    assert e.max() < n
    assert int((data.y == 1).sum()) > 0, "No positive nodes in THEIA"
    assert int((data.y == 0).sum()) > 0, "No negative nodes in THEIA"
