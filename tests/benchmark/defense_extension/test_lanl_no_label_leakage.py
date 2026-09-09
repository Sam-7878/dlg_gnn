"""Feature/label dependency tests for the LANL adapter."""
import inspect
import numpy as np
import torch

from gog_fraud.extensions.defense.defense_registry import load_defense_dataset
from gog_fraud.extensions.defense.lanl_redteam_adapter import LanlRedTeamGraphBuilder


def test_lanl_feature_builder_has_no_redteam_dependency():
    assert "redteam" not in inspect.signature(LanlRedTeamGraphBuilder.extract_features).parameters
    assert "redteam" not in inspect.signature(LanlRedTeamGraphBuilder.add_auth_event).parameters

    builder = LanlRedTeamGraphBuilder()
    builder.add_auth_event(1, "u", "c1", "c2", "Kerberos", "Network", "LogOn", True)
    before = builder.extract_features().copy()
    builder.add_redteam_compromise(2, "u", "c1", "c2")
    after = builder.extract_features().copy()
    np.testing.assert_array_equal(before, after)
    assert builder.build_labels().tolist() == [0, 1]


def test_lanl_feature_label_correlation_is_diagnostic_only():
    data = load_defense_dataset("LANL-RedTeam")
    x = data.x.detach().cpu().numpy().astype(np.float64)
    y = data.y.detach().cpu().numpy().astype(np.float64)

    assert torch.isfinite(data.x).all(), "LANL feature matrix contains NaN or Inf"
    assert len(x) == len(y), "Mismatch between feature rows and label length"

    # Correlation is retained as a diagnostic, not as proof of non-leakage.
    correlations = []
    for col_idx in range(x.shape[1]):
        feat = x[:, col_idx]
        if np.std(feat) > 1e-8 and np.std(y) > 1e-8:
            correlations.append(abs(float(np.corrcoef(feat, y)[0, 1])))
    assert correlations and all(np.isfinite(correlations))


def test_lanl_graph_structure():
    data = load_defense_dataset("LANL-RedTeam")
    n = data.num_nodes
    e = data.edge_index

    assert e.size(0) == 2
    assert e.min() >= 0
    assert e.max() < n
    assert int((data.y == 1).sum()) > 0, "No positive nodes in LANL"
    assert int((data.y == 0).sum()) > 0, "No negative nodes in LANL"
