import pytest

from gog_fraud.selection.router import SelectiveRouter, TriageOutput
from gog_fraud.selection.thresholds import RoutingThresholds, optimize_compute_constrained


def triage(score, variance=0.01):
    return TriageOutput(score, variance, variance ** 0.5, 0.0, None, 8)


def test_dual_threshold_boundaries():
    router = SelectiveRouter(tau_b=0.2, tau_f=0.8, tau_u=0.1, threshold_version="v1")
    assert router.route(triage(0.2)).route == "benign_direct"
    assert router.route(triage(0.8)).route == "fraud_direct"
    assert router.route(triage(0.5)).route == "deep_inspection"
    assert router.route(triage(0.1, 0.2)).reason == "uncertainty_threshold"


def test_risk_sensitive_routing_overrides_confidence():
    router = SelectiveRouter(tau_b=0.2, tau_f=0.8, tau_u=0.1, tau_r=0.9, tau_q=0.7, threshold_version="v2")
    assert router.route(triage(0.95)).route == "deep_inspection"
    assert router.route(triage(0.1), graph_risk_prior=0.8).reason == "graph_risk_threshold"


def test_invalid_thresholds_rejected():
    with pytest.raises(ValueError): SelectiveRouter(tau_b=0.8, tau_f=0.2, tau_u=0.1, threshold_version="bad")


def test_threshold_tuning_is_validation_only(tmp_path):
    candidates = [(0.2, 0.8, 0.1), (0.3, 0.7, 0.2)]
    selected = optimize_compute_constrained([0, 1, 0, 1], [0.1, 0.9, 0.4, 0.6], [0.01] * 4, deep_budget=0.5, candidates=candidates)
    path = tmp_path / "threshold.json"; selected.save(path)
    assert RoutingThresholds.load(path) == selected
    with pytest.raises(ValueError): optimize_compute_constrained([0], [0.1], [0.1], deep_budget=1, candidates=candidates, split="test")
