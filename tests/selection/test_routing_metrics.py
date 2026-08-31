# tests/selection/test_routing_metrics.py
import pytest
from gog_fraud.evaluation.routing_metrics import (
    SampleRoutingTrace,
    FlipMetrics,
    evaluate_routing_traces,
    traces_to_dataframe,
)


def test_routing_metrics_flips_and_net_gain():
    # 4 samples:
    # 0: Direct benign -> correct
    # 1: Direct fraud -> missed (FN)
    # 2: Deep inspection -> L1 predicted 0 (wrong), Fusion predicted 1 (correct) -> wrong_to_correct (FN->TP)
    # 3: Deep inspection -> L1 predicted 1 (correct), Fusion predicted 0 (wrong) -> correct_to_wrong (TP->FN)
    sample_ids = ["s0", "s1", "s2", "s3"]
    labels = [0, 1, 1, 1]
    l1_scores = [0.1, 0.2, 0.3, 0.8]  # threshold = 0.5
    mc_means = [0.1, 0.2, 0.3, 0.8]
    mc_vars = [0.01, 0.01, 0.05, 0.05]
    routes = ["benign_direct", "benign_direct", "deep_inspection", "deep_inspection"]
    route_reasons = ["score_below_tau_b", "score_below_tau_b", "uncertainty_threshold", "uncertainty_threshold"]
    l2_scores = [0.0, 0.0, 0.9, 0.2]
    fusion_scores = [0.1, 0.2, 0.85, 0.35]

    threshold = 0.5

    traces, metrics = evaluate_routing_traces(
        sample_ids=sample_ids,
        labels=labels,
        l1_scores=l1_scores,
        mc_means=mc_means,
        mc_vars=mc_vars,
        routes=routes,
        route_reasons=route_reasons,
        l2_scores=l2_scores,
        fusion_scores=fusion_scores,
        threshold=threshold,
    )

    assert len(traces) == 4
    assert metrics.total_samples == 4
    assert metrics.n_direct == 2
    assert metrics.n_deep == 2
    assert metrics.n_flips == 2
    assert metrics.wrong_to_correct == 1
    assert metrics.correct_to_wrong == 1
    assert metrics.net_gain == 0
    assert metrics.fraud_fn_to_tp == 1
    assert metrics.fraud_tp_to_fn == 1

    df = traces_to_dataframe(traces)
    assert len(df) == 4
    assert "route" in df.columns
    assert "final_score" in df.columns
    # Check trace content
    assert df.loc[df["sample_id"] == "s0", "route"].item() == "benign_direct"
    assert df.loc[df["sample_id"] == "s2", "route"].item() == "deep_inspection"
    assert df.loc[df["sample_id"] == "s2", "final_score"].item() == 0.85
