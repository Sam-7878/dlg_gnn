import pytest
import numpy as np
from gog_fraud.evaluation.selective_metrics import (
    compute_risk_coverage_curve,
    risk_coverage_to_dataframe,
    RiskCoveragePoint,
    SelectiveRiskSummary,
)


def test_risk_coverage_monotonicity():
    # 10 samples, perfect confidence ordering
    y_true = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    scores = [0.1, 0.1, 0.2, 0.1, 0.2, 0.9, 0.8, 0.9, 0.7, 0.4] # last one is an error (FN if th=0.5)
    # Perfect uncertainty ranking: lowest uncertainty on correct samples, high on incorrect
    uncertainties = [0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.04, 0.05, 0.06, 0.25]

    points, summary = compute_risk_coverage_curve(
        y_true, scores, uncertainties, decision_threshold=0.5, num_points=10
    )

    assert len(points) == 10
    assert summary.aurc > 0.0
    assert summary.e_aurc >= 0.0

    df = risk_coverage_to_dataframe(points)
    assert len(df) == 10
    # At coverage < 0.9, no errors are included, risk must be 0
    assert df.iloc[0]["selective_risk"] == 0.0
    assert df.iloc[-1]["selective_risk"] == 0.1  # 1 error out of 10
