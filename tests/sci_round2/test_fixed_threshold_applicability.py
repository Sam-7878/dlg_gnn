import numpy as np
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol


def test_unbounded_score_marks_fixed_point_five_not_applicable():
    result = evaluate_threshold_protocol([0, 0, 1, 1], [1, 2, 8, 9], [0, 0, 1, 1], [2, 3, 7, 8])
    assert not result.fixed_05_applicable
    assert np.isnan(result.f1_at_05)

