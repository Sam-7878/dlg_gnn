import numpy as np

from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol, topk_predictions


def test_validation_threshold_is_not_reselected_on_test():
    result = evaluate_threshold_protocol(
        [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9],
        [0, 1, 0, 1], [0.1, 0.2, 0.8, 0.9],
    )
    assert result.validation_threshold >= 0.8
    assert result.validation_f1 < result.oracle_best_f1
    assert result.topk_prevalence_source == "validation_labels"


def test_topk_predictions_are_exact_and_stable():
    pred = topk_predictions([0.5, 0.5, 0.4], 1)
    assert pred.tolist() == [1, 0, 0]

