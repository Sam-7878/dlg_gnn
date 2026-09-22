import numpy as np

from experiments.round5.analysis import adaptive_ece, classification_metrics, expected_calibration_error


def test_calibration_metrics_are_finite_and_bounded():
    labels = np.array([0, 0, 1, 1])
    probability = np.array([.1, .2, .8, .9])
    result = classification_metrics(labels, probability)
    assert all(np.isfinite(value) for value in result.values())
    assert 0 <= expected_calibration_error(labels, probability) <= 1
    assert 0 <= adaptive_ece(labels, probability, bins=2) <= 1
