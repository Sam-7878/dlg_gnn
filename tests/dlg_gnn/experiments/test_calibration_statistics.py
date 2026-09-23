import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from gog_fraud.evaluation.calibration import binary_calibration_metrics, fit_temperature, reliability_rows
from gog_fraud.evaluation.statistics import holm_adjust, paired_bootstrap_difference, paired_effect_size


def test_calibration_metrics_and_temperature_are_deterministic():
    y = np.array([0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = binary_calibration_metrics(y, p)
    assert metrics["brier"] == pytest.approx(0.025)
    assert len(reliability_rows(y, p, bins=10)) == 10
    assert fit_temperature(y, np.log(p / (1 - p))) > 0


def test_paired_bootstrap_effect_and_holm():
    y = np.array([0, 0, 0, 1, 1, 1])
    better = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]); worse = np.array([0.6, 0.4, 0.2, 0.3, 0.5, 0.7])
    result = paired_bootstrap_difference(y, better, worse, roc_auc_score, iterations=200, seed=7)
    assert result["difference"] > 0
    assert paired_effect_size([2, 3, 4], [1, 1, 1]) > 0
    assert holm_adjust([0.01, 0.04])[0] <= holm_adjust([0.01, 0.04])[1]
