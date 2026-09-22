import numpy as np
import pytest

from experiments.round5.analysis import apply_temperature, fit_temperature


def test_temperature_requires_two_class_validation_and_is_positive():
    with pytest.raises(ValueError):
        fit_temperature(np.zeros(5), np.full(5, 0.2))
    labels = np.array([0, 0, 0, 1, 1, 1])
    probability = np.array([.05, .1, .4, .6, .8, .95])
    temperature = fit_temperature(labels, probability)
    assert temperature > 0
    calibrated = apply_temperature(probability, temperature)
    assert np.all((calibrated > 0) & (calibrated < 1))
