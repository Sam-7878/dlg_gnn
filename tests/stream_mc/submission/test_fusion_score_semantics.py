import numpy as np
import pytest

from gog_fraud.production.calibrated_cascade import (
    PlattLogOddsCalibrator,
    apply_cascade,
    logit_to_probability,
    probability_to_logit,
    select_cascade_on_validation,
    weighted_logit_fusion,
)


def test_probability_logit_round_trip_is_stable() -> None:
    probabilities = np.array([1e-5, 0.1, 0.5, 0.9, 1.0 - 1e-5])
    np.testing.assert_allclose(logit_to_probability(probability_to_logit(probabilities)), probabilities, atol=1e-12)


def test_score_contract_rejects_logits_disguised_as_probabilities() -> None:
    with pytest.raises(ValueError, match="outside"):
        probability_to_logit(np.array([-1.0, 2.0]))


def test_fusion_endpoints_equal_the_selected_input() -> None:
    fast = np.array([0.1, 0.4, 0.8])
    deep = np.array([0.2, 0.7, 0.9])
    np.testing.assert_allclose(weighted_logit_fusion(fast, deep, 1.0), fast)
    np.testing.assert_allclose(weighted_logit_fusion(fast, deep, 0.0), deep)


def test_selection_uses_independent_final_threshold_and_is_test_label_free() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    fast = np.array([0.05, 0.15, 0.30, 0.40, 0.60, 0.85])
    deep = np.array([0.02, 0.10, 0.20, 0.80, 0.90, 0.95])
    calibrator = PlattLogOddsCalibrator.fit(fast, labels, random_state=7)
    calibrated = calibrator.transform(fast)
    selection = select_cascade_on_validation(
        labels, calibrated, deep, fast_weight_grid=[0.0, 0.5, 1.0], deep_budget_grid=[0.2, 0.5]
    )
    final, routed, fused = apply_cascade(calibrated, deep, selection)
    assert final.shape == routed.shape == fused.shape == labels.shape
    assert 0.0 <= selection.final_threshold <= 1.0
    assert selection.validation_f1 == pytest.approx(1.0)
