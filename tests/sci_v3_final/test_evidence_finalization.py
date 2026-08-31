from __future__ import annotations

import numpy as np
import pytest

from analysis.test_routing_flip_significance import holm
from gog_fraud.models.baselines.cascade import apply_budgeted_cascade, ambiguity_cutoff
from validation.validate_risk_control import binomial_upper, routing_counts


def test_deep_budget_cutoff_is_selected_from_validation_scores() -> None:
    validation = np.asarray([0.1, 0.4, 0.49, 0.51, 0.6, 0.9])
    cutoff = ambiguity_cutoff(validation, threshold=0.5, deep_budget=0.5)
    final, deep = apply_budgeted_cascade(
        np.asarray([0.1, 0.48, 0.8]),
        np.asarray([0.2, 0.9, 0.1]),
        threshold=0.5,
        margin_cutoff=cutoff,
    )
    assert deep.tolist() == [False, True, False]
    assert final.tolist() == pytest.approx([0.1, 0.9, 0.8])


def test_conditional_direct_fnr_uses_actual_direct_fraud_denominator() -> None:
    result = routing_counts(
        np.asarray([1, 1, 1, 0]),
        np.asarray([0.1, 0.9, 0.5, 0.1]),
        np.zeros(4),
        tau_b=0.2,
        tau_f=0.8,
        tau_u=0.1,
    )
    assert result["n_direct_fraud"] == 2
    assert result["false_negatives"] == 1
    assert result["risk"] == pytest.approx(0.5)


def test_clopper_pearson_upper_bound_is_fail_closed_without_support() -> None:
    assert binomial_upper(0, 0, 0.05) == 1.0
    assert 0.0 < binomial_upper(0, 100, 0.05) < 0.05


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm([0.01, 0.04, 0.20])
    assert all(0.0 <= value <= 1.0 for value in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
