import pytest
from gog_fraud.selection.risk_control import RiskControlledRouter
from gog_fraud.selection.expected_cost import ExpectedCostRouter, DEFAULT_COST_SCENARIOS


def test_risk_controlled_router_calibration():
    y_val = [0] * 90 + [1] * 10
    # True benigns have low scores, true frauds have high scores
    scores_val = [0.05] * 90 + [0.85] * 10
    uncertainties_val = [0.001] * 100

    router, cal_res = RiskControlledRouter.calibrate(
        y_val, scores_val, uncertainties_val, target_direct_fnr=0.05, split="validation"
    )

    assert cal_res.target_direct_fnr == 0.05
    assert cal_res.validation_direct_fnr <= 0.05
    assert cal_res.is_empirically_bounded is True


def test_expected_cost_router_optimization():
    y_val = [0] * 50 + [1] * 10
    scores_val = [0.1] * 50 + [0.9] * 10
    uncertainties_val = [0.002] * 60

    scenario = DEFAULT_COST_SCENARIOS[1]  # balanced
    router, opt_res = ExpectedCostRouter.optimize(
        y_val, scores_val, uncertainties_val, scenario, split="validation"
    )

    assert opt_res.scenario_name == "balanced"
    assert opt_res.expected_validation_cost >= 0.0
    assert opt_res.tau_b < opt_res.tau_f
