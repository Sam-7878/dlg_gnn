# src/gog_fraud/selection/expected_cost.py
"""
Expected Cost Selective Router for DLG-StreamMC.
Optimizes routing thresholds under asymmetric economic costs:
Total Cost = C_FN * FN + C_FP * FP + C_deep * N_deep
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .router import RoutingDecision, SelectiveRouter, TriageOutput


@dataclass
class CostScenario:
    name: str
    c_fn: float  # cost of false negative (missed fraud)
    c_fp: float  # cost of false positive (false alarm)
    c_deep: float  # computational cost of running Level 2 / Fusion


DEFAULT_COST_SCENARIOS = [
    CostScenario("compute_sensitive", c_fn=10.0, c_fp=1.0, c_deep=2.0),
    CostScenario("balanced", c_fn=50.0, c_fp=2.0, c_deep=1.0),
    CostScenario("fraud_risk_sensitive", c_fn=200.0, c_fp=5.0, c_deep=1.0),
]


@dataclass
class CostOptimizationResult:
    scenario_name: str
    tau_b: float
    tau_f: float
    tau_u: float
    expected_validation_cost: float
    validation_fn: int
    validation_fp: int
    validation_deep: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExpectedCostRouter:
    """
    Optimizes (tau_b, tau_f, tau_u) on validation data to minimize Total Expected Cost.
    """

    def __init__(
        self,
        tau_b: float,
        tau_f: float,
        tau_u: float,
        scenario: CostScenario,
        version: str = "cost-v1",
    ) -> None:
        self.router = SelectiveRouter(
            tau_b=tau_b,
            tau_f=tau_f,
            tau_u=tau_u,
            threshold_version=version,
        )
        self.scenario = scenario
        self.version = version

    def route(self, triage: TriageOutput) -> RoutingDecision:
        return self.router.route(triage)

    @classmethod
    def optimize(
        cls,
        y_val: Sequence[int],
        scores_val: Sequence[float],
        uncertainties_val: Sequence[float],
        scenario: CostScenario,
        tau_u_candidates: Sequence[float] = [0.001, 0.005, 0.01],
        tau_b_candidates: Sequence[float] = np.linspace(0.1, 0.4, 7),
        tau_f_candidates: Sequence[float] = np.linspace(0.6, 0.9, 7),
        split: str = "validation",
    ) -> Tuple["ExpectedCostRouter", CostOptimizationResult]:
        if split != "validation":
            raise ValueError("Expected cost optimization is strictly restricted to validation data.")

        y = np.asarray(y_val, dtype=int)
        s = np.asarray(scores_val, dtype=float)
        u = np.asarray(uncertainties_val, dtype=float)

        best_cost = float("inf")
        best_thresholds = (0.2, 0.8, 0.005)
        best_counts = (0, 0, 0)

        for tu in tau_u_candidates:
            for tb in tau_b_candidates:
                for tf in tau_f_candidates:
                    if tb >= tf:
                        continue

                    router = SelectiveRouter(tau_b=float(tb), tau_f=float(tf), tau_u=float(tu), threshold_version="cost_opt")
                    decisions = [
                        router.route(TriageOutput(mean_score=float(sc), variance=float(unc), std=math.sqrt(float(unc)), predictive_entropy=0.0, mutual_information=None, num_mc_samples=1))
                        for sc, unc in zip(s, u)
                    ]

                    # Direct decisions:
                    # benign_direct -> pred 0
                    # fraud_direct -> pred 1
                    # deep_inspection -> handled by deep model; incurs c_deep
                    n_deep = sum(d.route == "deep_inspection" for d in decisions)
                    direct_pred_0 = np.asarray([d.route == "benign_direct" for d in decisions])
                    direct_pred_1 = np.asarray([d.route == "fraud_direct" for d in decisions])

                    fn = int(np.sum(direct_pred_0 & (y == 1)))
                    fp = int(np.sum(direct_pred_1 & (y == 0)))

                    total_cost = scenario.c_fn * fn + scenario.c_fp * fp + scenario.c_deep * n_deep

                    if total_cost < best_cost:
                        best_cost = total_cost
                        best_thresholds = (float(tb), float(tf), float(tu))
                        best_counts = (fn, fp, n_deep)

        opt_tb, opt_tf, opt_tu = best_thresholds
        opt_fn, opt_fp, opt_deep = best_counts

        res = CostOptimizationResult(
            scenario_name=scenario.name,
            tau_b=opt_tb,
            tau_f=opt_tf,
            tau_u=opt_tu,
            expected_validation_cost=best_cost,
            validation_fn=opt_fn,
            validation_fp=opt_fp,
            validation_deep=opt_deep,
        )

        return cls(opt_tb, opt_tf, opt_tu, scenario, version=f"cost_{scenario.name}"), res
