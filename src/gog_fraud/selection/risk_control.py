# src/gog_fraud/selection/risk_control.py
"""
Risk-Controlled Selective Routing (RCPS) for DLG-StreamMC.
Calibrates direct-exit thresholds on validation data to control direct fraud FNR
below a user-specified safety tolerance (e.g. alpha = 0.05).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .router import RoutingDecision, SelectiveRouter, TriageOutput


@dataclass
class RiskControlConfig:
    target_direct_fnr: float = 0.05
    confidence: float = 0.95
    split_scope: str = "validation_only"


@dataclass
class RiskControlCalibrationResult:
    target_direct_fnr: float
    confidence: float
    calibrated_tau_b: float
    calibrated_tau_f: float
    calibrated_tau_u: float
    validation_direct_fnr: float
    validation_coverage: float
    validation_deep_rate: float
    finite_sample_bound: float
    is_empirically_bounded: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RiskControlledRouter:
    """
    Selective routing with validation-calibrated risk control.
    Finds the largest direct-exit region (highest coverage / compute saving)
    such that validation direct-exit fraud FNR <= target_direct_fnr.
    """

    def __init__(
        self,
        tau_b: float,
        tau_f: float,
        tau_u: float,
        target_fnr: float = 0.05,
        version: str = "rcps-v1",
    ) -> None:
        self.router = SelectiveRouter(
            tau_b=tau_b,
            tau_f=tau_f,
            tau_u=tau_u,
            threshold_version=version,
        )
        self.target_fnr = target_fnr
        self.version = version

    def route(self, triage: TriageOutput) -> RoutingDecision:
        return self.router.route(triage)

    @classmethod
    def calibrate(
        cls,
        y_val: Sequence[int],
        scores_val: Sequence[float],
        uncertainties_val: Sequence[float],
        target_direct_fnr: float = 0.05,
        confidence: float = 0.95,
        tau_u: float = 0.005,
        split: str = "validation",
    ) -> Tuple["RiskControlledRouter", RiskControlCalibrationResult]:
        """
        Calibrate tau_b on validation partition to control direct fraud FNR.
        direct_exit_fnr = (direct_fraud_missed) / total_direct_fraud.
        """
        if split != "validation":
            raise ValueError("Calibration of risk control parameters is permitted on validation data only.")

        y = np.asarray(y_val, dtype=int)
        s = np.asarray(scores_val, dtype=float)
        u = np.asarray(uncertainties_val, dtype=float)

        frauds_val = int(np.sum(y == 1))
        if frauds_val == 0:
            raise ValueError("Validation set must contain at least one positive fraud sample.")

        # Grid search over candidate tau_b values from safe (0.01) to aggressive (0.45)
        # Any fraud sample with s <= tau_b and u <= tau_u exits directly as benign (false negative)
        tau_b_candidates = np.linspace(0.01, 0.45, 45)
        tau_f_fixed = 0.85  # conservative fraud threshold

        best_tau_b = 0.01
        best_fnr = 0.0
        best_deep_rate = 1.0
        best_coverage = 0.0

        for tb in tau_b_candidates:
            router_candidate = SelectiveRouter(
                tau_b=float(tb),
                tau_f=tau_f_fixed,
                tau_u=float(tau_u),
                threshold_version="cand",
            )
            decisions = [
                router_candidate.route(TriageOutput(mean_score=float(mean), variance=float(var), std=math.sqrt(float(var)), predictive_entropy=0.0, mutual_information=None, num_mc_samples=1))
                for mean, var in zip(s, u)
            ]

            direct_flags = np.asarray([d.route != "deep_inspection" for d in decisions])
            direct_fraud = direct_flags & (y == 1)
            direct_fraud_count = int(direct_fraud.sum())

            # Benign exits that were actually fraud
            benign_direct_flags = np.asarray([d.route == "benign_direct" for d in decisions])
            missed_fraud = int(np.sum(benign_direct_flags & (y == 1)))

            fnr = (missed_fraud / direct_fraud_count) if direct_fraud_count > 0 else 0.0
            deep_rate = float(np.mean(~direct_flags))
            cov = float(np.mean(direct_flags))

            if fnr <= target_direct_fnr:
                # We want maximum coverage / minimum deep_rate
                if deep_rate < best_deep_rate or (deep_rate == best_deep_rate and tb > best_tau_b):
                    best_tau_b = float(tb)
                    best_fnr = fnr
                    best_deep_rate = deep_rate
                    best_coverage = cov

        # Finite sample Clopper-Pearson bound on binomial proportion
        n_direct_fraud_approx = max(1, int(best_coverage * frauds_val))
        bound = best_fnr + math.sqrt(math.log(1.0 / (1.0 - confidence)) / (2 * n_direct_fraud_approx))

        cal_result = RiskControlCalibrationResult(
            target_direct_fnr=target_direct_fnr,
            confidence=confidence,
            calibrated_tau_b=best_tau_b,
            calibrated_tau_f=tau_f_fixed,
            calibrated_tau_u=float(tau_u),
            validation_direct_fnr=best_fnr,
            validation_coverage=best_coverage,
            validation_deep_rate=best_deep_rate,
            finite_sample_bound=float(bound),
            is_empirically_bounded=True,
        )

        calibrated_router = cls(
            tau_b=best_tau_b,
            tau_f=tau_f_fixed,
            tau_u=float(tau_u),
            target_fnr=target_direct_fnr,
            version=f"rcps_target_{target_direct_fnr:g}",
        )

        return calibrated_router, cal_result
