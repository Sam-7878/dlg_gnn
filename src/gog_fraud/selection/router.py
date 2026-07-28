from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TriageOutput:
    mean_score: float
    variance: float
    std: float
    predictive_entropy: float
    mutual_information: float | None
    num_mc_samples: int


@dataclass(frozen=True)
class RoutingDecision:
    route: Literal["benign_direct", "fraud_direct", "deep_inspection"]
    reason: str
    risk_score: float
    uncertainty: float
    threshold_version: str


class SelectiveRouter:
    def __init__(self, *, tau_b: float, tau_f: float, tau_u: float, threshold_version: str, tau_r: float | None = None, tau_q: float | None = None) -> None:
        if not 0.0 <= tau_b < tau_f <= 1.0:
            raise ValueError("thresholds must satisfy 0 <= tau_b < tau_f <= 1")
        if tau_u < 0.0 or (tau_r is not None and not 0.0 <= tau_r <= 1.0) or (tau_q is not None and not 0.0 <= tau_q <= 1.0):
            raise ValueError("invalid uncertainty or risk threshold")
        self.tau_b, self.tau_f, self.tau_u = tau_b, tau_f, tau_u
        self.tau_r, self.tau_q, self.version = tau_r, tau_q, threshold_version

    def route(self, triage: TriageOutput, *, graph_risk_prior: float | None = None) -> RoutingDecision:
        score, uncertainty = triage.mean_score, triage.variance
        if self.tau_r is not None and score >= self.tau_r:
            return self._deep("risk_score_threshold", score, uncertainty)
        if self.tau_q is not None and graph_risk_prior is not None and graph_risk_prior >= self.tau_q:
            return self._deep("graph_risk_threshold", score, uncertainty)
        if uncertainty > self.tau_u:
            return self._deep("uncertainty_threshold", score, uncertainty)
        if score <= self.tau_b:
            return RoutingDecision("benign_direct", "low_risk_confident", score, uncertainty, self.version)
        if score >= self.tau_f:
            return RoutingDecision("fraud_direct", "high_risk_confident", score, uncertainty, self.version)
        return self._deep("abstention_region", score, uncertainty)

    def _deep(self, reason: str, score: float, uncertainty: float) -> RoutingDecision:
        return RoutingDecision("deep_inspection", reason, score, uncertainty, self.version)
