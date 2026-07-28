from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .router import SelectiveRouter, TriageOutput


@dataclass(frozen=True)
class RoutingThresholds:
    tau_b: float
    tau_f: float
    tau_u: float
    version: str
    selected_on: str = "validation"

    def save(self, path: str | Path) -> None:
        target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RoutingThresholds":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def optimize_compute_constrained(y_true: Sequence[int], scores: Sequence[float], uncertainties: Sequence[float], *, deep_budget: float, candidates: Iterable[tuple[float, float, float]], split: str = "validation") -> RoutingThresholds:
    if split != "validation":
        raise ValueError("threshold optimization is allowed on validation data only")
    if not (len(y_true) == len(scores) == len(uncertainties)) or not y_true:
        raise ValueError("non-empty arrays with equal lengths are required")
    best_key: tuple[float, float, float, float, float] | None = None
    best_thresholds: tuple[float, float, float] | None = None
    for tau_b, tau_f, tau_u in candidates:
        router = SelectiveRouter(tau_b=tau_b, tau_f=tau_f, tau_u=tau_u, threshold_version="candidate")
        decisions = [router.route(TriageOutput(score, unc, unc ** 0.5, 0.0, None, 1)) for score, unc in zip(scores, uncertainties)]
        deep_rate = sum(item.route == "deep_inspection" for item in decisions) / len(decisions)
        if deep_rate > deep_budget:
            continue
        false_negatives = sum(label == 1 and decision.route == "benign_direct" for label, decision in zip(y_true, decisions))
        positives = max(1, sum(int(label == 1) for label in y_true))
        objective = false_negatives / positives
        key = (objective, -deep_rate, tau_b, tau_f, tau_u)
        if best_key is None or key < best_key:
            best_key, best_thresholds = key, (tau_b, tau_f, tau_u)
    if best_thresholds is None:
        raise ValueError("no candidate satisfies the deep-route budget")
    tau_b, tau_f, tau_u = best_thresholds
    return RoutingThresholds(tau_b, tau_f, tau_u, f"validation_budget_{deep_budget:g}")
