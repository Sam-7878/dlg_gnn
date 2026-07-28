from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from gog_fraud.data.io.streaming_dataset import StatefulTransactionStream, StreamEvent
from gog_fraud.selection.router import RoutingDecision, SelectiveRouter, TriageOutput


@dataclass(frozen=True)
class PredictionTrace:
    sample_id: str
    chain: str
    event_time: int
    triage_mean: float
    uncertainty: float
    route: str
    final_score: float
    triage_latency_ms: float
    deep_latency_ms: float
    total_latency_ms: float
    model_version: str
    threshold_version: str


class StatefulStreamingEngine:
    """Model-agnostic selective inference engine with auditable sample traces."""

    def __init__(self, *, stream: StatefulTransactionStream, router: SelectiveRouter, triage_fn: Callable[[StreamEvent], TriageOutput], deep_fn: Callable[[StreamEvent], float], model_version: str) -> None:
        self.stream, self.router = stream, router
        self.triage_fn, self.deep_fn, self.model_version = triage_fn, deep_fn, model_version
        self.traces: list[PredictionTrace] = []

    def run(self, *, risk_prior_fn: Callable[[StreamEvent], float | None] | None = None, max_events: int | None = None) -> list[PredictionTrace]:
        for index, event in enumerate(self.stream):
            if max_events is not None and index >= max_events:
                break
            started = time.perf_counter()
            triage_started = time.perf_counter(); triage = self.triage_fn(event)
            triage_ms = (time.perf_counter() - triage_started) * 1000.0
            prior = risk_prior_fn(event) if risk_prior_fn else None
            decision = self.router.route(triage, graph_risk_prior=prior)
            deep_ms = 0.0
            if decision.route == "deep_inspection":
                deep_started = time.perf_counter(); final_score = float(self.deep_fn(event))
                deep_ms = (time.perf_counter() - deep_started) * 1000.0
            else:
                final_score = triage.mean_score
            self.traces.append(PredictionTrace(event.sample_id, event.chain_id, event.event_time, triage.mean_score, triage.variance, decision.route, final_score, triage_ms, deep_ms, (time.perf_counter() - started) * 1000.0, self.model_version, decision.threshold_version))
        return self.traces

    @property
    def prediction_hash(self) -> str:
        payload = json.dumps([asdict(trace) | {"triage_latency_ms": 0, "deep_latency_ms": 0, "total_latency_ms": 0} for trace in self.traces], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def routing_summary(self) -> dict[str, float | int]:
        total = len(self.traces)
        counts = {route: sum(trace.route == route for trace in self.traces) for route in ("benign_direct", "fraud_direct", "deep_inspection")}
        return {"num_samples": total, **{f"{key}_rate": value / total if total else 0.0 for key, value in counts.items()}}
