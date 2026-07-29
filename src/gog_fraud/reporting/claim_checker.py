from __future__ import annotations

from typing import Any


CLAIMS = (
    ("comparable to PyGOD", "main baseline table, identical split, 95% CI"),
    ("reduced deep inference", "sample routing trace and deep-route ratio"),
    ("bounded memory", "100k+ event memory trajectory and slope"),
    ("lower latency", "fair cold/steady latency comparison"),
    ("better calibration", "ECE/Brier/NLL and reliability data"),
    ("streaming capable", "stateful/recovery tests and scenario replay"),
    ("multi-chain robust", "per-chain and held-out cross-chain results"),
    ("analyst workload reduction", "review rate and direct-exit FNR"),
)


def verify_claims(*, experiments: list[dict[str, Any]], tests_passed: bool) -> list[dict[str, str]]:
    results = []
    for claim, required in CLAIMS:
        if claim == "streaming capable" and tests_passed:
            status, available, result = "PARTIALLY_SUPPORTED", "unit-level deterministic replay/checkpoint tests", "core state behavior verified; production/load scenario evidence missing"
        else:
            status, available, result = "MISSING_EVIDENCE", "none", "NOT_RUN"
        results.append({"claim": claim, "required_evidence": required, "available": available, "result": result, "status": status})
    return results
