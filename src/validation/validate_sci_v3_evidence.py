#!/usr/bin/env python3
"""Fail-closed consistency checks for canonical SCI-v3-final evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from validation.sci_v3_final_common import atomic_json, sha256_file


class EvidenceError(RuntimeError):
    pass


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def unique_consistent(frame: pd.DataFrame, keys: list[str], name: str, errors: list[str]) -> None:
    if frame.empty or any(key not in frame for key in keys):
        return
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        grouped = frame.loc[duplicate].groupby(keys, dropna=False).size()
        errors.append(f"{name}: inconsistent/ambiguous duplicate canonical keys: {grouped.index.tolist()[:5]}")


def run(canonical_dir: Path, report_path: Path, tolerance: float) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    required_files = [
        "canonical_metrics.csv",
        "canonical_routing.csv",
        "canonical_component_costs.csv",
        "canonical_end_to_end_costs.csv",
        "canonical_calibration.csv",
        "canonical_statistics.csv",
        "canonical_cross_chain.csv",
        "canonical_streaming.csv",
        "canonical_manifest.json",
    ]
    for name in required_files:
        require((canonical_dir / name).exists(), f"missing canonical artifact: {name}", errors)
    if errors:
        payload = {"status": "FAIL", "errors": errors, "warnings": warnings}
        atomic_json(report_path, payload)
        raise EvidenceError("; ".join(errors))

    routing = pd.read_csv(canonical_dir / "canonical_routing.csv")
    costs = pd.read_csv(canonical_dir / "canonical_end_to_end_costs.csv")
    calibration = pd.read_csv(canonical_dir / "canonical_calibration.csv")
    cross_chain = pd.read_csv(canonical_dir / "canonical_cross_chain.csv")
    statistics = pd.read_csv(canonical_dir / "canonical_statistics.csv")
    streaming = pd.read_csv(canonical_dir / "canonical_streaming.csv")
    unique_consistent(routing, ["chain", "seed", "policy", "T"], "routing", errors)
    if not routing.empty:
        residual = np.abs(routing.deep_avoidance_rate - (1.0 - routing.deep_route_rate))
        require(bool((residual <= tolerance).all()), f"deep avoidance arithmetic residual max={residual.max()}", errors)
        require(bool((routing.deep_rows_missing_fusion == 0).all()), "deep routes missing Fusion scores", errors)
        require(bool((routing.direct_rows_with_fusion == 0).all()), "direct routes unexpectedly contain Fusion scores", errors)
        require(bool((routing.n_direct + routing.n_deep == routing.n_total).all()), "routing population counts do not reconcile", errors)
    modeled = costs[costs.cost_type == "analytical_reconstruction"] if "cost_type" in costs else pd.DataFrame()
    if not modeled.empty:
        require("cost_residual_ms" in modeled, "modeled costs lack residual", errors)
        if "cost_residual_ms" in modeled:
            require(bool((modeled.cost_residual_ms.abs() <= tolerance).all()), f"modeled cost residual max={modeled.cost_residual_ms.abs().max()}", errors)
        require(bool((modeled.measured_selective_e2e == False).all()), "modeled cost mislabeled as measured E2E", errors)  # noqa: E712
    require(not calibration.empty, "no validation-only canonical calibration records", errors)
    if not calibration.empty:
        require(bool((calibration.fit_scope == "validation_only").all()), "calibration contains non-validation fit scope", errors)
    main_cross = cross_chain[cross_chain.evidence_role == "main_generalization_evidence"] if "evidence_role" in cross_chain else pd.DataFrame()
    require(len(main_cross) == 15, f"strict cross-chain main evidence expected 15 rows, got {len(main_cross)}", errors)
    if not main_cross.empty:
        require(bool((main_cross.protocol == "strict_target_temporal_holdout").all()), "main cross-chain evidence includes non-temporal protocol", errors)
        require(bool((main_cross.target_excluded_from_fit == True).all()), "target chain included in cross-chain fit", errors)  # noqa: E712
    require("statistic_family" in statistics and (statistics.statistic_family == "routing_flip").any(), "routing significance evidence missing", errors)
    require(len(streaming) >= 10, f"streaming scenario evidence incomplete: {len(streaming)}", errors)
    if not streaming.empty:
        required_scenarios = {"normal", "burst", "overload", "cache_pressure", "checkpoint_restart", "delayed_events", "out_of_order_events", "long_run", "no_purge", "unbounded_reference"}
        require(required_scenarios.issubset(set(streaming.scenario)), "required streaming scenarios missing", errors)
    manifest = json.loads((canonical_dir / "canonical_manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest.get("raw_artifacts", []):
        path = Path(artifact["path"])
        require(path.exists(), f"manifest raw artifact missing: {path}", errors)
        if path.exists():
            require(sha256_file(path) == artifact["sha256"], f"manifest hash mismatch: {path}", errors)
    payload = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "routing_rows": len(routing),
            "strict_cross_chain_rows": len(main_cross),
            "streaming_scenarios": len(streaming),
            "modeled_cost_rows": len(modeled),
        },
    }
    atomic_json(report_path, payload)
    if errors:
        raise EvidenceError("; ".join(errors))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", default="results/sci_v3_final/canonical")
    parser.add_argument("--report", default="results/sci_v3_final/evidence_validation.json")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args()
    try:
        payload = run(Path(args.canonical_dir).resolve(), Path(args.report).resolve(), args.tolerance)
    except EvidenceError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}))
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
