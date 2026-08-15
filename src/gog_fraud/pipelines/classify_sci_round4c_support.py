"""Create the evidence ledger that completes Round 4C support accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from gog_fraud.experiments.round4c_policy import canonical_hash
from gog_fraud.pipelines.analyze_sci_round4c import collect_results
from gog_fraud.pipelines.analyze_sci_round4c_completion import _oom_stage


RESOURCE_REASON = (
    "reproduced exact historical implementation materialization exceeds the "
    "current 8 GB-class GPU resource envelope"
)


def _raw_paths(root: Path) -> dict[tuple[str, str, int], str]:
    result = {}
    for path in sorted((root / "raw").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        result[(row["dataset"], row["model"], int(row["seed"]))] = str(path.relative_to(root))
    return result


def build_ledger(config: dict, output: Path) -> dict:
    raw = collect_results(output)
    paths = _raw_paths(output)
    classifications: list[dict] = []

    # Two production seeds constitute direct resource evidence.  Reddit may
    # use one measured seed plus deterministic N/E/architecture policy.
    gadnr = raw.loc[raw.model.eq("GADNR") & raw.status.eq("failed_oom")]
    for dataset, group in gadnr.groupby("dataset"):
        stages = {_oom_stage("GADNR", str(row.failure_message)) for row in group.itertuples()}
        # One seed may retain the full C++ traceback while the other stores a
        # compact allocator message.  Prefer the specific traced stage when
        # requested allocation and dataset/path are otherwise identical.
        stage = ("neighborhood_covariance_inverse"
                 if "neighborhood_covariance_inverse" in stages
                 else "native_neighborhood_or_sage_materialization")
        for row in group.itertuples():
            classifications.append({
                "dataset": dataset, "model": "GADNR", "seed": int(row.seed),
                "final_status": "unsupported_resource_exact_implementation",
                "evidence_mode": "measured", "restriction_reason": RESOURCE_REASON,
                "evidence_path": paths[(dataset, "GADNR", int(row.seed))],
                "oom_stage": stage, "observed_runtime_sec": float(row.total_wall_sec),
            })
        if dataset == "Reddit" and set(group.seed.astype(int)) == {42}:
            classifications.append({
                "dataset": dataset, "model": "GADNR", "seed": 43,
                "final_status": "unsupported_resource_exact_implementation",
                "evidence_mode": "policy",
                "restriction_reason": RESOURCE_REASON + "; seed does not change N/E/materialization shape",
                "evidence_path": paths[(dataset, "GADNR", 42)], "oom_stage": stage,
            })

    anomaly = raw.loc[raw.model.eq("AnomalyDAE") & raw.status.eq("failed_oom")]
    for dataset, group in anomaly.groupby("dataset"):
        for row in group.itertuples():
            classifications.append({
                "dataset": dataset, "model": "AnomalyDAE", "seed": int(row.seed),
                "final_status": "unsupported_resource_exact_implementation",
                "evidence_mode": "measured",
                "restriction_reason": "exact AnomalyDAE encoder COO/GAT materialization exceeds GPU memory",
                "evidence_path": paths[(dataset, "AnomalyDAE", int(row.seed))],
                "oom_stage": _oom_stage("AnomalyDAE", str(row.failure_message)),
                "observed_runtime_sec": float(row.total_wall_sec),
            })
        if dataset == "Reddit" and set(group.seed.astype(int)) == {42}:
            classifications.append({
                "dataset": dataset, "model": "AnomalyDAE", "seed": 43,
                "final_status": "unsupported_resource_exact_implementation",
                "evidence_mode": "policy",
                "restriction_reason": "same graph/backend/pair count and encoder materialization as measured seed 42",
                "evidence_path": paths[(dataset, "AnomalyDAE", 42)],
                "oom_stage": "encoder_gat_coo_aggregation",
            })

    operational = raw.loc[
        raw.model.eq("AnomalyDAE") & raw.status.eq("unsupported_operational")
        & raw.dataset.eq("Reddit")
    ]
    if not operational.empty:
        measured_row = operational.sort_values("seed").iloc[0]
        measured_seed = int(measured_row.seed)
        classifications.append({
            "dataset": "Reddit", "model": "AnomalyDAE", "seed": measured_seed,
            "final_status": "unsupported_operational", "evidence_mode": "measured",
            "restriction_reason": "exact production runtime exceeds predeclared 24 GPU-hour budget",
            "evidence_path": paths[("Reddit", "AnomalyDAE", measured_seed)],
            "observed_runtime_sec": float(measured_row.total_wall_sec),
        })
        other_seed = 43 if measured_seed == 42 else 42
        if not ((raw.dataset.eq("Reddit")) & raw.model.eq("AnomalyDAE") & raw.seed.eq(other_seed)).any():
            classifications.append({
                "dataset": "Reddit", "model": "AnomalyDAE", "seed": other_seed,
                "final_status": "unsupported_operational", "evidence_mode": "policy",
                "restriction_reason": "same graph/backend/epochs/O(N^2) pair count as measured operational limit",
                "evidence_path": paths[("Reddit", "AnomalyDAE", measured_seed)],
            })

    guard_path = output / "resources/dgraphfin_anomalydae_guard.json"
    if guard_path.exists():
        guard = json.loads(guard_path.read_text(encoding="utf-8"))
        if guard.get("guard_completed") and float(guard.get("cumulative_active_sec", 0)) >= 24 * 3600:
            classifications.extend([
                {
                    "dataset": "DGraphFin", "model": "AnomalyDAE", "seed": 42,
                    "final_status": "unsupported_operational", "evidence_mode": "measured",
                    "restriction_reason": "exact nonlinear all-pairs production runtime exceeds predeclared 24 GPU-hour budget",
                    "evidence_path": str(guard_path.relative_to(output)),
                    "observed_runtime_sec": float(guard["cumulative_active_sec"]),
                    "observed_epochs": int(guard.get("observed_epochs", 0)),
                    "production_projection_hours": float(guard.get("production_projection_hours", 211.4)),
                },
                {
                    "dataset": "DGraphFin", "model": "AnomalyDAE", "seed": 43,
                    "final_status": "unsupported_operational", "evidence_mode": "policy",
                    "restriction_reason": "same graph/backend/epochs/O(N^2) pair count; model seed does not alter production complexity",
                    "evidence_path": str(guard_path.relative_to(output)),
                    "production_projection_hours": float(guard.get("production_projection_hours", 211.4)),
                },
            ])

    # A measured result always wins over a policy projection if both exist.
    ranked = {"policy": 0, "measured": 1}
    deduplicated = {}
    for row in classifications:
        key = (row["dataset"], row["model"], int(row["seed"]))
        previous = deduplicated.get(key)
        if previous is None or ranked[row["evidence_mode"]] > ranked[previous["evidence_mode"]]:
            deduplicated[key] = row
    ordered = sorted(deduplicated.values(), key=lambda row: (row["dataset"], row["model"], row["seed"]))
    return {
        "policy": "raw observations are immutable; explained reproducible resource limits are classified separately",
        "config_hash": canonical_hash(config),
        "classifications": ordered,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_root"])
    ledger = build_ledger(config, output)
    path = output / "manifests/support_reclassification.json"
    path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "classifications": len(ledger["classifications"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
