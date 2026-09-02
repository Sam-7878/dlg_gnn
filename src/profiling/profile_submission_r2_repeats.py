"""Five-repeat timing and extended MC sensitivity for SCI-v3 R2.

The replay uses the calibrated validation-selected GIN routing threshold/cutoff.
The legacy replay's score postprocessing is retained only to execute the same
deep-stage workload; classification metrics from the all-benign 500-event raw
prefix are explicitly undefined and are not used as accuracy evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from gog_fraud.production.closure import load_seed_bundle
from profiling.raw_event_selective_e2e_profiler import raw_events, replay
from validation.sci_v3_final_common import atomic_csv, atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    root, source = Path(cfg["output_root"]), Path(cfg["source_results"])
    event_path = source / "profiling/raw_events_100000.parquet"
    events = raw_events([], int(cfg["runtime"]["samples_per_repeat"]), event_path)[:int(cfg["runtime"]["samples_per_repeat"])]
    trace = pd.read_csv(root / "cascade/interface_trace.csv").set_index("seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for seed in map(int, cfg["seeds"]):
        level1, level2, tabular, metadata = load_seed_bundle(source / f"checkpoints/seed{seed}", device)
        selection = trace.loc[seed]
        metadata["thresholds"]["ProductionLevel1GIN"] = float(selection.fast_threshold)
        metadata["cutoffs"]["ProductionLevel1GIN"] = {"dual": float(selection.route_cutoff),
                                                       "risk_controlled": float(selection.route_cutoff)}
        for repeat in range(1, int(cfg["runtime"]["repeats"]) + 1):
            _, summary = replay(events, seed, "ProductionLevel1GIN", "dual", 1,
                                level1, level2, tabular, metadata, base, device)
            summary.update({"repeat": repeat, "selection": "validation_calibrated_route",
                            "accuracy_evidence_status": "UNDEFINED_ALL_BENIGN_RAW_PREFIX"})
            rows.append(summary)
        print(f"timing seed={seed}", flush=True)
    atomic_csv(root / "runtime/five_repeat_timing.csv", pd.DataFrame(rows))

    seed = int(cfg["seeds"][0]); level1, level2, tabular, metadata = load_seed_bundle(source / f"checkpoints/seed{seed}", device)
    selection = trace.loc[seed]
    metadata["thresholds"]["ProductionLevel1GIN"] = float(selection.fast_threshold)
    metadata["cutoffs"]["ProductionLevel1GIN"] = {"dual": float(selection.route_cutoff),
                                                   "risk_controlled": float(selection.route_cutoff)}
    sensitivity = []
    for samples in map(int, cfg["mc_sensitivity"]["samples"]):
        _, summary = replay(events, seed, "ProductionLevel1GIN", "dual", samples,
                            level1, level2, tabular, metadata, base, device)
        summary.update({"selection": "representative_seed11", "accuracy_evidence_status": "UNDEFINED_ALL_BENIGN_RAW_PREFIX"})
        sensitivity.append(summary); print(f"mc T={samples}", flush=True)
    atomic_csv(root / "runtime/mc_sensitivity_latency.csv", pd.DataFrame(sensitivity))
    atomic_json(root / "runtime/timing_protocol.json", {
        "timing_repeats": int(cfg["runtime"]["repeats"]), "events_per_repeat": len(events),
        "warmup_samples_requested": int(cfg["runtime"]["warmup_samples"]),
        "prefix_positive_support": int(sum(event.payload["label"] for event in events)),
        "calibrated_route_workload": True, "classification_metrics_used": False,
        "reason": "the preserved identical raw prefix has no fraud positives; held-out graph predictions provide accuracy evidence",
        "device": str(device)})
    print(json.dumps({"timing_rows": len(rows), "mc_rows": len(sensitivity), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
