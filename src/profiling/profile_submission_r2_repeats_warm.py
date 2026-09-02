"""Warm-started five-repeat timing for the calibrated SCI-v3 R2 route."""
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


def configure(metadata: dict, row: pd.Series) -> None:
    metadata["thresholds"]["ProductionLevel1GIN"] = float(row.fast_threshold)
    metadata["cutoffs"]["ProductionLevel1GIN"] = {"dual": float(row.route_cutoff), "risk_controlled": float(row.route_cutoff)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure_timing.yaml"))
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    source, output = Path(cfg["source_results"]), Path(cfg["output_root"])
    events = raw_events([], int(cfg["runtime"]["samples_per_repeat"]), source / "profiling/raw_events_100000.parquet")[:int(cfg["runtime"]["samples_per_repeat"])]
    warm = events[:int(cfg["runtime"]["warmup_samples"])]
    selections = pd.read_csv(output / "cascade/interface_trace.csv").set_index("seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for seed in map(int, cfg["seeds"]):
        level1, level2, tabular, metadata = load_seed_bundle(source / f"checkpoints/seed{seed}", device)
        configure(metadata, selections.loc[seed])
        replay(warm, seed, "ProductionLevel1GIN", "dual", 1, level1, level2, tabular, metadata, base, device)
        for repeat in range(1, int(cfg["runtime"]["repeats"]) + 1):
            _, summary = replay(events, seed, "ProductionLevel1GIN", "dual", 1, level1, level2, tabular, metadata, base, device)
            summary.update({"repeat": repeat, "warmup_events": len(warm), "selection": "validation_calibrated_route",
                            "accuracy_evidence_status": "UNDEFINED_ALL_BENIGN_RAW_PREFIX"}); rows.append(summary)
        print(f"warm timing seed={seed}", flush=True)
    atomic_csv(output / "runtime/five_repeat_timing.csv", pd.DataFrame(rows))
    seed = int(cfg["seeds"][0]); level1, level2, tabular, metadata = load_seed_bundle(source / f"checkpoints/seed{seed}", device)
    configure(metadata, selections.loc[seed]); sensitivity = []
    for samples in map(int, cfg["mc_sensitivity"]["samples"]):
        replay(warm, seed, "ProductionLevel1GIN", "dual", samples, level1, level2, tabular, metadata, base, device)
        _, summary = replay(events, seed, "ProductionLevel1GIN", "dual", samples, level1, level2, tabular, metadata, base, device)
        summary.update({"warmup_events": len(warm), "selection": "representative_seed11",
                        "accuracy_evidence_status": "UNDEFINED_ALL_BENIGN_RAW_PREFIX"}); sensitivity.append(summary)
        print(f"warm mc T={samples}", flush=True)
    atomic_csv(output / "runtime/mc_sensitivity_latency.csv", pd.DataFrame(sensitivity))
    atomic_json(output / "runtime/timing_protocol.json", {"timing_repeats": int(cfg["runtime"]["repeats"]),
        "events_per_repeat": len(events), "warmup_samples_executed": len(warm), "prefix_positive_support": 0,
        "calibrated_route_workload": True, "classification_metrics_used": False, "device": str(device)})
    print(json.dumps({"timing_rows": len(rows), "mc_rows": len(sensitivity), "warmup": len(warm), "device": str(device)}, indent=2))


if __name__ == "__main__":
    main()
