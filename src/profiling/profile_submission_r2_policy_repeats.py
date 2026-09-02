"""Warm-started five-repeat comparison of all production routing policies."""
from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure_timing.yaml"))
    args = parser.parse_args(); cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    base = yaml.safe_load(Path(cfg["base_config"]).read_text(encoding="utf-8"))
    source, output = Path(cfg["source_results"]), Path(cfg["output_root"])
    events = raw_events([], int(cfg["runtime"]["samples_per_repeat"]), source/"profiling/raw_events_100000.parquet")[:int(cfg["runtime"]["samples_per_repeat"])]
    warm = events[:int(cfg["runtime"]["warmup_samples"])]
    selections = pd.read_csv(output/"cascade/interface_trace.csv").set_index("seed")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rows = []
    for seed in map(int, cfg["seeds"]):
        level1, level2, tabular, frozen = load_seed_bundle(source/f"checkpoints/seed{seed}", device)
        calibrated = copy.deepcopy(frozen); selected = selections.loc[seed]
        calibrated["thresholds"]["ProductionLevel1GIN"] = float(selected.fast_threshold)
        calibrated["cutoffs"]["ProductionLevel1GIN"] = {"dual": float(selected.route_cutoff), "risk_controlled": float(selected.route_cutoff)}
        replay(warm, seed, "ProductionLevel1GIN", "direct_only", 1, level1, level2, tabular, frozen, base, device)
        policies = [
            ("direct_only", "direct_only", frozen),
            ("full_deep", "no_routing", frozen),
            ("legacy_dual", "dual", frozen),
            ("legacy_risk_controlled", "risk_controlled", frozen),
            ("validation_calibrated_dual", "dual", calibrated),
        ]
        for policy_name, replay_policy, metadata in policies:
            for repeat in range(1, int(cfg["runtime"]["repeats"])+1):
                _, summary = replay(events, seed, "ProductionLevel1GIN", replay_policy, 1,
                    level1, level2, tabular, metadata, base, device)
                summary.update({"policy": policy_name, "repeat": repeat, "warmup_events": len(warm),
                    "accuracy_evidence_status": "UNDEFINED_ALL_BENIGN_RAW_PREFIX"}); rows.append(summary)
        print(f"policy timing seed={seed}", flush=True)
    frame = pd.DataFrame(rows); atomic_csv(output/"runtime/five_repeat_policy_timing.csv", frame)
    summary = frame.groupby("policy").agg(repeats=("repeat", "count"), seeds=("seed", "nunique"),
        deep_route_rate=("deep_route_rate", "mean"), mean_latency_ms=("mean_latency_ms", "mean"),
        mean_latency_sd_ms=("mean_latency_ms", "std"), p95_latency_ms=("p95_latency_ms", "mean"),
        p99_latency_ms=("p99_latency_ms", "mean"), throughput_events_s=("throughput_events_per_second", "mean"),
        rss_peak_bytes=("rss_peak_bytes", "max"), vram_peak_bytes=("vram_peak_bytes", "max")).reset_index()
    full = float(summary.loc[summary.policy=="full_deep", "mean_latency_ms"].iloc[0])
    summary["latency_reduction_vs_full_deep_pct"] = 100*(full-summary.mean_latency_ms)/full
    atomic_csv(output/"runtime/five_repeat_policy_summary.csv", summary)
    atomic_json(output/"runtime/policy_timing_protocol.json", {"warmup_events": len(warm), "events_per_repeat": len(events),
        "repeats_per_seed_policy": int(cfg["runtime"]["repeats"]), "seeds": list(cfg["seeds"]),
        "identical_prefix": True, "positive_support": 0, "accuracy_metrics_used": False, "device": str(device)})
    print(json.dumps(summary[["policy","deep_route_rate","mean_latency_ms","latency_reduction_vs_full_deep_pct"]].to_dict("records"), indent=2))


if __name__ == "__main__":
    main()
