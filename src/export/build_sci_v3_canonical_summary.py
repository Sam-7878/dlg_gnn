#!/usr/bin/env python3
"""Build the SCI-v3-final single source of truth from raw machine-readable artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gog_fraud.evaluation.calibration import binary_calibration_metrics, fit_temperature
from validation.sci_v3_final_common import atomic_csv, atomic_json, binary_metrics, sha256_file


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def build_metrics(repo: Path) -> pd.DataFrame:
    frames = []
    main = repo / "results/results_sci_v2/paper_eligible_results_long.csv"
    if main.exists():
        frame = pd.read_csv(main)
        wanted = [column for column in ("chain", "seed", "model", "roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy", "fraud_recall", "precision", "fnr", "threshold", "prediction_path", "prediction_sha256") if column in frame]
        frames.append(frame[wanted].assign(evidence_family="legacy_main_raw_predictions"))
    for path, family in (
        (repo / "results/sci_v3/baselines/tabular/tabular_baselines_metrics.csv", "tabular_baseline"),
        (repo / "results/sci_v3/baselines/gnn/supervised_gnn_baselines_metrics.csv", "supervised_gnn_baseline"),
        (repo / "results/sci_v3_final/baselines/tabular_l2_cascade_metrics.csv", "selective_cascade"),
    ):
        if path.exists():
            frame = pd.read_csv(path)
            rename = {"recall": "fraud_recall", "prediction_artifact": "prediction_path"}
            frames.append(frame.rename(columns=rename).assign(evidence_family=family, source_artifact=str(path)))
    selective = repo / "results/sci_v3/selective_risk/selective_risk_summary.csv"
    if selective.exists():
        frames.append(pd.read_csv(selective).assign(model="DLG-StreamMC", evidence_family="selective_risk", source_artifact=str(selective)))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_routing(repo: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((repo / "results/sci_v3/traces").glob("trace__*__*.parquet")):
        frame = pd.read_parquet(path)
        y = frame.label.to_numpy(dtype=int)
        final = frame.final_score.to_numpy(dtype=float)
        threshold_candidates = frame.loc[frame.final_decision == 1, "final_score"]
        threshold = float(threshold_candidates.min()) if len(threshold_candidates) else 1.0
        deep = frame.route == "deep_inspection"
        metrics = binary_metrics(y, final, threshold)
        l1_metrics = binary_metrics(y, frame.l1_score, threshold)
        before = frame.l1_decision.to_numpy(dtype=int)
        after = frame.final_decision.to_numpy(dtype=int)
        before_correct, after_correct = before == y, after == y
        rows.append(
            {
                "chain": frame.chain.iloc[0],
                "seed": int(frame.seed.iloc[0]),
                "policy": frame.policy.iloc[0],
                "T": int(frame.mc_samples.iloc[0]),
                "n_total": len(frame),
                "n_direct": int((~deep).sum()),
                "n_deep": int(deep.sum()),
                "deep_route_rate": float(deep.mean()),
                "deep_avoidance_rate": float((~deep).mean()),
                "deep_rows_missing_fusion": int(frame.loc[deep, "fusion_score"].isna().sum()),
                "direct_rows_with_fusion": int(frame.loc[~deep, "fusion_score"].notna().sum()),
                "wrong_to_correct": int((~before_correct & after_correct).sum()),
                "correct_to_wrong": int((before_correct & ~after_correct).sum()),
                "delta_f1_raw": float(metrics["f1"] - l1_metrics["f1"]),
                "delta_recall_raw": (
                    float(metrics["fraud_recall"] - l1_metrics["fraud_recall"])
                    if metrics["fraud_recall"] is not None and l1_metrics["fraud_recall"] is not None
                    else np.nan
                ),
                "raw_trace_artifact": str(path),
                "raw_trace_sha256": sha256_file(path),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def build_costs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_frames = []
    old = repo / "results/sci_v3/component_cost_benchmark.csv"
    if old.exists():
        component_frames.append(pd.read_csv(old).assign(scope="legacy_model_component", source_artifact=str(old)))
    profile = repo / "results/sci_v3_final/profiling/end_to_end_stage_profile.csv"
    if profile.exists():
        component_frames.append(pd.read_csv(profile).assign(source_artifact=str(profile)))
    components = pd.concat(component_frames, ignore_index=True, sort=False) if component_frames else pd.DataFrame()

    end_frames = []
    gate = repo / "results/sci_v3/gate_a_cost_evaluation.csv"
    if gate.exists():
        modeled = pd.read_csv(gate).rename(columns={"cost_selective_ms": "selective_cost_ms", "cost_full_ms": "full_cost_ms"})
        if old.exists():
            raw = pd.read_csv(old)
            averages = raw.groupby(["chain", "T"], as_index=False).agg(
                l1_deterministic_ms=("l1_deterministic_ms", "mean"),
                mc_l1_mean_ms=("mc_l1_mean_ms", "mean"),
            )
            modeled = modeled.merge(averages, on=["chain", "T"], how="left")
            modeled["selective_cost_recomputed_ms"] = modeled.mc_l1_mean_ms + modeled.p_deep * (modeled.full_cost_ms - modeled.l1_deterministic_ms)
            modeled["cost_residual_ms"] = modeled.selective_cost_ms - modeled.selective_cost_recomputed_ms
        modeled["cost_type"] = "analytical_reconstruction"
        modeled["measured_selective_e2e"] = False
        modeled["source_artifact"] = str(gate)
        end_frames.append(modeled)
    if profile.exists():
        frame = pd.read_csv(profile)
        total = frame[frame.stage == "total end-to-end batch inference"].copy()
        total["cost_type"] = "measured_batch_e2e"
        total["measured_selective_e2e"] = False
        total["full_cost_ms"] = total["mean_ms"]
        total["source_artifact"] = str(profile)
        end_frames.append(total)
    return components, (pd.concat(end_frames, ignore_index=True, sort=False) if end_frames else pd.DataFrame())


def build_calibration(repo: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((repo / "results/sci_v3_final/risk_control/raw_predictions").glob("*.parquet")):
        frame = pd.read_parquet(path)
        validation = frame[frame.split == "validation"]
        test = frame[frame.split == "test"]
        if validation.empty or test.empty:
            continue
        logits_validation = np.log(np.clip(validation.score, 1e-7, 1 - 1e-7) / np.clip(1 - validation.score, 1e-7, 1))
        temperature = fit_temperature(validation.label.to_numpy(), logits_validation.to_numpy())
        logits_test = np.log(np.clip(test.score, 1e-7, 1 - 1e-7) / np.clip(1 - test.score, 1e-7, 1))
        scaled = 1.0 / (1.0 + np.exp(-logits_test / temperature))
        chain, seed_text = path.stem.split("__seed")
        for method, scores in (("mc_dropout", test.score.to_numpy()), ("temperature_scaled_mc", scaled)):
            rows.append(
                {
                    "chain": chain,
                    "seed": int(seed_text),
                    "method": method,
                    "temperature": temperature if method.startswith("temperature") else 1.0,
                    "fit_scope": "validation_only",
                    "evaluation_scope": "test_only",
                    "raw_prediction_artifact": str(path),
                    **binary_calibration_metrics(test.label.to_numpy(), scores),
                }
            )
    return pd.DataFrame(rows)


def build_statistics(repo: Path) -> pd.DataFrame:
    frames = []
    for path, kind in (
        (repo / "results/sci_v3_final/statistics/routing_flip_significance.csv", "routing_flip"),
        (repo / "results/sci_v3_final/statistics/lpp_equivalence.csv", "lpp_equivalence"),
        (repo / "results/sci_v3_final/risk_control/risk_control_alpha_sweep.csv", "risk_control"),
    ):
        if path.exists():
            frames.append(pd.read_csv(path).assign(statistic_family=kind, source_artifact=str(path)))
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_cross_chain(repo: Path) -> pd.DataFrame:
    frames = []
    strict = repo / "results/sci_v3_final/cross_chain/cross_chain_temporal_strict_metrics.csv"
    if strict.exists():
        frames.append(pd.read_csv(strict).assign(evidence_role="main_generalization_evidence", source_artifact=str(strict)))
    diagnostic = repo / "results/results_sci_v2/cross_chain/cross_chain_metrics.csv"
    if diagnostic.exists():
        old = pd.read_csv(diagnostic)
        transfer = old[old.held_out_target_excluded_from_fit == True].copy()  # noqa: E712
        transfer["protocol"] = "cross_chain_full_history_diagnostic"
        transfer["evidence_role"] = "secondary_distribution_shift_diagnostic"
        transfer["source_artifact"] = str(diagnostic)
        frames.append(transfer)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def build_streaming(repo: Path) -> pd.DataFrame:
    path = repo / "results/sci_v3_final/streaming/table_streaming_scenarios.csv"
    return pd.read_csv(path).assign(source_artifact=str(path)) if path.exists() else pd.DataFrame()


def run(repo: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = build_metrics(repo)
    routing = build_routing(repo)
    components, end_to_end = build_costs(repo)
    calibration = build_calibration(repo)
    statistics = build_statistics(repo)
    cross_chain = build_cross_chain(repo)
    streaming = build_streaming(repo)
    tables = {
        "canonical_metrics.csv": metrics,
        "canonical_routing.csv": routing,
        "canonical_component_costs.csv": components,
        "canonical_end_to_end_costs.csv": end_to_end,
        "canonical_calibration.csv": calibration,
        "canonical_statistics.csv": statistics,
        "canonical_cross_chain.csv": cross_chain,
        "canonical_streaming.csv": streaming,
    }
    manifest_inputs = set()
    for frame in tables.values():
        for column in ("source_artifact", "raw_trace_artifact", "raw_prediction_artifact", "prediction_path"):
            if column in frame:
                manifest_inputs.update(str(value) for value in frame[column].dropna().unique())
    counts = {}
    for name, frame in tables.items():
        atomic_csv(output_dir / name, frame)
        counts[name] = len(frame)
    artifacts = []
    for value in sorted(manifest_inputs):
        path = Path(value)
        if not path.is_absolute():
            path = repo / path
        if path.exists() and path.is_file():
            artifacts.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    atomic_json(output_dir / "canonical_manifest.json", {"builder": __file__, "counts": counts, "raw_artifacts": artifacts})
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output-dir", default="results/sci_v3_final/canonical")
    args = parser.parse_args()
    counts = run(Path(args.repo).resolve(), Path(args.output_dir).resolve())
    print(json.dumps(counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
