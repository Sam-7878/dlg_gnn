"""Fail-closed SCI-v3 production-path submission gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from validation.sci_v3_final_common import atomic_json, sha256_file


class SubmissionError(RuntimeError): pass


def run(repo: Path) -> dict:
    root = repo / "results/sci_v3_submission"; errors: list[str] = []; warnings: list[str] = []
    def require(value: bool, message: str) -> None:
        if not value: errors.append(message)
    required = [root / "method_identity.json", root / "canonical/canonical_manifest.json", root / "profiling/raw_event_e2e_summary.csv",
        root / "baselines/table_cascade_accuracy_cost.csv", root / "risk_control/temporal_risk_audit.csv",
        root / "integrated/table_integrated_100k.csv", root / "revision_manifest.json",
        root / "streaming/table_integrated_streaming_e2e.csv",
        root / "figures_and_tables/figures/fig_risk_cost_frontier.pdf",
        root / "figures_and_tables/figures/fig_accuracy_latency_frontier.pdf",
        root / "figures_and_tables/figures/fig_fnr_latency_frontier.pdf",
        root / "figures_and_tables/figures/fig_integrated_memory_latency.pdf",
        root / "figures_and_tables/figures/fig_integrated_queue_latency.pdf",
        root / "figures_and_tables/figures/fig_method_production_path.pdf"]
    reports = repo / "docs/work_reports/sci_v3_submission"
    required.extend(reports / name for name in ("method_identity_closure.md", "raw_event_e2e_runtime_report.md",
        "measured_cascade_frontier_report.md", "integrated_streaming_inference_report.md", "final_claim_freeze_report.md"))
    for path in required: require(path.exists(), f"missing artifact: {path.relative_to(repo)}")
    if errors:
        payload = {"status": "FAIL", "errors": errors, "warnings": warnings}; atomic_json(root / "submission_validation.json", payload); raise SubmissionError("; ".join(errors))
    identity = json.loads((root / "method_identity.json").read_text(encoding="utf-8"))
    require(identity["paper_method_backbone"] == identity["executed_backbone"], "paper/executed backbone mismatch")
    require(identity["profiler_backbone"] == identity["executed_backbone"], "profiler/executed backbone mismatch")
    require(identity["routing_backbone"] == identity["executed_backbone"], "routing/executed backbone mismatch")
    for checkpoint in sorted((root / "checkpoints").glob("seed*/metadata.json")):
        checkpoint_identity = json.loads(checkpoint.read_text(encoding="utf-8"))["method_identity"]
        require(all(checkpoint_identity[key] == identity[key] for key in ("paper_method_backbone", "executed_backbone", "profiler_backbone", "routing_backbone")), f"checkpoint identity mismatch: {checkpoint}")
    runtime = pd.read_csv(root / "profiling/raw_event_e2e_summary.csv")
    required_support = {"N", "N_positive", "N_negative", "metric_defined", "undefined_reason"}
    require(required_support.issubset(runtime.columns), "runtime rows lack metric support fields")
    require(set(runtime.measurement_type.dropna()) <= {"measured_e2e", "measured_component", "modeled_reconstruction"}, "invalid cost provenance")
    require((runtime.events > 0).all(), "empty runtime scenario")
    for scenario, policy in (("XGBoostFastTriage", "only"), ("XGBoostFastTriage", "dual"),
                             ("LightGBMFastTriage", "only"), ("LightGBMFastTriage", "dual")):
        selected = runtime[(runtime.scenario == scenario) & (runtime.policy == policy) & (runtime.events == 500)]
        require(set(selected.seed) == {11, 22, 33, 44, 55}, f"missing 5-seed runtime: {scenario}/{policy}")
    for policy in ("no_routing", "dual", "risk_controlled", "adaptive_mc"):
        selected = runtime[(runtime.scenario == "ProductionLevel1GIN") & (runtime.policy == policy) & (runtime.events == 500) & (runtime.mc_samples == 1)]
        require(set(selected.seed) == {11, 22, 33, 44, 55}, f"missing production 5-seed policy runtime: {policy}")
        sweep = runtime[(runtime.scenario == "ProductionLevel1GIN") & (runtime.policy == policy) & (runtime.events == 500) & (runtime.seed == 11)]
        require(set(sweep.mc_samples) == {1, 3, 5, 8}, f"incomplete MC sweep: {policy}")
    traces = list((root / "profiling/raw_traces").glob("*.csv")) + [root / "integrated/raw_trace_100k.csv"]; require(bool(traces), "no raw execution traces")
    direct_count = deep_count = 0
    for path in traces:
        frame = pd.read_csv(path); direct = ~frame.deep_executed.astype(bool); deep = ~direct
        direct_count += int(direct.sum()); deep_count += int(deep.sum())
        if direct.any():
            require(bool((frame.loc[direct, ["relation_preparation_ms", "level2_ms", "fusion_ms"]] == 0).all().all()), f"direct exit executed deep stage: {path.name}")
        if "__no_routing__" in path.name:
            require(bool(deep.all()), f"no-routing did not execute full deep path: {path.name}")
    require(direct_count > 0 and deep_count > 0, "traces do not cover both direct and deep routes")
    risk = pd.read_csv(root / "risk_control/temporal_risk_audit.csv")
    risk_required = {"chain", "N_calibration", "N_test", "N_test_fraud", "target_alpha", "observed_FNR", "exact_upper_bound", "coverage", "deep_rate", "undefined_reason"}
    require(risk_required.issubset(risk.columns), "risk audit schema incomplete")
    zero = risk.N_test_fraud == 0
    if zero.any(): require(bool(risk.loc[zero, "observed_FNR"].isna().all()), "zero-positive risk metric was fabricated")
    require(bool((risk.theorem_assumptions_verified == False).all()), "distribution-free assumptions mislabeled verified")  # noqa: E712
    cross = pd.read_csv(root / "canonical/canonical_cross_chain.csv")
    if "n_fraud" in cross:
        zero_cross = cross.n_fraud.fillna(-1) == 0
        require(bool(cross.loc[zero_cross, "roc_auc"].isna().all()), "zero-positive cross-chain ROC-AUC was fabricated")
    integrated = pd.read_csv(root / "integrated/table_integrated_100k.csv")
    require(int(integrated.events.iloc[0]) >= 100000, "integrated replay is below 100k events")
    require(int(integrated.event_loss_count.iloc[0]) == 0, "integrated replay lost events")
    require(bool(integrated.checkpoint_restart_verified.iloc[0]), "checkpoint state round-trip failed")
    require(int(integrated.checkpoint_prediction_disagreement.iloc[0]) == 0, "checkpoint prediction disagreement")
    require(int(integrated.oom_failures.iloc[0]) == 0, "integrated replay OOM")
    require(integrated.measurement_type.iloc[0] == "measured_e2e", "integrated cost provenance is not measured_e2e")
    cascade = pd.read_csv(root / "baselines/table_cascade_accuracy_cost.csv")
    runtime_rows = cascade[cascade.measurement_type_runtime.notna()]
    require(bool((runtime_rows.measurement_type_runtime == "measured_e2e").all()), "cascade table mixes cost provenance")
    require({"accuracy_measurement_unit", "runtime_measurement_unit"}.issubset(cascade.columns), "cascade measurement units are ambiguous")
    manifest = json.loads((root / "revision_manifest.json").read_text(encoding="utf-8")); require(bool(manifest.get("claims")), "claim manifest empty")
    for claim in manifest.get("claims", []):
        for key, hash_key in (("generated_table_or_figure", "generated_sha256"), ("canonical_row_source", "canonical_sha256"), ("raw_trace_or_prediction", "raw_sha256")):
            path = repo / claim[key]; require(path.exists(), f"claim artifact missing: {path}")
            if path.exists(): require(sha256_file(path) == claim[hash_key], f"claim hash mismatch: {path}")
        require(bool(claim.get("config_hash")), "claim missing config hash"); require(bool(claim.get("code_git_sha")), "claim missing git SHA")
    payload = {"status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": warnings,
        "checks": {"runtime_rows": len(runtime), "raw_traces": len(traces), "direct_events": direct_count, "deep_events": deep_count,
            "risk_rows": len(risk), "integrated_events": int(integrated.events.iloc[0]), "claims": len(manifest.get("claims", []))}}
    atomic_json(root / "submission_validation.json", payload)
    if errors: raise SubmissionError("; ".join(errors))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo", default="."); args = parser.parse_args()
    try: payload = run(Path(args.repo).resolve())
    except SubmissionError as error: print(json.dumps({"status": "FAIL", "error": str(error)})); return 1
    print(json.dumps(payload, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
