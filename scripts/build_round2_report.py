from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gog_fraud.reporting.collector import collect_evidence, git_metadata
from gog_fraud.reporting.evidence_index import write_evidence_index


def _read(path: Path, default):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return default


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", default="."); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); root = Path(args.repo_root).resolve(); output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    chains = ("ethereum", "bsc", "polygon")
    manifests = {chain: _read(root / f"results_sci/manifests/{chain}.json", {}) for chain in chains}
    audits = {chain: _read(root / f"results_sci/splits/{chain}_leakage_audit_v1.json", {}) for chain in chains}
    smoke = _read(root / "results_sci/streaming/smoke_round2/smoke_summary.json", [])
    in_scope_tests = _read(output / "test_summaries/round2_in_scope_summary.json", {"status": "NOT_RUN"})
    repository_tests = _read(output / "test_summaries/round2_repository_summary.json", {"status": "NOT_RUN"})
    git = git_metadata(root)
    run_manifest_paths = sorted(
        (root / "results_sci/manifests").glob("*/run_manifest.json"),
        key=lambda path: path.stat().st_mtime,
    )
    strict_run = _read(run_manifest_paths[-1], {}) if run_manifest_paths else {}
    try:
        tagged_sha = subprocess.check_output(
            ["git", "rev-list", "-n", "1", "dlg-streammc-sci-v1.0"],
            cwd=root, text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        tagged_sha = ""
    clean_tagged_execution = bool(
        tagged_sha
        and strict_run.get("git_dirty") is False
        and strict_run.get("git_sha") == tagged_sha
        and "--require-clean-git" in strict_run.get("cli_args", [])
    )
    full_manifests = len(manifests) == 3 and all(row.get("manifest_complete") and row.get("files_failed") == 0 and row.get("hash_verification") == "full" for row in manifests.values())
    split_exists = all((root / f"results_sci/splits/{chain}_holdout_v1.json").is_file() for chain in chains)
    leakage_pass = len(audits) == 3 and all(row.get("status") == "PASS" and row.get("violations") == 0 for row in audits.values())
    result_rows = sum(1 for path in (root / "results_sci").rglob("*.csv") if path.parent.name not in {"manifests", "splits"}) if (root / "results_sci").exists() else 0
    blockers = []
    if not full_manifests: blockers.append("Three-chain full manifest gate is not satisfied.")
    if not leakage_pass: blockers.append("Processed-feature sample-level leakage audit is not PASS.")
    if not clean_tagged_execution: blockers.append("No strict execution from the clean tagged code baseline was captured.")
    if not result_rows: blockers.append("No paper-eligible baseline/main/MC/calibration/resource/temporal/cross-chain result rows exist.")
    blockers.append("The legacy run_baseline_benchmark path contains hard-coded demonstration metrics and is now guarded as non-paper-eligible; real PyGOD runs remain required.")
    gate = {
        "three_chain_manifest": full_manifests, "fixed_split_manifests": split_exists,
        "sample_level_leakage_pass": leakage_pass, "dependency_lock": (root / "requirements-sci-lock.txt").is_file(),
        "in_scope_tests_pass": in_scope_tests.get("status") == "PASS", "clean_tagged_code": clean_tagged_execution,
        "main_baseline_5_seed": False, "mc_sensitivity": False, "calibration": False, "streaming_100k": False,
        "temporal_robustness": False, "held_out_cross_chain": False, "statistical_tests": False,
    }
    phases = {
        "P0_data_manifest": "PASS" if full_manifests else "INCOMPLETE",
        "P0_temporal_split": "PASS" if split_exists else "NOT_RUN",
        "P0_sample_leakage": "PASS" if leakage_pass else "INCOMPLETE",
        "Phase_A_smoke": "PASS_NOT_PAPER_ELIGIBLE" if len(smoke) == 3 and all(row.get("status") == "PASS" for row in smoke) else "NOT_RUN",
        "Phase_B_pilot": "NOT_RUN_BLOCKED_BY_LEAKAGE_GATE", "Phase_C_main": "NOT_RUN_BLOCKED_BY_LEAKAGE_GATE",
        "Phase_D_mc": "NOT_RUN", "Phase_E_routing": "NOT_RUN", "Phase_F_calibration": "NOT_RUN",
        "Phase_G_streaming_resource": "NOT_RUN", "Phase_H_temporal": "NOT_RUN", "Phase_I_cross_chain": "NOT_RUN", "Phase_J_statistics": "NOT_RUN",
    }
    model = {
        "generated_at": datetime.now(timezone.utc).isoformat(), "overall_status": "NOT_READY",
        "paper_revision_gate": "CLOSED", "git": git, "manifests": manifests, "leakage_audits": audits,
        "smoke": smoke, "tests": {"in_scope": in_scope_tests, "repository_wide": repository_tests},
        "strict_orchestrator_run": strict_run, "tagged_sha": tagged_sha,
        "gate": gate, "phases": phases, "paper_eligible_result_files": result_rows, "blockers": blockers,
    }
    json_path = output / "DLG_StreamMC_SCI_Round2_Development_Experiment_Report.json"
    json_path.write_text(json.dumps(model, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_rows = "\n".join(
        f"| {chain} | {row.get('manifest_complete', 'MISSING')} | {row.get('files_discovered', 'MISSING')} | {row.get('transactions', 'MISSING')} | {row.get('fraud', 'MISSING')} | {row.get('benign', 'MISSING')} | {row.get('files_failed', 'MISSING')} | {row.get('hash_verification', 'MISSING')} |"
        for chain, row in manifests.items()
    )
    gate_rows = "\n".join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gate.items())
    phase_rows = "\n".join(f"| {name} | {status} |" for name, status in phases.items())
    report = f"""# DLG-StreamMC SCI Round 2 Development and Experiment Report

**Overall:** `NOT_READY`

**Paper revision gate:** `CLOSED`
**Generated:** `{model['generated_at']}`

## 1. Outcome

Round 2 development infrastructure and a three-chain 100-sample smoke path were implemented. Smoke outputs are explicitly non-model and not paper-eligible. Main experiments were not started because the fail-closed sample-level leakage gate is not satisfied by the legacy processed graph artifacts.

## 2. Dataset Manifest Audit

| Chain | Complete | Files | Transactions | Fraud | Benign | Failed | Hash verification |
|---|---:|---:|---:|---:|---:|---:|---|
{manifest_rows}

Canonical root is `/mnt/d/_Work/_data/GoG`; raw physical input is `/mnt/d/_Work/_data/dataset/transactions`. They are preprocessing-derived datasets, not symlinks or byte-identical copies.

## 3. Temporal Split and Leakage Audit

Fixed holdout and rolling-origin artifacts: `{'present' if split_exists else 'missing'}`. Raw event-time ordering audit: `{', '.join(f'{chain}={row.get("raw_event_time_audit", "MISSING")}' for chain, row in audits.items()) or 'NOT_RUN'}`. Full processed-feature leakage status: `{', '.join(f'{chain}={row.get("status", "MISSING")}' for chain, row in audits.items()) or 'NOT_RUN'}`.

Raw ordering violations: `{', '.join(f'{chain}={row.get("violations", "MISSING")}' for chain, row in audits.items()) or 'NOT_RUN'}`. Legacy graph JSON also does not contain feature-source timestamps, normalizer fit intervals, relation construction intervals, or KNN candidate-pool provenance. Either condition prevents a full leakage `PASS`.

## 4. Development Completed

- Metadata-guarded hash cache using absolute path, size, mtime_ns, and inode.
- Explicit complete/truncated/failed/hash-verification manifest fields.
- Canonical dataset and label provenance audit tooling.
- Immutable fixed and rolling split artifact generator.
- Fail-closed, resumable SCI orchestrator with resolved config and run manifest.
- CPU/CUDA synchronized latency, RSS/VRAM, component timing, trace, and memory-slope profiler.
- Evidence test allowlist excluding llama, micro_rag, and unrelated mocks.
- Legacy hard-coded baseline and Markdown/LaTeX scientific-claim generators are fail-closed.
- Calibration, reliability-source, paired-bootstrap, effect-size, and Holm correction utilities.
- Versioned SCI data/model/routing/streaming/hardware/experiment config snapshots.

## 5. Smoke Verification

`{len(smoke)}` chains completed. Each uses 100 real dataset contracts with a deterministic non-label heuristic solely to exercise streaming, routing, trace, profiler, and checkpoint paths. `paper_metrics_generated=false` for every chain.

## 6. Test Scope

- In-scope: `{in_scope_tests.get('status', 'NOT_RUN')}`, passed `{in_scope_tests.get('passed', 0)}`.
- Repository-wide: `{repository_tests.get('status', 'NOT_RUN')}`, passed `{repository_tests.get('passed', 0)}`.
- Excluded from SCI evidence: `tests/llama/**`, `tests/micro_rag/**`, `tests/mock/**`, and non-allowlisted unit/mock tests.

## 7. Phase Status

| Phase | Status |
|---|---|
{phase_rows}

Strict orchestrator evidence: git SHA `{strict_run.get('git_sha', 'NOT_RUN')}`, git dirty `{strict_run.get('git_dirty', 'NOT_RUN')}`, status `{strict_run.get('status', 'NOT_RUN')}`. The failed status is expected fail-closed behavior caused by the leakage gate and absence of configured real paper stages; it is not counted as a completed experiment.

## 8. Paper Revision Gate

| Gate | Result |
|---|---|
{gate_rows}

## 9. Blocking Issues

{chr(10).join(f'- {item}' for item in blockers)}

## 10. Scientific Interpretation

No main detection, baseline fairness, MC sensitivity, calibration, 100k-event resource, temporal robustness, cross-chain, confidence interval, or significance result is claimed. The correct scientific state remains `NOT_READY`; manuscript quantitative revision must not begin.
"""
    report_path = output / "DLG_StreamMC_SCI_Round2_Development_Experiment_Report.md"
    report_path.write_text(report, encoding="utf-8")
    records = [record for record in collect_evidence(root) if not record.path.endswith((report_path.name, json_path.name, "DLG_StreamMC_SCI_Round2_Evidence_Index.csv"))]
    write_evidence_index(records, output / "DLG_StreamMC_SCI_Round2_Evidence_Index.csv")


if __name__ == "__main__": main()
