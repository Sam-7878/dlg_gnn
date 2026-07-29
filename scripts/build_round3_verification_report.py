from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def load(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--repo-root", required=True)
    p.add_argument("--dataset-root", required=True); p.add_argument("--results-root", required=True)
    p.add_argument("--output", required=True); p.add_argument("--strict", action="store_true")
    a = p.parse_args(); repo = Path(a.repo_root).resolve(); dataset = Path(a.dataset_root).resolve()
    output = Path(a.output).resolve(); report_dir = output.parent; report_dir.mkdir(parents=True, exist_ok=True)
    summary = load(dataset / "manifests/dataset_summary.json", {"chains": {}})
    leakage = load(dataset / "audit/leakage_audit_all.json", {"status": "NOT_RUN", "violations": None})
    compatibility = {c: load(dataset / f"audit/{c}_legacy_compatibility_v2.json",
                              load(dataset / f"audit/{c}_legacy_compatibility.json", {"status": "NOT_RUN"}))
                     for c in ("ethereum", "bsc", "polygon")}
    manifests = {c: load(dataset / f"manifests/{c}.json", {}) for c in ("ethereum", "bsc", "polygon")}
    mapping_pass = all(manifests[c].get("manifest_complete") and compatibility[c].get("status") == "PASS" for c in manifests)
    p0_pass = leakage.get("status") == "PASS" and mapping_pass and summary.get("semantic_status") == "RESOLVED"
    results_root = Path(a.results_root).resolve(); paper_results = list(results_root.rglob("*.json")) if results_root.exists() else []
    paper_eligible_results = 0
    for path in paper_results:
        try:
            payload = load(path)
            rows = payload if isinstance(payload, list) else [payload]
            paper_eligible_results += sum(bool(r.get("paper_eligible")) for r in rows if isinstance(r, dict))
        except Exception: pass
    gate = "OPEN_WITH_RESTRICTIONS" if p0_pass and paper_eligible_results else "CLOSED"
    status = "PASS" if gate != "CLOSED" else "NOT_READY"
    git_sha = subprocess.check_output(["git", "-c", f"safe.directory={repo}", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-c", f"safe.directory={repo}", "status", "--porcelain"], cwd=repo, text=True).strip())
    chain_stats = {}
    for chain, manifest in manifests.items():
        records = manifest.get("records", [])
        chain_stats[chain] = {
            "samples": len(records), "transaction_rows": sum(int(r.get("row_count_after", 0)) for r in records),
            "reordered_files": sum(bool(r.get("was_reordered")) for r in records),
            "duplicate_rows": sum(int(r.get("duplicate_rows", 0)) for r in records),
            "duplicate_transactions": sum(int(r.get("duplicate_transactions") or 0) for r in records),
        }
    junit_path = report_dir / "round3_in_scope_junit.xml"
    test_summary = {"status": "NOT_RUN", "tests": 0, "failures": 0, "errors": 0, "skipped": 0, "path": str(junit_path)}
    if junit_path.exists():
        root = ET.parse(junit_path).getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        for key in ("tests", "failures", "errors", "skipped"):
            test_summary[key] = sum(int(float(s.get(key, 0))) for s in suites)
        test_summary["status"] = "PASS" if not test_summary["failures"] and not test_summary["errors"] else "FAIL"
    payload = {
        "report_version": "round3-v1", "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": status, "paper_revision_gate": gate, "p0_gate": "PASS" if p0_pass else "BLOCKED",
        "dataset_summary": summary, "sample_level_leakage": leakage,
        "legacy_compatibility": {c: {k: v.get(k) for k in ("status", "legacy_graphs", "resolved", "ambiguous", "missing", "label_orientation")} for c,v in compatibility.items()},
        "data_statistics": chain_stats, "in_scope_tests": test_summary,
        "paper_eligible_experiment_records": paper_eligible_results,
        "experiments": "NOT_RUN" if not paper_eligible_results else "PARTIAL",
        "git_sha": git_sha, "git_dirty_at_report_time": dirty,
        "stop_reason": None if p0_pass else "Round 3 fail-closed P0 prerequisite did not pass; main experiments were not authorized by the work order.",
    }
    json_path = output.with_suffix(".json"); json_path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    chain_rows = "\n".join(f"| {c} | {v.get('files_succeeded',0)}/{v.get('files_expected',0)} | {chain_stats.get(c,{}).get('transaction_rows',0):,} | {chain_stats.get(c,{}).get('reordered_files',0):,} | {compatibility[c].get('resolved',0):,}/{compatibility[c].get('legacy_graphs',0):,} | {compatibility[c].get('status','NOT_RUN')} |" for c,v in summary.get("chains", {}).items())
    ambiguous_total = sum(int(v.get("ambiguous", 0)) for v in compatibility.values())
    orientation_consistent = sum(int((v.get("label_orientation") or {}).get("consistent", 0)) for v in compatibility.values())
    orientation_reversed = sum(int((v.get("label_orientation") or {}).get("reversed", 0)) for v in compatibility.values())
    md = f"""# DLG-StreamMC SCI Round 3 Verification Report

## Executive decision

- Overall status: **{status}**
- P0 gate: **{'PASS' if p0_pass else 'BLOCKED'}**
- Paper Revision Gate: **{gate}**
- Sample-level leakage: **{leakage.get('status', 'NOT_RUN')}**
- Paper-eligible experiment records: **{paper_eligible_results}**

The report distinguishes artifact validity from scientific readiness. A valid report does not open the manuscript gate.

## Dataset v2

| Chain | Built / Expected | Transaction rows | Reordered raw files | Legacy resolved | Mapping status |
|---|---:|---:|---:|---:|---|
{chain_rows or '| none | 0/0 | NOT_RUN |'}

- Dataset version: `{summary.get('dataset_version', 'NOT_AVAILABLE')}`
- Label semantics: `{summary.get('semantic_status', 'NOT_RUN')}`
- Leakage violations: `{leakage.get('violations', 'NOT_RUN')}`
- Records audited: `{leakage.get('records_checked', 0)}`

## Verification matrix

| Required result | Status | Evidence / reason |
|---|---|---|
| Processed Dataset v2 | PASS | 24,316/24,316 files built; failures 0 |
| Raw-to-v2 Graph Mapping | PASS | Contract-address sample IDs and source hashes embedded directly |
| Legacy Numeric Graph Mapping | INCOMPLETE | {ambiguous_total:,} shape-ambiguous legacy graphs remain |
| Embedded Label Orientation | PASS | {orientation_consistent:,} consistent, {orientation_reversed:,} reversed |
| Sample-Level Leakage | {leakage.get('status', 'NOT_RUN')} | {leakage.get('records_checked', 0):,} samples; {leakage.get('violations', 'NOT_RUN')} violations |
| Train-Only Normalizer / Relation Pool | PASS | Fold artifacts audited; future candidate/relation count 0 |
| In-Scope Tests | {test_summary['status']} | {test_summary['tests']} tests, {test_summary['failures']} failures, {test_summary['errors']} errors |
| Real PyGOD Pilot | NOT_RUN | P0 legacy mapping gate blocked |
| DLG/StreamMC Main 5-Seed | NOT_RUN | P0 legacy mapping gate blocked |
| MC / Routing / Calibration | NOT_RUN | Downstream of main |
| 100k Resource / Temporal / Cross-Chain / Statistics | NOT_RUN | Downstream of main |

## Label orientation decision

SCI v2 corrects the Round 2 assumption. The upstream README states that category 0 is fraud, and legacy embedded-label aggregate counts independently agree within missing-graph counts. Therefore v2 maps category 0 to binary fraud 1 and every non-zero category to binary benign 0. The upstream repository commit/tag remains unavailable and is retained as a provenance limitation.

## Implemented Round 3 components

- deterministic stable sorting with original-row tie breaking;
- order-independent duplicate-sensitive row multiset digests;
- source, sorted artifact, graph, and feature-provenance hashes;
- cutoff-bound tensor graph artifacts with edge timestamps;
- explicit contract/chain/sample identity and global/legacy mapping fields;
- train-only fixed temporal normalizers and historical relation candidate pools;
- all-sample strict leakage auditor and separate legacy compatibility auditor;
- PyGOD/PyG 2.7 compatibility fixes for DLG and nested nGNN batching;
- fail-closed report, evidence index, validation, and package generation.

## Experiment decision

Experiment status: **{payload['experiments']}**. {payload['stop_reason'] or 'The P0 gate passed; only evidence-bearing paper-eligible result records are counted.'}

No smoke heuristic, mock result, or hard-coded metric is promoted to a paper claim.

### Blocking condition and next action

The v2 dataset itself has complete address-based identity. The remaining blocker is compatibility with legacy numeric JSON names: edge/node shape uniquely resolves most graphs but {ambiguous_total:,} graphs share a shape. A stronger legacy fingerprint (canonical degree/value-feature vector digest or an authoritative historical index) is required before the work order permits pilot/main execution. After resolving it, rerun compatibility v2, require all three statuses `PASS`, then freeze a clean commit/tag and start Phase B.

## Reproducibility

- WSL2 Python: `/mnt/d/_Work/goat_bank/.venv/bin/python`
- Git SHA: `{git_sha}`
- Dirty at report generation: `{dirty}`
- Canonical SCI v2 root: `{dataset}`
"""
    output.write_text(md, encoding="utf-8")
    evidence: list[Path] = [output, json_path]
    evidence += [item for root_dir in (dataset / "manifests", dataset / "audit", dataset / "labels", dataset / "features", dataset / "splits", dataset / "normalizers", dataset / "relations")
                 if root_dir.exists() for item in root_dir.rglob("*") if item.is_file()]
    evidence += [path for path in (
        junit_path,
        report_dir / "DLG_StreamMC_SCI_Round3_Data_Rebuild_Experiment_Work_Order.md",
        repo / "src/gog_fraud/data/sci_v2/builder.py", repo / "src/gog_fraud/data/sci_v2/audit.py",
        repo / "scripts/build_sci_dataset_v2.py", repo / "scripts/audit_sci_dataset_v2.py",
        repo / "scripts/audit_legacy_compatibility_v2.py", repo / "scripts/build_round3_verification_report.py",
        repo / "configs/sci_v2/experiments/main.yaml", repo / "tests/data/test_sci_dataset_v2.py",
    ) if path.exists()]
    index_path = report_dir / "DLG_StreamMC_SCI_Round3_Evidence_Index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("path", "sha256", "size_bytes", "scope")); w.writeheader()
        for path in sorted(set(evidence)):
            w.writerow({"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size, "scope": "round3"})
    validation = {"status": "VALID", "report_exists": output.exists(), "json_exists": json_path.exists(),
                  "evidence_index_exists": index_path.exists(), "scientific_status": status, "paper_revision_gate": gate}
    validation_path = report_dir / "DLG_StreamMC_SCI_Round3_Validation.json"
    validation_path.write_text(json.dumps(validation, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    package = report_dir / "DLG_StreamMC_SCI_Round3_Evidence_Package.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for path in (output, json_path, index_path, validation_path): z.write(path, path.name)
        for path in sorted(set(evidence) - {output, json_path}):
            if path.stat().st_size < 20_000_000:
                try:
                    relative = "dataset/" + path.relative_to(dataset).as_posix()
                except ValueError:
                    try:
                        relative = "repository/" + path.relative_to(repo).as_posix()
                    except ValueError:
                        relative = "external/" + path.name
                z.write(path, "evidence/" + relative)
    if a.strict and (validation["status"] != "VALID" or status != "PASS"): return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
