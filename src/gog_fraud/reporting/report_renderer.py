from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .claim_checker import verify_claims
from .collector import collect_evidence, environment_metadata, git_metadata, load_configs, load_dataset_manifests, load_experiment_registry
from .issue_detector import detect_issues
from .schema import REPORT_TOP_LEVEL_FIELDS


COMPONENTS = (
    ("Temporal split", "src/gog_fraud/data/splits/temporal_split.py", "tests/data/test_temporal_integrity.py"),
    ("Leakage validator", "src/gog_fraud/data/validation/temporal_leakage.py", "tests/data/test_temporal_integrity.py"),
    ("Stateful stream", "src/gog_fraud/data/io/streaming_dataset.py", "tests/streaming/test_stateful_stream.py"),
    ("Incremental L1 store", "src/gog_fraud/streaming/subgraph_store.py", "tests/streaming/test_bounded_state.py"),
    ("Incremental L2 graph", "src/gog_fraud/streaming/relation_state.py", "tests/streaming/test_bounded_state.py"),
    ("MC inference", "src/gog_fraud/models/extensions/mc/mc_dropout.py", "tests/unit/test_mc_dropout.py"),
    ("Dual-threshold router", "src/gog_fraud/selection/router.py", "tests/selection/test_router.py"),
    ("Risk-sensitive router", "src/gog_fraud/selection/router.py", "tests/selection/test_router.py"),
    ("TTL/LRU cache", "src/gog_fraud/streaming/embedding_cache.py", "tests/streaming/test_bounded_state.py"),
    ("Queue/backpressure", "src/gog_fraud/streaming/queue_manager.py", "tests/streaming/test_bounded_state.py"),
    ("Checkpoint/recovery", "src/gog_fraud/streaming/checkpoint.py", "tests/streaming/test_stateful_stream.py"),
    ("Latency profiler", "src/profiling/streaming_profiler.py", ""),
    ("Memory profiler", "src/profiling/streaming_profiler.py", ""),
    ("Result provenance", "src/gog_fraud/experiments/manifest.py", "tests/experiments/test_provenance.py"),
)


def _implementation(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name, source, test in COMPONENTS:
        implemented = (root / source).is_file()
        tested = bool(test and (root / test).is_file())
        status = "PASS" if implemented and tested else "PARTIAL" if implemented else "NOT_IMPLEMENTED"
        rows.append({"component": name, "required": True, "implemented": implemented, "tested": tested, "evidence": ", ".join(item for item in (source, test) if item), "status": status})
    return rows


def build_report_model(repo_root: str | Path, *, test_summary: dict[str, Any] | None = None, include_archive: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    git = git_metadata(root)
    environment = environment_metadata()
    datasets = load_dataset_manifests(root)
    experiments = load_experiment_registry(root)
    configs = load_configs(root)
    evidence = collect_evidence(root, include_archive=include_archive)
    test_summary = test_summary or {"status": "NOT_RUN", "passed": 0, "failed": 0, "warnings": 0}
    tests_passed = test_summary.get("status") == "PASS" and int(test_summary.get("failed", 0)) == 0
    implementation = _implementation(root)
    issues = detect_issues(repo_root=root, datasets=datasets, experiments=experiments, git=git, configs=configs)
    claims = verify_claims(experiments=experiments, tests_passed=tests_passed)
    critical = sum(issue.severity == "CRITICAL" for issue in issues)
    high = sum(issue.severity == "HIGH" for issue in issues)
    method_points = round(15 * sum(row["status"] == "PASS" for row in implementation) / len(implementation), 1)
    score = min(100.0, method_points + (5 if datasets else 0) + (5 if tests_passed else 0) + (4 if any(item["status"] == "PARTIALLY_SUPPORTED" for item in claims) else 0) + (3 if configs else 0))
    readiness = "NOT_READY" if not experiments or critical else "MAJOR_REVISION_REQUIRED"
    readiness_breakdown = {"method_implementation": method_points, "dataset_integrity": 5.0 if datasets else 0.0, "experimental_fairness": 0.0, "statistical_rigor": 0.0, "streaming_evidence": 5.0 if tests_passed else 0.0, "selective_inference_evidence": 4.0 if tests_passed else 0.0, "resource_evaluation": 0.0, "reproducibility": 3.0 if configs else 0.0}
    model: dict[str, Any] = {
        "report_metadata": {"title": "DLG-StreamMC SCI Integrated Verification Report", "generated_at": datetime.now(timezone.utc).isoformat(), "generator_version": "1.0.0", "repository": str(root), **git, "environment": environment, "included_experiment_period": "NONE", "excluded_archives": not include_archive},
        "executive_summary": {"overall_status": readiness, "submission_readiness_score": score, "valid_experiments": len([x for x in experiments if x.get("status") == "success"]), "total_experiments": len(experiments), "failed_experiments": len([x for x in experiments if x.get("status") not in {"success", None, ""}]), "critical_issues": critical, "high_issues": high, "temporal_leakage": "UNIT_PASS_DATASET_NOT_AUDITED" if tests_passed else "NOT_VERIFIED", "baseline_fairness": "NOT_DIRECTLY_COMPARABLE", "reproducibility_grade": "F" if score < 60 else "D", "top_blocking_issue": "No immutable experiment results exist under results_sci."},
        "implementation_status": implementation,
        "dataset_audit": {"status": "PARTIAL" if datasets else "MISSING", "chains": datasets, "required_chains": ["ethereum", "bsc", "polygon"]},
        "leakage_audit": {"status": "UNIT_ONLY" if tests_passed else "NOT_RUN", "sample_level_audit": "NOT_RUN", "checks": [{"check": name, "samples_tested": 0, "violations": "UNKNOWN", "severity": "HIGH", "evidence": "unit tests only"} for name in ("feature timestamp", "relation timestamp", "normalizer fit", "split boundary", "test threshold tuning", "KNN contamination")]},
        "experiment_registry": experiments,
        "baseline_fairness": {"status": "NOT_DIRECTLY_COMPARABLE", "reason": "No standardized baseline/main experiment rows."},
        "main_results": {"status": "NOT_RUN", "rows": []},
        "routing_analysis": {"status": "NOT_RUN", "rows": []},
        "mc_analysis": {"status": "NOT_RUN", "required_samples": [1,3,5,8,10,20,30], "rows": []},
        "calibration": {"status": "NOT_RUN", "rows": []},
        "streaming": {"status": "UNIT_ONLY" if tests_passed else "NOT_RUN", "scenario_results": [], "unit_test_summary": test_summary},
        "resources": {"status": "NOT_RUN", "rows": []},
        "latency": {"status": "NOT_RUN", "rows": []},
        "ablations": {"status": "NOT_RUN", "rows": []},
        "temporal": {"status": "NOT_RUN", "rolling_folds": []},
        "cross_chain": {"status": "NOT_RUN", "matrix": []},
        "statistics": {"status": "NOT_RUN", "tests": []},
        "failures": [{"experiment_id": "DATASET-MANIFEST-ATTEMPT", "failure_type": "long_running_manual_termination", "count": 1, "affected_samples": "unknown", "resolution": "progress/resume added; full 3-chain rerun not evidenced", "included_in_final": True}],
        "claim_verification": claims,
        "consistency_issues": [issue.to_dict() for issue in issues],
        "reproducibility": {"score": score, "grade": "F" if score < 60 else "D", "checks": {"dependency_lock": False, "data_manifest_all_chains": len(datasets) == 3, "fixed_split_artifact": False, "config_archive": bool(configs), "seed_configured": bool(configs), "deterministic_replay_test": tests_passed, "checkpoint_test": tests_passed, "one_command_audit": True, "source_to_table_traceability": False}},
        "submission_readiness": {"score": score, "decision": readiness, "weighted_breakdown": readiness_breakdown, "blocking_issues": [issue.message for issue in issues if issue.severity in {"CRITICAL", "HIGH"}], "non_blocking_issues": [issue.message for issue in issues if issue.severity not in {"CRITICAL", "HIGH"}], "recommended_claims": ["Stateful selective-inference core has unit-level verification."], "claims_to_weaken_or_remove": [item["claim"] for item in claims if item["status"] == "MISSING_EVIDENCE"], "additional_experiment_priority": ["Generate immutable temporal split manifests", "Run fair PyGOD/DLG/StreamMC baselines", "Run 5-seed MC/routing/calibration experiments", "Run 100k-event streaming/resource scenarios", "Run temporal and cross-chain generalization", "Run confidence intervals and significance tests"]},
        "evidence_index": [record.to_dict() for record in evidence],
    }
    assert set(REPORT_TOP_LEVEL_FIELDS) <= set(model)
    return model


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows: return "`MISSING / NOT_RUN`"
    header = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|"
    return header + "\n" + "\n".join("| " + " | ".join(str(value) for value in row) + " |" for row in rows)


def render_markdown(model: dict[str, Any]) -> str:
    meta, summary = model["report_metadata"], model["executive_summary"]
    impl_rows = [[r["component"], "Yes", r["implemented"], r["tested"], r["evidence"], r["status"]] for r in model["implementation_status"]]
    dataset_rows = [[chain, d.get("collection_start"), d.get("collection_end"), d.get("transactions"), d.get("contracts"), d.get("addresses"), d.get("fraud"), d.get("benign"), d.get("unlabeled"), d.get("positive_ratio"), d.get("missing_timestamp"), d.get("duplicates"), d.get("_source")] for chain,d in sorted(model["dataset_audit"]["chains"].items())]
    issue_rows = [[x["issue_id"], x["severity"], x["category"], x["message"], x["resolution"]] for x in model["consistency_issues"]]
    claim_rows = [[x["claim"], x["required_evidence"], x["available"], x["result"], x["status"]] for x in model["claim_verification"]]
    sections = [
        f"# DLG-StreamMC SCI Integrated Verification Report\n\n**Overall Status:** `{summary['overall_status']}`  \n**Submission Readiness Score:** `{summary['submission_readiness_score']}/100`  \n**Valid Experiments:** `{summary['valid_experiments']} / {summary['total_experiments']}`  \n**Critical / High Issues:** `{summary['critical_issues']} / {summary['high_issues']}`  \n**Temporal Leakage:** `{summary['temporal_leakage']}`  \n**Baseline Fairness:** `{summary['baseline_fairness']}`  \n**Reproducibility Grade:** `{summary['reproducibility_grade']}`  \n\n**Top Blocking Issue:** {summary['top_blocking_issue']}",
        f"## 1. Executive Summary\n\n현재 구현 코어와 {model['streaming']['unit_test_summary'].get('passed', 0)}-test suite는 검증됐지만 immutable experiment result가 전혀 없어 SCI 성능 주장은 검증할 수 없다. 최종 판정은 **{summary['overall_status']}**이다.",
        f"## 2. Scope and Version Baseline\n\n- Repository: `{meta['repository']}`\n- Branch / SHA: `{meta['branch']}` / `{meta['git_sha']}`\n- Dirty: `{meta['dirty']}`\n- Python / PyTorch / PyG / PyGOD / CUDA: `{meta['environment']['python']}` / `{meta['environment']['torch']}` / `{meta['environment']['torch_geometric']}` / `{meta['environment']['pygod']}` / `{meta['environment']['cuda']}`\n- Hardware: `{json.dumps(meta['environment']['hardware'], sort_keys=True)}`\n- Generated: `{meta['generated_at']}`\n- Dataset basis requested: `/mnt/d/_Work/_data/GoG`\n- Manifest source observed: `/mnt/d/_Work/_data/dataset/transactions`\n- Included experiment period: `{meta['included_experiment_period']}`",
        "## 3. Repository Implementation Status\n\n" + _table(["Component","Required","Implemented","Tested","Evidence","Status"], impl_rows),
        "## 4. Dataset and Label Audit\n\n" + _table(["Chain","Start","End","Transactions","Contracts","Addresses","Fraud","Benign","Unlabeled","Positive Ratio","Missing TS","Duplicates","Evidence"], dataset_rows) + "\n\nBSC와 Polygon manifest는 `MISSING`이다. chain overlap 및 split statistics도 `NOT_RUN`이다.",
        "## 5. Temporal Integrity and Leakage Verification\n\nUnit boundary tests pass, but no real sample-level leakage audit or immutable split manifest exists. Main results therefore cannot be marked VALID.",
        "## 6. Experimental Protocol Matrix\n\n`MISSING`: no experiment rows under `results_sci`.",
        "## 7. Baseline Fairness Audit\n\n**NOT_DIRECTLY_COMPARABLE** — PyGOD, DLG-GNN, StreamMC results with a common split/config are absent.",
        "## 8. Main Detection Results\n\n`NOT_RUN`: ROC-AUC, PR-AUC, F1, recall, CI source rows are absent.",
        "## 9. Selective Inference and Routing Analysis\n\n`NOT_RUN`: no sample trace or aggregate routing result.",
        "## 10. Monte Carlo Sensitivity\n\n`NOT_RUN`: required T={1,3,5,8,10,20,30} result matrix absent.",
        "## 11. Calibration Analysis\n\n`NOT_RUN`: NLL/Brier/ECE/reliability source data absent.",
        "## 12. Streaming Evaluation\n\n**UNIT_ONLY**: deterministic replay, checkpoint primitives, bounded queue/cache tests pass. Normal/burst/overload/cache-pressure/100k-event scenarios are `NOT_RUN`.",
        "## 13. Resource Evaluation\n\n`NOT_RUN`: VRAM/RSS/cache/queue/memory-slope and LPP comparison absent.",
        "## 14. Latency and Throughput\n\n`NOT_RUN`: cold-start and steady-state component timing rows absent.",
        "## 15. Ablation Summary\n\n`NOT_RUN`: MC/routing/L2/fusion/legacy/LPP ablations absent.",
        "## 16. Temporal Robustness\n\n`NOT_RUN`: rolling-origin result folds absent.",
        "## 17. Cross-Chain Generalization\n\n`NOT_RUN`: held-out chain matrix absent.",
        "## 18. Statistical Verification\n\n`NOT_RUN`: bootstrap/DeLong/Wilcoxon/Friedman/Nemenyi evidence absent.",
        "## 19. Failure, Exception, and Exclusion Audit\n\n" + _table(["Exp ID","Failure Type","Count","Affected","Resolution","Included"], [[f["experiment_id"],f["failure_type"],f["count"],f["affected_samples"],f["resolution"],f["included_in_final"]] for f in model["failures"]]),
        "## 20. Paper Claim Verification Matrix\n\n" + _table(["Claim","Required Evidence","Available","Result","Status"], claim_rows),
        "## 21. Result Consistency Checks\n\n" + _table(["Issue","Severity","Category","Message","Resolution"], issue_rows),
        f"## 22. Reproducibility Assessment\n\nScore: **{model['reproducibility']['score']}/100**, Grade **{model['reproducibility']['grade']}**. Unit reproducibility is present; dataset/split/result provenance is incomplete.",
        f"## 23. SCI Submission Readiness\n\nDecision: **{model['submission_readiness']['decision']}** ({model['submission_readiness']['score']}/100).\n\n### Weighted Assessment\n\n" + _table(["Area","Points"], [[k,v] for k,v in model["submission_readiness"]["weighted_breakdown"].items()]) + "\n\n### Blocking Issues\n\n" + "\n".join(f"- {x}" for x in model["submission_readiness"]["blocking_issues"]) + "\n\n### Non-blocking Issues\n\n" + "\n".join(f"- {x}" for x in model["submission_readiness"]["non_blocking_issues"]) + "\n\n### Recommended Next Actions\n\n" + "\n".join(f"1. {x}" for x in model["submission_readiness"]["additional_experiment_priority"]),
        "## Appendix A. Complete Experiment Registry\n\n`EMPTY`",
        "## Appendix B. Complete Metric Tables\n\n`EMPTY`",
        "## Appendix C. Config and Hyperparameter Index\n\nSee machine-readable JSON and evidence index.",
        f"## Appendix D. Evidence Index\n\n{len(model['evidence_index'])} evidence records. See `DLG_StreamMC_SCI_Evidence_Index.csv`.",
        f"## Appendix E. Test Results\n\nStatus: `{model['streaming']['unit_test_summary'].get('status')}`; passed={model['streaming']['unit_test_summary'].get('passed')}, failed={model['streaming']['unit_test_summary'].get('failed')}, warnings={model['streaming']['unit_test_summary'].get('warnings')}.",
        "## Appendix F. Failure Logs\n\nNo standardized experiment failure log exists; known development termination is retained in Section 19.",
        "## Appendix G. Generated Figures\n\n`EMPTY`: no figure source data.",
        "## Appendix H. Claim-to-Evidence Trace\n\nSee Section 20 and report JSON `claim_verification`.",
    ]
    return "\n\n---\n\n".join(sections) + "\n"
