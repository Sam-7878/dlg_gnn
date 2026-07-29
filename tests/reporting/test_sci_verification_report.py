import csv
import json
from pathlib import Path

from gog_fraud.reporting.evidence_index import write_evidence_index
from gog_fraud.reporting.report_renderer import build_report_model, render_markdown
from gog_fraud.reporting.schema import EvidenceRecord, REPORT_TOP_LEVEL_FIELDS
from gog_fraud.reporting.validator import validate_report
from gog_fraud.reporting.collector import collect_evidence


def _fixture_repo(root: Path):
    (root / "configs/sci").mkdir(parents=True)
    (root / "configs/sci/main.yaml").write_text("experiment:\n  seeds: [42]\n", encoding="utf-8")
    (root / "docs/work_reports/100_stream_mc_update/artifacts/dataset_manifests").mkdir(parents=True)
    manifest = {"chain": "ethereum", "source_root": "/data", "transactions": 10, "contracts": 2, "addresses": 3, "fraud": 1, "benign": 1, "unlabeled": 0, "positive_ratio": 0.5, "missing_timestamp": 0, "duplicates": 0, "file_hashes": {"a.csv": "abc"}}
    (root / "docs/work_reports/100_stream_mc_update/artifacts/dataset_manifests/ethereum.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_model_has_all_sections_and_marks_missing_results(tmp_path):
    _fixture_repo(tmp_path)
    generated_dir = tmp_path / "docs/work_reports/101_stream_mc_check_result"
    generated_dir.mkdir(parents=True)
    (generated_dir / "DLG_StreamMC_SCI_Report_Validation.json").write_text("{}", encoding="utf-8")
    model = build_report_model(tmp_path, test_summary={"status": "PASS", "passed": 85, "failed": 0, "warnings": 1})
    assert set(REPORT_TOP_LEVEL_FIELDS) <= set(model)
    assert model["main_results"]["status"] == "NOT_RUN"
    assert model["executive_summary"]["overall_status"] == "NOT_READY"
    assert not any(row["path"].endswith("DLG_StreamMC_SCI_Report_Validation.json") for row in model["evidence_index"])
    markdown = render_markdown(model)
    assert "## 23. SCI Submission Readiness" in markdown
    assert "## Appendix H." in markdown


def test_validator_detects_evidence_hash_mismatch(tmp_path):
    _fixture_repo(tmp_path)
    model = build_report_model(tmp_path)
    report = tmp_path / "report.md"; report.write_text(render_markdown(model), encoding="utf-8")
    report_json = tmp_path / "report.json"; report_json.write_text(json.dumps(model), encoding="utf-8")
    rows = [EvidenceRecord(**row) for row in model["evidence_index"]]
    index = tmp_path / "evidence.csv"; write_evidence_index(rows, index)
    assert validate_report(report_path=report, json_path=report_json, evidence_index_path=index, repo_root=tmp_path)["status"] == "VALID"
    source = tmp_path / rows[0].path; source.write_text("changed", encoding="utf-8")
    result = validate_report(report_path=report, json_path=report_json, evidence_index_path=index, repo_root=tmp_path)
    assert result["status"] == "INVALID"
    assert any("hash mismatch" in error for error in result["errors"])


def test_evidence_collector_excludes_unrelated_tests(tmp_path):
    for relative in (
        "tests/streaming/test_engine.py",
        "tests/llama/test_graphrag.py",
        "tests/micro_rag/run_real_pipeline.py",
        "tests/unit/test_mc_dropout.py",
        "tests/unit/test_unrelated_mock.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def test_placeholder(): pass\n", encoding="utf-8")
    paths = {record.path for record in collect_evidence(tmp_path)}
    assert "tests/streaming/test_engine.py" in paths
    assert "tests/unit/test_mc_dropout.py" in paths
    assert "tests/llama/test_graphrag.py" not in paths
    assert "tests/micro_rag/run_real_pipeline.py" not in paths
    assert "tests/unit/test_unrelated_mock.py" not in paths
