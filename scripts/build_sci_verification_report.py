from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gog_fraud.reporting.evidence_index import write_evidence_index
from gog_fraud.reporting.report_renderer import build_report_model, render_markdown
from gog_fraud.reporting.validator import validate_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--results-root")
    parser.add_argument("--configs-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-summary")
    parser.add_argument("--include-archive", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    root, output = Path(args.repo_root).resolve(), Path(args.output).resolve()
    test_summary = json.loads(Path(args.test_summary).read_text(encoding="utf-8")) if args.test_summary else None
    model = build_report_model(root, test_summary=test_summary, include_archive=args.include_archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    evidence_path = output.parent / "DLG_StreamMC_SCI_Evidence_Index.csv"
    output.write_text(render_markdown(model), encoding="utf-8")
    json_path.write_text(json.dumps(model, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    from gog_fraud.reporting.schema import EvidenceRecord
    write_evidence_index((EvidenceRecord(**row) for row in model["evidence_index"]), evidence_path)
    result = validate_report(report_path=output, json_path=json_path, evidence_index_path=evidence_path, repo_root=root)
    validation_path = output.parent / "DLG_StreamMC_SCI_Report_Validation.json"
    validation_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 1 if args.strict and result["status"] != "VALID" else 0


if __name__ == "__main__": raise SystemExit(main())
