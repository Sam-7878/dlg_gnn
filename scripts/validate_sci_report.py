from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gog_fraud.reporting.validator import validate_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = Path(args.report)
    result = validate_report(report_path=report, json_path=args.report_json or report.with_suffix(".json"), evidence_index_path=args.evidence_index, repo_root=args.repo_root)
    print(json.dumps(result, indent=2))
    return 1 if args.strict and result["status"] != "VALID" else 0


if __name__ == "__main__": raise SystemExit(main())
