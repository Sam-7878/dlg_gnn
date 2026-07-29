from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gog_fraud.reporting.package_builder import build_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--report", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = Path(args.report)
    output = build_package(repo_root=args.repo_root, report=report, report_json=args.report_json or report.with_suffix(".json"), evidence_index=args.evidence_index, output=args.output)
    print(output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
