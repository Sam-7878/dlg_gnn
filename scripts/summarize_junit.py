from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree


def warning_count(log_path: str | None) -> int:
    if not log_path:
        return 0
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"(\d+) warnings?", text)
    return int(matches[-1]) if matches else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--junit", required=True)
    parser.add_argument("--log")
    parser.add_argument("--output", required=True)
    parser.add_argument("--command", required=True)
    args = parser.parse_args()
    root = ElementTree.parse(args.junit).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.get("skipped", 0)) for suite in suites)
    elapsed = sum(float(suite.get("time", 0.0)) for suite in suites)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "command": args.command,
        "status": "PASS" if failures + errors == 0 else "FAIL",
        "passed": total - failures - errors - skipped,
        "failed": failures + errors,
        "skipped": skipped,
        "warnings": warning_count(args.log),
        "tests": total,
        "elapsed_s": elapsed,
        "junit": str(Path(args.junit)),
    }
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
