#!/usr/bin/env python3
"""One-command fail-closed SCI-v3 submission evidence build."""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-experiments", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    commands: list[list[str]] = []
    if not args.skip_experiments:
        commands.append([sys.executable, "src/analysis/run_production_path_closure.py"])
        profiler = [sys.executable, "src/profiling/raw_event_selective_e2e_profiler.py"]
        if args.smoke: profiler.append("--smoke")
        commands.append(profiler)
    commands.extend([
        [sys.executable, "src/export/build_sci_v3_submission.py"],
        [sys.executable, "src/validation/validate_sci_v3_submission.py"],
    ])
    for command in commands: subprocess.run(command, check=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
