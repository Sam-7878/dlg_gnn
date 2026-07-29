"""CLI for the Round 4 SCI-v2 main experiment gate."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gog_fraud.experiments.round4_policy import check_main_prerequisites


def _git_clean(root: Path) -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
        return not status.strip()
    except (OSError, subprocess.CalledProcessError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--dependency-lock", default="requirements-sci-lock.txt")
    parser.add_argument("--output")
    parser.add_argument("--allow-partial-legacy-compatibility", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    lock = Path(args.dependency_lock)
    if not lock.is_absolute():
        lock = repo / lock
    result = check_main_prerequisites(
        args.dataset_root, git_clean_at_start=_git_clean(repo), dependency_lock=lock,
    )
    payload = result.to_dict()
    payload["legacy_is_non_blocking"] = True
    encoded = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    if args.output:
        target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result.authorized else (2 if args.strict else 1)


if __name__ == "__main__":
    raise SystemExit(main())
