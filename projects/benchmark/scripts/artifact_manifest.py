#!/usr/bin/env python3
"""Write or verify the curated A07 paper tree, excluding LaTeX build caches."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "projects/benchmark/paper/current"
MANIFEST = PAPER / "ARTIFACT_MANIFEST.json"
INCLUDE = {".tex", ".bib", ".pdf", ".csv", ".eps", ".sty", ".bst", ".png"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write() -> None:
    files = []
    for path in sorted(PAPER.rglob("*")):
        if path.is_file() and path.suffix.lower() in INCLUDE:
            files.append({"path": path.relative_to(ROOT).as_posix(),
                          "bytes": path.stat().st_size, "sha256": digest(path)})
    MANIFEST.write_text(json.dumps({"schema_version": 1, "role": "A07 candidate paper; not a public release",
                                    "files": files}, indent=2) + "\n")
    print(f"WROTE {len(files)} curated paper hashes")


def verify() -> None:
    manifest = json.loads(MANIFEST.read_text())
    expected = {item["path"]: item for item in manifest["files"]}
    actual = {path.relative_to(ROOT).as_posix() for path in PAPER.rglob("*")
              if path.is_file() and path.suffix.lower() in INCLUDE}
    if set(expected) != actual:
        raise RuntimeError(f"paper file set drift: missing={sorted(set(expected)-actual)}, extra={sorted(actual-set(expected))}")
    for relative, item in expected.items():
        path = ROOT / relative
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise RuntimeError(f"paper artifact drift: {relative}")
    print(f"PASS {len(expected)} curated paper hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write", action="store_true", help="refresh the manifest after a reviewed paper rebuild")
    actions.add_argument("--verify", action="store_true", help="check the committed paper artifacts")
    args = parser.parse_args()
    write() if args.write else verify()
