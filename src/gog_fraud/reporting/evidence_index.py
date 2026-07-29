from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Iterable

from .schema import EvidenceRecord


EVIDENCE_FIELDS = tuple(EvidenceRecord.__dataclass_fields__)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evidence_index(records: Iterable[EvidenceRecord], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVIDENCE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_dict())


def read_evidence_index(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def verify_evidence_index(path: str | Path, repo_root: str | Path) -> list[str]:
    root = Path(repo_root).resolve()
    errors: list[str] = []
    seen: set[str] = set()
    for row in read_evidence_index(path):
        evidence_id = row["evidence_id"]
        if evidence_id in seen:
            errors.append(f"duplicate evidence_id: {evidence_id}")
        seen.add(evidence_id)
        source = root / row["path"]
        if not source.is_file():
            errors.append(f"missing evidence: {row['path']}")
        elif sha256_file(source) != row["sha256"]:
            errors.append(f"hash mismatch: {row['path']}")
    return errors
