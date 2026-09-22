"""Audit and safely unpack the public GoG source distribution."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from experiments.round7.provenance import sha256_file


DRIVE_FOLDER_ID = "1VV5ht9Eh8WGtKfkS0ipIk0FNI7g-WJfJ"
UPSTREAM_COMMIT = "7264f1bf510f7ba4f5041ac7a29b606abc12f262"
UPSTREAM_FILES = {
    "global_graph/bsc_contract_to_number_mapping.json": ("1ipgXQB57z5Rp413Zqdipac7cGTBpYxLR", 345_834),
    "global_graph/bsc_graph_more_than_1_ratio.csv": ("14BfGzkOJEme3fWo0-Y9cmxjl1MCcMkY7", 15_344_641),
    "global_graph/ethereum_contract_to_number_mapping.json": ("1r7z5ffYIk9PXD8lJjnc5UUImY895App5", 697_712),
    "global_graph/ethereum_graph_more_than_1_ratio.csv": ("1cTx1m4F67hyBTJgcEKbjlpOmBsd8SFQ_", 46_325_637),
    "global_graph/polygon_contract_to_number_mapping.json": ("1lhmf79NtbE-iscY1EX71TDrWlp7N7Phv", 103_566),
    "global_graph/polygon_graph_more_than_1_ratio.csv": ("1iRKJou7gkBXt0TUz6Xwpq3gBSWEWsxOJ", 8_214_272),
    "transactions/bsc.zip": ("1P97qDEBaWmZfs8DldXo-oNBtVpShCXzY", 5_671_092_781),
    "transactions/ethereum.zip": ("13wgfMbvdcpiwyM5GEQBcWfDviW_Eseta", 4_484_986_142),
    "transactions/polygon.zip": ("1OP7vC51RMFSmMp9dv1Mimb0aDcq1Nfv6", 2_805_020_440),
    "labels.csv": ("1GggOxrUKHl_-2HaRlcVe9tiQPtJZWPR6", 1_306_970),
}


def locate_distribution(root: Path) -> Path:
    nested = root / "Token Data"
    return nested if nested.is_dir() else root


def audit_distribution(root: Path) -> dict[str, Any]:
    distribution = locate_distribution(root)
    entries = []
    for relative, (file_id, expected_bytes) in UPSTREAM_FILES.items():
        path = distribution / relative
        actual_bytes = path.stat().st_size if path.is_file() else None
        complete = actual_bytes == expected_bytes
        entries.append({
            "relative_path": relative,
            "google_drive_file_id": file_id,
            "expected_bytes": expected_bytes,
            "actual_bytes": actual_bytes,
            "size_match": complete,
            "sha256": sha256_file(path) if complete else None,
        })
    return {
        "source": f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}",
        "distribution_root": str(distribution),
        "all_files_complete": all(row["size_match"] for row in entries),
        "entries": entries,
    }


def _csv_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members: dict[str, zipfile.ZipInfo] = {}
    for info in archive.infolist():
        if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".csv":
            continue
        name = PurePosixPath(info.filename).name.lower()
        if name in members:
            raise RuntimeError(f"duplicate CSV basename in archive: {name}")
        members[name] = info
    return members


def inspect_transaction_zip(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        members = _csv_members(archive)
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "csv_members": len(members),
            "compressed_bytes": sum(info.compress_size for info in members.values()),
            "uncompressed_bytes": sum(info.file_size for info in members.values()),
            "first_member": min(members) if members else None,
            "last_member": max(members) if members else None,
        }


def audit_zip_against_manifest(
    zip_path: Path,
    manifest: dict[str, Any],
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    expected = {Path(row["source_path"]).name.lower(): row["source_sha256"] for row in manifest["records"]}
    mismatches: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as archive:
        members = _csv_members(archive)
        missing = sorted(set(expected) - set(members))
        unexpected = sorted(set(members) - set(expected))
        for index, name in enumerate(sorted(set(expected) & set(members)), start=1):
            digest = hashlib.sha256()
            with archive.open(members[name]) as source:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected[name]:
                mismatches.append({"member": name, "expected_sha256": expected[name], "actual_sha256": actual})
            if progress is not None:
                progress(index, len(expected))
    return {
        "zip_path": str(zip_path),
        "expected_records": len(expected),
        "archive_csv_members": len(members),
        "missing_members": missing,
        "unexpected_members": unexpected,
        "hash_mismatches": mismatches,
        "all_source_files_exact": not missing and not unexpected and not mismatches,
    }


def safe_extract_flat(zip_path: Path, destination: Path) -> dict[str, Any]:
    """Extract transaction CSVs by basename using atomic per-file replacement."""
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = _csv_members(archive)
        for name, info in sorted(members.items()):
            target = destination / name
            if target.is_file() and target.stat().st_size == info.file_size:
                skipped += 1
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            with archive.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
            if temporary.stat().st_size != info.file_size:
                raise RuntimeError(f"extracted size mismatch: {name}")
            os.replace(temporary, target)
            extracted += 1
    return {"destination": str(destination), "extracted": extracted, "skipped": skipped, "files": extracted + skipped}


def load_preserved_manifest(evidence_zip: Path, chain: str) -> dict[str, Any]:
    member = f"evidence/dataset/manifests/{chain}.json"
    with zipfile.ZipFile(evidence_zip) as archive:
        return json.loads(archive.read(member))

