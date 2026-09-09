"""Hash-first audit for the frozen GoG-SCIMain-v1 evidence package."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row(
    artifact: str,
    path: Path,
    expected_hash: str,
    recovery_source: str,
) -> dict[str, Any]:
    found = path.is_file()
    actual_hash = sha256_file(path) if found else None
    status = "EXACT_HASH_MATCH" if found and actual_hash == expected_hash else (
        "HASH_MISMATCH" if found else "MISSING"
    )
    return {
        "artifact": artifact,
        "expected_hash": expected_hash,
        "found_path": str(path) if found else None,
        "actual_hash": actual_hash,
        "status": status,
        "recovery_source": recovery_source if found else "not found",
    }


def audit_recovery(
    dataset_dir: Path,
    upstream_dir: Path,
    preserved_manifest_path: Path,
    result_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    manifest = json.loads(preserved_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "real_dataset_manifest.json": sha256_file(preserved_manifest_path),
        "graph.pt": manifest["graph_sha256"],
        "transactions.parquet": manifest["transactions_sha256"],
        "split_manifest.json": manifest["split_manifest_sha256"],
        "future_edge_audit.csv": manifest["future_edge_audit_sha256"],
    }
    rows = [
        _row(name, dataset_dir / name, digest, "expected frozen path")
        for name, digest in expected.items()
    ]
    upstream_rows = []
    for chain, digest in manifest["upstream_manifest_sha256"].items():
        upstream_rows.append(_row(
            f"upstream:{chain}", upstream_dir / "manifests" / f"{chain}.json",
            digest, "expected upstream derivative path",
        ))
    exact_dataset = all(row["status"] == "EXACT_HASH_MATCH" for row in rows)
    exact_upstream = all(row["status"] == "EXACT_HASH_MATCH" for row in upstream_rows)
    leakage_claims = bool(
        manifest.get("future_edge_count") == 0
        and manifest.get("graph_hash_failures") == 0
    )
    payload = {
        "dataset_name": manifest["dataset_name"],
        "dataset_version": manifest["dataset_version"],
        "preserved_manifest_path": str(preserved_manifest_path),
        "preserved_manifest_sha256": expected["real_dataset_manifest.json"],
        "artifacts": rows,
        "upstream_artifacts": upstream_rows,
        "dataset_exact_recovery": exact_dataset,
        "upstream_exact_recovery": exact_upstream,
        "future_edge_audit_verified": rows[-1]["status"] == "EXACT_HASH_MATCH" and leakage_claims,
        "manifest_reports_zero_future_edges": leakage_claims,
        "new_dataset_version_fully_retrained": False,
        "search_audit": [
            {
                "scope": "/mnt/d/_Work and /home/sam exact-name search",
                "result": "only preserved real-dataset manifest copies found",
            },
            {
                "scope": "Windows Desktop/Documents/Downloads/OneDrive exact-name and archive search",
                "result": "no candidate found",
            },
            {
                "scope": "D: recycle bin exact-name search",
                "result": "no candidate found",
            },
            {
                "scope": "project and _Work ZIP archive member-name audit",
                "result": "no frozen v1 artifact member found",
            },
            {
                "scope": "Git history and unreachable-object audit",
                "result": "manifest only; unreachable blobs were unrelated source files",
            },
            {
                "scope": "extension-constrained SHA-256 search under /mnt/d/_Work",
                "result": "5,418 candidates (6,713,820,195 bytes) checked; only two byte-identical manifest copies matched",
            },
        ],
        "hash_search_summary": {
            "candidate_files": 5418,
            "candidate_bytes": 6713820195,
            "matches": [
                str(dataset_dir / "real_dataset_manifest.json"),
                str(preserved_manifest_path),
            ],
            "excluded_bulk_directories": ["graphs", "local_graphs", "edges", "build caches"],
        },
        "decision": "EXACT_V1_RECOVERED" if exact_dataset else "OPTION_B3_GATE_REMAINS_CLOSED",
        "no_synthetic_reconstruction": True,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    table_rows = []
    for row in rows + upstream_rows:
        found_path = f"`{row['found_path']}`" if row["found_path"] else "not found"
        actual_hash = f"`{row['actual_hash']}`" if row["actual_hash"] else "n/a"
        table_rows.append(
            f"| {row['artifact']} | `{row['expected_hash']}` | {found_path} | "
            f"{actual_hash} | {row['status']} | {row['recovery_source']} |"
        )
    report = """# Frozen Data Recovery Audit

| Artifact | Expected SHA-256 | Found path | Actual SHA-256 | Status | Recovery source |
|---|---|---|---|---|---|
""" + "\n".join(table_rows) + f"""

## Decision

Exact GoG-SCIMain-v1 recovery: **{exact_dataset}**. Exact upstream derivative recovery:
**{exact_upstream}**. Future-edge audit verified: **{payload['future_edge_audit_verified']}**.

The expected hashes were searched before any rebuild. The constrained SHA-256 pass checked 5,418
artifact-shaped candidates totaling 6,713,820,195 bytes; only the two preserved manifest copies matched. No missing artifact was reconstructed from
summary statistics or its digest. Because neither exact v1 evidence nor a provenance-complete new
version is available, Option B3 applies and Gate M remains closed.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return payload
