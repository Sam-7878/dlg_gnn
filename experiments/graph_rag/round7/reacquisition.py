"""Persist exact-source audits for reacquired GoG transaction archives."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.round7.provenance import sha256_file
from experiments.round7.upstream import audit_distribution, inspect_transaction_zip


ROUND2_EVIDENCE = ROOT / "docs/work_reports/_2026/102_stream_mc_round_2/DLG_StreamMC_SCI_Round2_Evidence_Package.zip"
ROUND3_EVIDENCE = ROOT / "docs/work_reports/_2026/103_stream_mc_round_3/DLG_StreamMC_SCI_Round3_Evidence_Package.zip"


def expected_source_hashes(chain: str) -> tuple[dict[str, str], dict[str, Any]]:
    if chain == "ethereum":
        member = "DLG_StreamMC_SCI_Round2_Evidence_Package/results_sci/manifests/ethereum.json"
        with zipfile.ZipFile(ROUND2_EVIDENCE) as archive:
            manifest = json.loads(archive.read(member))
        hashes = {Path(name).name.lower(): digest for name, digest in manifest["file_hashes"].items()}
        lineage = {
            "evidence_package": str(ROUND2_EVIDENCE),
            "member": member,
            "schema": "round2 file_hashes",
            "manifest_complete": bool(manifest["manifest_complete"]),
            "physical_root": manifest["physical_root"],
        }
        return hashes, lineage
    member = f"evidence/dataset/manifests/{chain}.json"
    with zipfile.ZipFile(ROUND3_EVIDENCE) as archive:
        raw = archive.read(member)
        manifest = json.loads(raw)
    hashes = {Path(row["source_path"]).name.lower(): row["source_sha256"] for row in manifest["records"]}
    lineage = {
        "evidence_package": str(ROUND3_EVIDENCE),
        "member": member,
        "schema": "round3 SCI v2 records",
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "manifest_complete": bool(manifest["manifest_complete"]),
        "physical_root": manifest["raw_root"],
        "generated_at": manifest["generated_at"],
    }
    return hashes, lineage


def audit_extracted_chain(raw_root: Path, chain: str) -> dict[str, Any]:
    expected, lineage = expected_source_hashes(chain)
    chain_root = raw_root / chain
    paths = {path.name.lower(): path for path in chain_root.glob("*.csv")}
    missing = sorted(set(expected) - set(paths))
    unexpected = sorted(set(paths) - set(expected))
    mismatches = []
    for index, name in enumerate(sorted(set(expected) & set(paths)), start=1):
        actual = sha256_file(paths[name])
        if actual != expected[name]:
            mismatches.append({"file": name, "expected_sha256": expected[name], "actual_sha256": actual})
        if index % 500 == 0 or index == len(expected):
            print(f"[{chain}] source hash {index}/{len(expected)} mismatches={len(mismatches)}", flush=True)
    return {
        "chain": chain,
        "lineage": lineage,
        "expected_files": len(expected),
        "actual_files": len(paths),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "hash_mismatches": mismatches,
        "all_source_files_exact": not missing and not unexpected and not mismatches,
    }


def run(download_root: Path, raw_root: Path, output: Path, chains: list[str]) -> dict[str, Any]:
    distribution = audit_distribution(download_root)
    chain_audits = []
    zip_audits = []
    for chain in chains:
        zip_path = Path(distribution["distribution_root"]) / f"transactions/{chain}.zip"
        zip_audits.append(inspect_transaction_zip(zip_path))
        chain_audits.append(audit_extracted_chain(raw_root, chain))
    payload = {
        "audit_time_utc": datetime.now(timezone.utc).isoformat(),
        "official_distribution": distribution,
        "zip_inventory": zip_audits,
        "extracted_source_audit": chain_audits,
        "all_downloads_complete": bool(distribution["all_files_complete"]),
        "all_extracted_sources_exact": all(row["all_source_files_exact"] for row in chain_audits),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chains", nargs="+", choices=("ethereum", "bsc", "polygon"), required=True)
    args = parser.parse_args()
    result = run(args.download_root, args.raw_root, args.output, args.chains)
    print(json.dumps({
        "all_downloads_complete": result["all_downloads_complete"],
        "all_extracted_sources_exact": result["all_extracted_sources_exact"],
        "chains": args.chains,
    }, indent=2))
    return 0 if result["all_extracted_sources_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

