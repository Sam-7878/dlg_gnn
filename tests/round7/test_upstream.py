import hashlib
import zipfile
from pathlib import Path

from experiments.round7.upstream import audit_zip_against_manifest, inspect_transaction_zip, safe_extract_flat


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    archive_path = tmp_path / "chain.zip"
    payloads = {"nested/0xabc.csv": b"timestamp,from,to,value\n1,a,b,2\n", "0xdef.csv": b"timestamp,from,to,value\n2,b,c,3\n"}
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in payloads.items():
            archive.writestr(name, content)
    manifest = {
        "records": [
            {"source_path": f"/raw/{Path(name).name}", "source_sha256": hashlib.sha256(content).hexdigest()}
            for name, content in payloads.items()
        ]
    }
    return archive_path, manifest


def test_zip_manifest_audit_and_flat_extract(tmp_path: Path) -> None:
    archive_path, manifest = _fixture(tmp_path)
    inspection = inspect_transaction_zip(archive_path)
    assert inspection["csv_members"] == 2
    assert audit_zip_against_manifest(archive_path, manifest)["all_source_files_exact"] is True
    result = safe_extract_flat(archive_path, tmp_path / "raw")
    assert result == {"destination": str(tmp_path / "raw"), "extracted": 2, "skipped": 0, "files": 2}
    assert safe_extract_flat(archive_path, tmp_path / "raw")["skipped"] == 2


def test_zip_manifest_audit_detects_hash_change(tmp_path: Path) -> None:
    archive_path, manifest = _fixture(tmp_path)
    manifest["records"][0]["source_sha256"] = "0" * 64
    audit = audit_zip_against_manifest(archive_path, manifest)
    assert audit["all_source_files_exact"] is False
    assert len(audit["hash_mismatches"]) == 1

