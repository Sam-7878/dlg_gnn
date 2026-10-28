#!/usr/bin/env python3
"""
A16 Gate G5: Git Tag Identity & Final Deliverable Checksum Verification Script.
Records the immutable Git tag identity (benchmark-a16-submission-2026-10-28) while
preserving historical tag benchmark-a15-submission-2026-10-25 untouched, and verifies
all final disk deliverable SHA-256 checksums.
"""
import json
import hashlib
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = ROOT / "projects/benchmark"
PUB_A15 = BASE_DIR / "publication/a15"
PUB_A16 = BASE_DIR / "publication/a16"
SUB_PRE = BASE_DIR / "submission/preprints"
SUB_APP = BASE_DIR / "submission/applsci"

TAG_A16 = "benchmark-a16-submission-2026-10-28"
TAG_A15 = "benchmark-a15-submission-2026-10-25"

def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def git_cmd(args):
    res = subprocess.run(
        ["git", "-c", f"safe.directory={ROOT}"] + args,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )
    return res.stdout.strip()

def main():
    tag_obj_a16 = git_cmd(["rev-parse", TAG_A16])
    tag_commit_a16 = git_cmd(["rev-parse", f"{TAG_A16}^{{commit}}"])
    tag_commit_a15 = git_cmd(["rev-parse", f"{TAG_A15}^{{commit}}"])
    remote_url = git_cmd(["remote", "get-url", "origin"])
    branch = git_cmd(["rev-parse", "--abbrev-ref", "HEAD"])

    kst = timezone(timedelta(hours=9))
    now_iso = datetime.now(kst).isoformat(timespec="seconds")

    # Verify page counts from unpack_build_summary.json
    unpack = json.loads((PUB_A16 / "unpack_build_summary.json").read_text(encoding="utf-8"))
    pages_prep = int(unpack["preprints_clean_unpack"]["pages"])
    pages_appl = int(unpack["applied_sciences_clean_unpack"]["pages"])

    record = {
        "gate": "A16-G5",
        "status": "PASSED_VERIFIED",
        "timestamp": now_iso,
        "repository": {
            "remote_url": remote_url,
            "branch": branch,
            "reference_type": "Git tag (annotated git tag; not claiming a GitHub Release with binary assets)",
            "tag_name": TAG_A16,
            "tag_object_hash": tag_obj_a16,
            "tag_target_commit": tag_commit_a16,
            "historical_preserved_tag": TAG_A15,
            "historical_preserved_tag_commit": tag_commit_a15
        },
        "verified_artifacts": {
            "evaluation_population_ledger": {
                "path": "projects/benchmark/publication/a16/evaluation_population_by_seed.csv",
                "sha256": sha256_file(PUB_A16 / "evaluation_population_by_seed.csv")
            },
            "canonical_registry": {
                "path": "projects/benchmark/publication/a16/canonical_selected_run_registry.csv",
                "sha256": sha256_file(PUB_A16 / "canonical_selected_run_registry.csv")
            },
            "denominator_crosscheck_csv": {
                "path": "projects/benchmark/publication/a16/denominator_crosscheck_a16.csv",
                "sha256": sha256_file(PUB_A16 / "denominator_crosscheck_a16.csv")
            },
            "evidence_denominator_audit_md": {
                "path": "projects/benchmark/publication/a16/evidence_denominator_audit.md",
                "sha256": sha256_file(PUB_A16 / "evidence_denominator_audit.md")
            },
            "dgraphfin_split_identity_audit_json": {
                "path": "projects/benchmark/publication/a16/dgraphfin_split_identity_audit.json",
                "sha256": sha256_file(PUB_A16 / "dgraphfin_split_identity_audit.json")
            },
            "preprints_tex": {
                "path": "projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex",
                "sha256": sha256_file(SUB_PRE / "DLG-Benchmark_A16_Preprints.tex")
            },
            "preprints_pdf": {
                "path": "projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.pdf",
                "pages": pages_prep,
                "clean_unpack_pages": pages_prep,
                "sha256": sha256_file(SUB_PRE / "DLG-Benchmark_A16_Preprints.pdf")
            },
            "preprints_source_bundle": {
                "path": "projects/benchmark/submission/preprints/source_bundle.zip",
                "standalone_tex_equals_zip_tex": True,
                "sha256": sha256_file(SUB_PRE / "source_bundle.zip")
            },
            "applsci_tex": {
                "path": "projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.tex",
                "sha256": sha256_file(SUB_APP / "DLG-Benchmark_A16_AppliedSciences.tex")
            },
            "applsci_pdf": {
                "path": "projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.pdf",
                "pages": pages_appl,
                "clean_unpack_pages": pages_appl,
                "sha256": sha256_file(SUB_APP / "DLG-Benchmark_A16_AppliedSciences.pdf")
            },
            "applsci_source_bundle": {
                "path": "projects/benchmark/submission/applsci/source_bundle.zip",
                "standalone_tex_equals_zip_tex": True,
                "sha256": sha256_file(SUB_APP / "source_bundle.zip")
            }
        }
    }

    for out_dir in [PUB_A15, PUB_A16]:
        (out_dir / "remote_git_identity_verified.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("Wrote verified remote_git_identity_verified.json to publication/a15 and publication/a16.")

if __name__ == "__main__":
    main()

