#!/usr/bin/env python3
"""Gate A23-G0: Archive A22 evidence snapshot, record SHA-256 manifest, and generate a22_review_findings.csv."""

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PUB_A22 = ROOT / "projects" / "benchmark" / "publication" / "a22"
PUB_A23 = ROOT / "projects" / "benchmark" / "publication" / "a23"
ARCHIVE_DIR = PUB_A23 / "archive_a22_snapshot"
SUB_PREPRINTS = ROOT / "projects" / "benchmark" / "submission" / "preprints"
SUB_APPLSCI = ROOT / "projects" / "benchmark" / "submission" / "applsci"
CANONICAL_REGISTRY = ROOT / "projects" / "benchmark" / "publication" / "a16" / "canonical_selected_run_registry.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_cmd(args: list[str]) -> str:
    res = subprocess.run(["git"] + args, cwd=ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip()


def main() -> None:
    PUB_A23.mkdir(parents=True, exist_ok=True)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Copy all current A22 publication files into archive_a22_snapshot/
    archived_files = {}
    for item in sorted(PUB_A22.iterdir()):
        if item.is_file():
            dest = ARCHIVE_DIR / item.name
            shutil.copy2(item, dest)
            archived_files[f"projects/benchmark/publication/a22/{item.name}"] = {
                "archived_to": f"projects/benchmark/publication/a23/archive_a22_snapshot/{item.name}",
                "sha256": sha256_file(item),
                "bytes": item.stat().st_size,
            }

    # Also archive the current pre-A23 SHA256SUMS.txt from both submission folders
    for label, sub_dir in [("preprints", SUB_PREPRINTS), ("applsci", SUB_APPLSCI)]:
        sha_file = sub_dir / "SHA256SUMS.txt"
        if sha_file.exists():
            dest = ARCHIVE_DIR / f"a22_{label}_SHA256SUMS.txt"
            shutil.copy2(sha_file, dest)
            archived_files[f"projects/benchmark/submission/{label}/SHA256SUMS.txt"] = {
                "archived_to": f"projects/benchmark/publication/a23/archive_a22_snapshot/a22_{label}_SHA256SUMS.txt",
                "sha256": sha256_file(sha_file),
                "bytes": sha_file.stat().st_size,
            }

    # Check git tag and commit history
    tag_commit = git_cmd(["rev-list", "-n", "1", "benchmark-a22-submission-2026-10-30"])
    head_commit = git_cmd(["rev-parse", "HEAD"])
    canonical_registry_sha = sha256_file(CANONICAL_REGISTRY)

    manifest = {
        "gate": "A23-G0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "historic_a22_tag": "benchmark-a22-submission-2026-10-30",
        "historic_a22_tag_commit": tag_commit,
        "pre_a23_head_commit": head_commit,
        "post_report_orcid_commit_explanation": (
            "Commit 7e764da3a55d9d13ab50a2dbebab13670946a6cd was tagged as benchmark-a22-submission-2026-10-30 "
            "with Preprints.pdf=334cd3cc6287fd92ff30628d50d998d66b62ccbc761586c43e1601b62a9cfbc5 and "
            "AppliedSciences.pdf=54684e5245238eb797a2acde19a01ba7c6c4a46b2f6c9a5e6d74ffca02e207a8. "
            "Subsequent commit 93e4c82790b9ac1a6b12f4eddf037858c83921c3 updated author ORCID metadata in the TeX files "
            "and rebuilt the PDFs/ZIPs/SHA256SUMS.txt without regenerating A22_SUBMISSION_READINESS_REPORT.md. "
            "A23 preserves A22_SUBMISSION_READINESS_REPORT.md untouched as a historical record and issues "
            "A23_SUBMISSION_READINESS_REPORT.md and A23_SHA256SUMS.txt from a single frozen post-build pass."
        ),
        "canonical_selected_run_registry": {
            "path": "projects/benchmark/publication/a16/canonical_selected_run_registry.csv",
            "sha256": canonical_registry_sha,
            "total_valid_runs": 455,
            "primary_13_datasets_runs": 420,
            "external_lanl_runs": 35,
        },
        "archived_a22_files": archived_files,
    }

    manifest_path = PUB_A23 / "a22_archived_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 2. Write a22_review_findings.csv
    findings = [
        {
            "finding_id": "A22-F1",
            "source_artifact": "projects/benchmark/publication/a22/A22_SUBMISSION_READINESS_REPORT.md;projects/benchmark/submission/*/SHA256SUMS.txt",
            "severity": "P0",
            "primary_paper_impact": "Release metadata & submission package SHA-256 coherence (report listed 7e764da hashes while working tree had post-ORCID 93e4c82 hashes)",
            "disposition": "RESOLVED_IN_A23",
            "resolution_artifact": "projects/benchmark/publication/a23/A23_SHA256SUMS.txt;projects/benchmark/publication/a23/A23_SUBMISSION_READINESS_REPORT.md",
        },
        {
            "finding_id": "A22-F2",
            "source_artifact": "projects/benchmark/publication/a22/a22_dataset_topology_diagnostics.csv;projects/benchmark/publication/a22/a22_data_and_split_provenance.md",
            "severity": "P0",
            "primary_paper_impact": "Supplement & provenance clarity (DGraphFin model_input_N listed as 367702 labeled population instead of 1225601 graph nodes; mismatched upstream denominators yielded node_retention_ratio > 1)",
            "disposition": "RESOLVED_IN_A23",
            "resolution_artifact": "projects/benchmark/publication/a23/dataset_graph_identity_canonical.csv;projects/benchmark/publication/a23/dataset_graph_identity_canonical.md",
        },
        {
            "finding_id": "A22-F3",
            "source_artifact": "projects/benchmark/publication/a22/a22_ablation_provenance_audit.csv;projects/benchmark/submission/*/DLG-Benchmark_A16_*.tex (Table A3 caption & Limitations item 7)",
            "severity": "P0",
            "primary_paper_impact": "Manuscript text & Table A3 caption (over-causal wording claiming permutation 'proves' or 'confirms' gains stem from regularized representation capacity)",
            "disposition": "RESOLVED_IN_A23",
            "resolution_artifact": "projects/benchmark/publication/a23/ablation_provenance_noncausal.csv;projects/benchmark/publication/a23/a23_g2_prohibited_phrase_scan.json",
        },
        {
            "finding_id": "A22-F4",
            "source_artifact": "projects/benchmark/publication/a22/a22_stat_view_membership.csv",
            "severity": "P0",
            "primary_paper_impact": "Supplement & text consistency (View S3 described as all real entity fraud labels despite BitcoinOTC using canonical synthetic node anomaly injection on a real trust graph)",
            "disposition": "RESOLVED_IN_A23",
            "resolution_artifact": "projects/benchmark/publication/a23/s3_label_provenance_gate.csv;projects/benchmark/publication/a23/a23_stat_view_membership.csv",
        },
        {
            "finding_id": "A22-F5",
            "source_artifact": "projects/benchmark/publication/a22/a22_telemetry_audit.md;projects/benchmark/publication/a22/a22_gadnr_numerical_diagnostic.json;Table 6 footnote",
            "severity": "P0",
            "primary_paper_impact": "Table 6 footnote & diagnostic note (must distinguish observed Flickr-Syn OOM, historical NO_APPROVED_RAW_RUN failures on 4 datasets, and synthetic rank-deficient covariance illustration)",
            "disposition": "RESOLVED_IN_A23",
            "resolution_artifact": "projects/benchmark/publication/a23/gadnr_failure_evidence_note.md",
        },
        {
            "finding_id": "A22-F6",
            "source_artifact": "projects/benchmark/publication/a22/a22_exact_loss_source_trace.md;projects/benchmark/publication/a22/a22_objective_branch_decision.md",
            "severity": "P1",
            "primary_paper_impact": "Audit note mathematical notation & test scope (wrote Z^T A[:,i] = (AZ)_i which holds only for symmetric A, used shorthand path dlg_gnn/model/* instead of src/gog_fraud/models/pygod/*, and clarified N=40 CPU unit-test scope)",
            "disposition": "RESOLVED_IN_A23",
            "resolution_artifact": "projects/benchmark/publication/a23/g1_exact_loss_canonical_trace.md;projects/benchmark/publication/a23/a23_g4_asymmetric_orientation_test.json",
        },
    ]

    findings_path = PUB_A23 / "a22_review_findings.csv"
    with findings_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "finding_id",
                "source_artifact",
                "severity",
                "primary_paper_impact",
                "disposition",
                "resolution_artifact",
            ],
        )
        writer.writeheader()
        writer.writerows(findings)

    # 3. Write A23_CORRECTION_NOTE_ON_A22.md
    note_md = f"""# A23 Correction Note on Historic A22 Submission Readiness Report

- **Date:** 2026-10-30
- **Historic A22 Tag Preserved:** `benchmark-a22-submission-2026-10-30` (`{tag_commit}`)
- **Post-Tag ORCID Commit:** `{head_commit}`
- **Untouched Historic Report:** `projects/benchmark/publication/a22/A22_SUBMISSION_READINESS_REPORT.md`
- **Archived A22 Snapshot Directory:** `projects/benchmark/publication/a23/archive_a22_snapshot/`

## 1. Purpose of This Note

In accordance with Work Order `DLG-BENCH-WO-A23-FINAL-PROVENANCE-ONLY`, the signed A22 report (`projects/benchmark/publication/a22/A22_SUBMISSION_READINESS_REPORT.md`) and historic Git tag `benchmark-a22-submission-2026-10-30` are preserved untouched as an immutable audit record. This correction note documents why a post-tag commit caused SHA-256 drift between `A22_SUBMISSION_READINESS_REPORT.md` and the working-tree submission bundles, and summarizes the six documentary/provenance corrections completed in **A23**.

## 2. Root Cause of the A22 Report vs. Package SHA-256 Discrepancy (`A22-F1`)

1. At commit `7e764da3a55d9d13ab50a2dbebab13670946a6cd` (tagged `benchmark-a22-submission-2026-10-30`), the compiled manuscripts had SHA-256 digests:
   - `DLG-Benchmark_A16_Preprints.pdf`: `334cd3cc6287fd92ff30628d50d998d66b62ccbc761586c43e1601b62a9cfbc5`
   - `DLG-Benchmark_A16_AppliedSciences.pdf`: `54684e5245238eb797a2acde19a01ba7c6c4a46b2f6c9a5e6d74ffca02e207a8`
   These exact digests were recorded in `A22_SUBMISSION_READINESS_REPORT.md`.
2. Immediately afterward, commit `93e4c82790b9ac1a6b12f4eddf037858c83921c3` updated author ORCID metadata in `DLG-Benchmark_A16_Preprints.tex` and `DLG-Benchmark_A16_AppliedSciences.tex`, recompiled the PDFs and `source_bundle.zip` archives, and updated `submission/preprints/SHA256SUMS.txt` and `submission/applsci/SHA256SUMS.txt`, **without** regenerating `A22_SUBMISSION_READINESS_REPORT.md`.
3. In **A23**, all SHA-256 digests in `submission/preprints/SHA256SUMS.txt`, `submission/applsci/SHA256SUMS.txt`, `publication/a23/A23_SHA256SUMS.txt`, and `publication/a23/A23_SUBMISSION_READINESS_REPORT.md` are computed in a single post-build pass from the final frozen files on disk, eliminating any post-report rebuild drift.

## 3. Summary of A22 Independent Review Findings & A23 Resolutions

| Finding ID | Severity | Summary of Issue | A23 Resolution Artifact |
| :--- | :---: | :--- | :--- |
| `A22-F1` | P0 | Post-report ORCID rebuild caused SHA-256 mismatch between `A22_SUBMISSION_READINESS_REPORT.md` and `submission/*/SHA256SUMS.txt`. | `publication/a23/A23_SHA256SUMS.txt`, `A23_SUBMISSION_READINESS_REPORT.md` |
| `A22-F2` | P0 | `a22_dataset_topology_diagnostics.csv` used DGraphFin's `evaluation_nodes` (`367,702`) instead of `graph_nodes` (`1,225,601`) and divided mismatched upstream denominators (`node_retention_ratio > 1`). | `publication/a23/dataset_graph_identity_canonical.csv`, `dataset_graph_identity_canonical.md` |
| `A22-F3` | P0 | Over-causal mechanistic interpretation of the Elliptic permutation/zero-vector ablation in Table A3 caption, Limitations item 7, and `a22_ablation_provenance_audit.csv`. | `publication/a23/ablation_provenance_noncausal.csv`, `a23_g2_prohibited_phrase_scan.json` |
| `A22-F4` | P0 | `a22_stat_view_membership.csv` described View S3 as all real entity fraud labels, overlooking that `BitcoinOTC` uses canonical injected node anomaly labels (`180/6,005 = 3.00%`) on a real trust graph. | `publication/a23/s3_label_provenance_gate.csv`, `a23_stat_view_membership.csv` |
| `A22-F5` | P0 | GADNR failure narrative needed explicit separation between Flickr-Syn 8-GB OOM, historical `NO_APPROVED_RAW_RUN` failures on 4 datasets, and the synthetic rank-deficient covariance illustration. | `publication/a23/gadnr_failure_evidence_note.md` |
| `A22-F6` | P1 | `a22_exact_loss_source_trace.md` wrote `Z^T A[:,i] = (AZ)_i` (valid only for symmetric `A`), used shorthand paths `dlg_gnn/model/*`, and needed explicit $N=40$ CPU fixture scope. | `publication/a23/g1_exact_loss_canonical_trace.md`, `a23_g4_asymmetric_orientation_test.json` |
"""
    (PUB_A23 / "A23_CORRECTION_NOTE_ON_A22.md").write_text(note_md, encoding="utf-8")
    print("A23-G0 completed successfully.")


if __name__ == "__main__":
    main()

