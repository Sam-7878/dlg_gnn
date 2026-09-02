"""Hash-freeze every final R2 evidence, exact deliverable, report, and manuscript file."""
from __future__ import annotations

import json
from pathlib import Path

from validation.sci_v3_final_common import atomic_json, sha256_file


ROOT = Path("results/sci_v3_submission_r2")
MANIFEST = ROOT / "manuscript/final_artifact_manifest.json"
REPORT_ROOT = Path("docs/work_reports/sci_v3_submission_r2")
ROUND_ROOT = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2")


def record(path: Path, display: str) -> dict[str, object]:
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    artifacts = []
    for path in sorted(item for item in ROOT.rglob("*") if item.is_file() and item != MANIFEST):
        artifacts.append(record(path, str(path.relative_to(ROOT))))
    for path in sorted(item for item in REPORT_ROOT.rglob("*") if item.is_file()):
        artifacts.append(record(path, str(path)))
    for relative in (
        "implementation_report.md",
        "_41_01_DLG_StreamMC/DLG-StreamMC.tex",
        "_41_01_DLG_StreamMC/DLG-StreamMC.pdf",
        "_41_01_DLG_StreamMC/references.bib",
        "_41_01_DLG_StreamMC/DLG_StreamMC_Appendices_Reader_Guide_v1.tex",
    ):
        path = ROUND_ROOT / relative
        artifacts.append(record(path, str(path)))
    atomic_json(MANIFEST, {
        "version": 2,
        "status": "FROZEN",
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    })
    print(json.dumps({"status": "FROZEN", "artifact_count": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
