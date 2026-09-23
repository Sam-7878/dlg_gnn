from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", default="."); parser.add_argument("--output", required=True)
    args = parser.parse_args(); root = Path(args.repo_root).resolve(); output = Path(args.output).resolve()
    report_dir = root / "docs/work_reports/102_stream_mc_update2"
    selected: list[Path] = []
    for relative in ("configs/sci", "results_sci"):
        base = root / relative
        if base.exists(): selected.extend(path for path in base.rglob("*") if path.is_file())
    for pattern in ("DLG_StreamMC_SCI_Round2_Development_Experiment_Report.*", "DLG_StreamMC_SCI_Round2_Evidence_Index.csv", "Dataset_Provenance_and_Label_Audit.md", "dataset_provenance_label_audit.json", "dataset_exclusions.csv", "environment_manifest.json", "strict_orchestrator.log", "test_summaries/*"):
        selected.extend(path for path in report_dir.glob(pattern) if path.is_file())
    integrated_reports = root / "reports"
    if integrated_reports.exists():
        selected.extend(
            path for path in integrated_reports.glob("DLG_StreamMC_SCI_*")
            if path.is_file()
        )
    if (root / "requirements-sci-lock.txt").is_file(): selected.append(root / "requirements-sci-lock.txt")
    selected = sorted(set(selected))
    with tempfile.TemporaryDirectory(prefix="dlg_round2_") as temporary:
        package = Path(temporary) / "DLG_StreamMC_SCI_Round2_Evidence_Package"
        copied: list[Path] = []
        for source in selected:
            relative = source.relative_to(root)
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target); copied.append(target)
        manifest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(copied),
            "files": {path.relative_to(package).as_posix(): sha256(path) for path in copied},
        }
        (package / "REPORT_MANIFEST.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(package.parent).as_posix())
    print(output)


if __name__ == "__main__": main()
