from __future__ import annotations

import json
import importlib.metadata
import platform
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .evidence_index import sha256_file


def build_package(*, repo_root: str | Path, report: str | Path, report_json: str | Path, evidence_index: str | Path, output: str | Path) -> Path:
    root = Path(repo_root).resolve()
    output_path = Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix="dlg_streammc_report_") as temp:
        package = Path(temp) / "DLG_StreamMC_SCI_Report_Package"
        for directory in ("tables", "figures", "source_data", "configs", "split_manifests", "test_summaries", "failure_summaries", "reproduction"):
            (package / directory).mkdir(parents=True, exist_ok=True)
        copied: list[Path] = []
        for source in (Path(report), Path(report_json), Path(evidence_index)):
            target = package / source.name
            shutil.copy2(source, target); copied.append(target)
        for source in (root / "configs/sci").rglob("*.yaml") if (root / "configs/sci").exists() else ():
            target = package / "configs" / source.name
            shutil.copy2(source, target); copied.append(target)
        test_root = root / "docs/work_reports/101_stream_mc_check_result/test_summaries"
        for source in test_root.glob("*") if test_root.exists() else ():
            if source.is_file():
                target = package / "test_summaries" / source.name
                shutil.copy2(source, target); copied.append(target)
        readme = package / "reproduction/README.md"
        readme.write_text("# Reproduction\n\nRun `scripts/build_sci_verification_report.py`, then `scripts/validate_sci_report.py`. Missing experiment results remain NOT_RUN.\n", encoding="utf-8")
        commands = package / "reproduction/commands.sh"
        commands.write_text("PYTHONPATH=src python scripts/build_sci_verification_report.py --repo-root . --output docs/work_reports/101_stream_mc_check_result/DLG_StreamMC_SCI_Integrated_Verification_Report.md --strict\n", encoding="utf-8")
        environment = package / "reproduction/environment.txt"
        environment.write_text(f"OS={platform.platform()}\nPython={platform.python_version()}\n", encoding="utf-8")
        lock = package / "reproduction/requirements-lock.txt"
        distributions = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions() if dist.metadata.get('Name'))
        lock.write_text("\n".join(distributions) + "\n", encoding="utf-8")
        copied.extend((readme, commands, environment, lock))
        manifest = {"generated_at": datetime.now(timezone.utc).isoformat(), "files": {path.relative_to(package).as_posix(): sha256_file(path) for path in copied}}
        manifest_path = package / "REPORT_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(package.rglob("*")):
                if path.is_file(): archive.write(path, path.relative_to(package.parent).as_posix())
    return output_path
