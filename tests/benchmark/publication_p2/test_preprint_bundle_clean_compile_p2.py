#!/usr/bin/env python3
"""
test_preprint_bundle_clean_compile_p2.py

Round P2 Gate: Verifies that the updated Preprints.org submission bundle
(publication/benchmark/preprints/DLG_Benchmark_Preprints_Submission.zip) unpacks into an
isolated clean temporary environment and compiles cleanly with 0 errors via
a full 4-pass sequence.
"""

from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PREPRINTS_DIR = REPO_ROOT / "publication" / "benchmark" / "preprints"
PREPRINT_ZIP = PREPRINTS_DIR / "DLG_Benchmark_Preprints_Submission.zip"
PREPRINT_PDF = PREPRINTS_DIR / "DLG-Benchmark-Preprint.pdf"


def test_preprint_zip_contains_updated_assets():
    assert PREPRINT_ZIP.exists(), f"Missing {PREPRINT_ZIP}"
    with zipfile.ZipFile(PREPRINT_ZIP, "r") as zf:
        names = zf.namelist()
        assert "DLG-Benchmark-Preprint.tex" in names
        assert "references.bib" in names
        assert "graphical_abstract.png" in names
        assert any(n.startswith("generated/") for n in names)


def test_preprint_pdf_artifact_valid():
    assert PREPRINT_PDF.exists(), f"Missing {PREPRINT_PDF}"
    assert PREPRINT_PDF.stat().st_size > 200 * 1024, "Preprint PDF is too small"


def _run_tex_cmd(cmd_list, cwd: Path):
    if shutil.which("pdflatex"):
        res = subprocess.run(cmd_list, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res
    wsl_exe = shutil.which("wsl")
    if wsl_exe:
        wsl_path = str(cwd).replace("\\", "/")
        if ":" in wsl_path:
            drive, rest = wsl_path.split(":", 1)
            wsl_path = f"/mnt/{drive.lower()}{rest}"
        cmd_str = f"cd '{wsl_path}' && {' '.join(cmd_list)}"
        res = subprocess.run([wsl_exe, "-d", "Ubuntu", "--", "bash", "-c", cmd_str],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res
    pytest.skip("Neither pdflatex nor WSL is available for isolated compilation test")


def test_isolated_clean_compilation_p2():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(PREPRINT_ZIP, "r") as zf:
            zf.extractall(tmp_path)

        p1 = _run_tex_cmd(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "DLG-Benchmark-Preprint.tex"], cwd=tmp_path)
        assert p1.returncode == 0, f"Pass 1 failed:\n{p1.stdout[-1500:]}"

        b = _run_tex_cmd(["bibtex", "DLG-Benchmark-Preprint"], cwd=tmp_path)
        assert b.returncode == 0, f"BibTeX failed:\n{b.stdout[-1500:]}"

        p2 = _run_tex_cmd(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "DLG-Benchmark-Preprint.tex"], cwd=tmp_path)
        assert p2.returncode == 0, f"Pass 2 failed:\n{p2.stdout[-1500:]}"

        p3 = _run_tex_cmd(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "DLG-Benchmark-Preprint.tex"], cwd=tmp_path)
        assert p3.returncode == 0, f"Pass 3 failed:\n{p3.stdout[-1500:]}"

        out_pdf = tmp_path / "DLG-Benchmark-Preprint.pdf"
        assert out_pdf.exists()
        assert out_pdf.stat().st_size > 200 * 1024
