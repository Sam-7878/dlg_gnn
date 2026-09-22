#!/usr/bin/env python3
"""
test_release_dependency_closure.py

Round M5 Freeze Gate: Verifies that the public release package contains the complete
dependency closure necessary to import and execute both primary and sensitivity runners.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RELEASE_ZIP = REPO_ROOT / "outputs/benchmark/manuscript_m5/release/DLG_GNN_Benchmark_M5_Release.zip"


def test_release_contains_required_modules():
    assert RELEASE_ZIP.exists()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(RELEASE_ZIP, "r") as zf:
            zf.extractall(tmp_path)

        root = tmp_path / "dlg_gnn"
        src_dir = root / "src"
        assert src_dir.exists()

        # Check required module files
        required_files = [
            src_dir / "gog_fraud" / "__init__.py",
            src_dir / "gog_fraud" / "models" / "pygod" / "shared_reconstruction.py",
            src_dir / "gog_fraud" / "data" / "dgraphfin_aligned.py",
            src_dir / "gog_fraud" / "evaluation" / "reproducibility.py",
            src_dir / "gog_fraud" / "evaluation" / "threshold_protocol.py",
            src_dir / "gog_fraud" / "experiments" / "round5_policy.py",
            src_dir / "gog_fraud" / "pipelines" / "run_sci_round5.py",
            src_dir / "gog_fraud" / "pipelines" / "run_sci_round4c.py",
            src_dir / "analysis" / "__init__.py",
            src_dir / "analysis" / "utils.py",
        ]
        for rf in required_files:
            assert rf.exists(), f"Missing required dependency in release: {rf}"

        # Clean unpack import test
        test_code = (
            f"import sys; sys.path.insert(0, r'{src_dir}'); "
            "import gog_fraud.models.pygod.shared_reconstruction; "
            "import gog_fraud.data.dgraphfin_aligned; "
            "import gog_fraud.evaluation.threshold_protocol; "
            "import gog_fraud.pipelines.run_sci_round5; "
            "print('All dependency closure modules imported successfully!')"
        )
        res = subprocess.run([sys.executable, "-c", test_code], cwd=root, capture_output=True, text=True)
        assert res.returncode == 0, f"Import test failed:\nStdout: {res.stdout}\nStderr: {res.stderr}"
