#!/usr/bin/env python3
"""
test_environment_versioning_split.py

Round P4 Publication Gate:
Verifies that:
1. 'provenance/frozen_execution_environment.json' specifies the historical immutable benchmark run environment.
2. 'provenance/current_reproduction_environment.json' specifies the active verified reproduction environment.
3. Both files have distinct 'environment_type' fields.
"""

import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PROV_DIR = REPO_ROOT / "outputs/benchmark/manuscript_m5/provenance"


def test_frozen_environment_json():
    frozen_path = PROV_DIR / "frozen_execution_environment.json"
    assert frozen_path.exists(), f"Missing {frozen_path}"
    data = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert data.get("environment_type") == "frozen_execution_environment"
    assert data.get("python") == "3.12.13"
    assert "2.5.1" in data.get("torch", "")


def test_current_environment_json():
    curr_path = PROV_DIR / "current_reproduction_environment.json"
    assert curr_path.exists(), f"Missing {curr_path}"
    data = json.loads(curr_path.read_text(encoding="utf-8"))
    assert data.get("environment_type") == "current_reproduction_environment"
    assert data.get("python") == "3.12.13"
    assert "2.5.1" in data.get("torch", "")
    assert "GREEN" in data.get("verification_status", "")
