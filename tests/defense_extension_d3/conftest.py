import json
from pathlib import Path

import pytest


GATE_PATH = Path("outputs/sci_defense_extension_real/manifests/official_source_gate.json")


@pytest.fixture(scope="session")
def d3_gate():
    assert GATE_PATH.exists(), "Run scripts/defense_extension/run_d3_source_gate.py first"
    return json.loads(GATE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def require_real_sources(d3_gate):
    if not d3_gate["official_raw_available"]:
        pytest.skip("D3 official-source gate failed; downstream preprocessing/benchmark is forbidden")
    return d3_gate
