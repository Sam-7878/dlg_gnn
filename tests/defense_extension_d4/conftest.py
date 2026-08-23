from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def d4_root() -> Path:
    root = Path("outputs/sci_defense_extension_real_final")
    assert root.exists(), "Run scripts/defense_extension_real/finalize_round_d4.py first"
    return root
