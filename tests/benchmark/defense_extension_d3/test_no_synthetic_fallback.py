from pathlib import Path

import yaml


def test_no_synthetic_fallback(d3_gate):
    config = yaml.safe_load(Path("configs/benchmark/sci_defense_extension_real.yaml").read_text(encoding="utf-8"))
    assert config["experiment"]["synthetic_inputs_allowed"] is False
    assert d3_gate["synthetic_fallback_used"] is False
    assert d3_gate["d1_synthetic_archived_and_excluded"] is True
    assert not Path("outputs/sci_defense_extension_real/processed").exists()
