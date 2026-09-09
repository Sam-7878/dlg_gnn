from pathlib import Path


def test_synthetic_baseline_is_explicitly_guarded():
    source = Path("src/gog_fraud/pipelines/run_baseline_benchmark.py").read_text(encoding="utf-8")
    assert "--allow-synthetic-demo" in source
    assert "SYNTHETIC_DEMO_NOT_PAPER_ELIGIBLE" in source
    assert "parser.error" in source
