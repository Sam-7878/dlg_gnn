import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_near_perfect_shortcut_cannot_pass_paper_gate():
    shortcuts = pd.read_csv(RESULTS / "shortcut_baselines.csv")
    gate = json.loads((RESULTS / "paper_ready_gate_v4.json").read_text())
    expected = {
        "label prevalence only", "entity_type only", "source_dataset only", "timestamp only",
        "degree only", "wallet-present flag only", "domain-present flag only", "text length only",
    }
    assert expected <= set(shortcuts.baseline)
    detected = ((shortcuts.roc_auc >= 0.95) & (shortcuts.auc_pr >= 0.95)).any()
    assert shortcuts.near_perfect.astype(bool).any() == detected
    assert gate["checks"]["shortcut_baselines_not_near_perfect"] is (not detected)
    if detected:
        assert gate["paper_ready"] is False
