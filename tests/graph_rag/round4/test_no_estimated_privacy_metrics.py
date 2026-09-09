from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


def test_no_estimated_privacy_metrics():
    path = ROOT / "results/graphrag/round_4/privacy_utility.csv"
    raw = path.read_text().lower()
    for forbidden in ("estimated", "approx", "placeholder"):
        assert forbidden not in raw
    frame = pd.read_csv(path)
    required = {
        "attack_accuracy", "attack_balanced_accuracy", "attack_macro_f1",
        "attack_roc_auc", "attack_pr_auc", "majority_baseline", "random_baseline",
    }
    assert required <= set(frame.columns)
    assert set(frame.metric_source) == {"measured"}

