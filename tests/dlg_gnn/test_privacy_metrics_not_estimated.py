import csv
import math

import _round3_bootstrap  # noqa: F401

from experiments.round3.artifact_paths import ROUND3_RESULTS


def test_privacy_metrics_are_numeric_measured_outputs():
    path = ROUND3_RESULTS / "real_privacy_utility.csv"
    text = path.read_text(encoding="utf-8").lower()
    assert "estimated" not in text
    assert "placeholder" not in text
    rows = list(csv.DictReader(text.splitlines()))
    assert rows
    for row in rows:
        for key in ("attack_balanced_accuracy", "attack_macro_f1", "attack_roc_auc"):
            value = float(row[key])
            assert math.isfinite(value)
            assert 0.0 <= value <= 1.0
