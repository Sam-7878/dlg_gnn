from pathlib import Path

from experiments.round5.analysis import SEEDS, load_prediction_panel


ROOT = Path(__file__).resolve().parents[2]


def test_frozen_five_seed_test_predictions_are_identity_aligned():
    identity, panel = load_prediction_panel(ROOT / "results/graphrag/round_4/raw_predictions", 1)
    assert set(panel) == set(SEEDS)
    assert len(identity) == 3648
    assert int(identity.label.sum()) == 107
    assert not identity.event_id.duplicated().any()
