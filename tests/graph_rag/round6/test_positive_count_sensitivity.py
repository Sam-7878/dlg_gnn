import numpy as np

from experiments.round6.evidence import _stratified_draws


def test_stratified_draw_preserves_positive_count():
    labels = np.array([1] * 7 + [0] * 23)
    for draw in _stratified_draws(labels, 20, 7):
        assert len(draw) == len(labels)
        assert int(labels[draw].sum()) == 7

