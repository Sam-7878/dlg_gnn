import numpy as np
import pytest

from experiments.round7.statistics import paired_panel_comparison


def test_paired_statistics_are_reproducible_and_stratified() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1])
    left = [np.asarray([0.1, 0.2, 0.3, 0.8, 0.7, 0.9])]
    right = [np.asarray([0.4, 0.3, 0.2, 0.6, 0.5, 0.7])]
    first = paired_panel_comparison(labels, left, right, n_resamples=100, seed=9)
    second = paired_panel_comparison(labels, left, right, n_resamples=100, seed=9)
    assert first == second
    assert first["mean_auc_pr_difference"] >= 0
    assert first["n_bootstrap"] == first["n_class_stratified_bootstrap"] == 100
    assert first["n_randomization"] == 100


def test_paired_statistics_reject_misalignment() -> None:
    with pytest.raises(ValueError, match="align"):
        paired_panel_comparison(np.asarray([0, 1]), [np.asarray([0.2])], [np.asarray([0.1])], n_resamples=2)

