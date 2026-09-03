import numpy as np

from experiments.round5.analysis import paired_ap_comparison


def test_paired_comparison_preserves_event_support():
    labels = np.array([0] * 80 + [1] * 20)
    strong = np.linspace(0, 1, 100)
    weak = np.full(100, .2)
    result = paired_ap_comparison(labels, strong, weak, n_bootstrap=100, n_randomization=100, seed=7)
    assert result["n_events"] == 100 and result["n_positive"] == 20
    assert result["mean_auc_pr_difference"] > 0
    assert result["n_bootstrap"] == 100 and result["n_randomization"] == 100
