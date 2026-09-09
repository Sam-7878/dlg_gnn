from gog_fraud.experiments.round2_validity import validation_support


def test_low_positive_counts_warn_without_excluding_rows():
    result = validation_support([0] * 20 + [1] * 3, [0] * 20 + [1] * 2)
    assert result["threshold_unstable_warning"] and result["metric_low_support_warning"]
    assert result["validation_positive"] == 3

