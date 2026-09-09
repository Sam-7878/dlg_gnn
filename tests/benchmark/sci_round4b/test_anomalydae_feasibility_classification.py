from gog_fraud.experiments.round4b_policy import classify_exact_runtime


def test_runtime_above_declared_24_gpu_hour_limit_is_algorithmic_unsupported():
    assert classify_exact_runtime(24 * 3600 + 1) == "unsupported_algorithmic"
    assert classify_exact_runtime(24 * 3600) == "supported_exact"
