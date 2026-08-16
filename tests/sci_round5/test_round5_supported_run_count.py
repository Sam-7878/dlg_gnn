import pandas as pd

from gog_fraud.experiments.round5_policy import supported_run_count


def test_run_count_uses_supported_pairs_not_nominal_matrix():
    support = pd.DataFrame({"support_status": ["supported", "unsupported_operational", "supported"]})
    assert supported_run_count(support, [42, 43, 44, 45, 46]) == 10
