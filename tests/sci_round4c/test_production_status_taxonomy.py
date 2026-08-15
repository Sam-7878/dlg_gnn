import pytest

from gog_fraud.experiments.round4c_policy import FINAL_STATUSES, validate_status


def test_status_taxonomy_is_closed_and_explicit():
    assert {"success","unsupported_algorithmic","unsupported_operational","failed_numerical","failed_cuda","failed_oom","failed_data","failed_other"} == FINAL_STATUSES
    with pytest.raises(ValueError): validate_status("oom")

