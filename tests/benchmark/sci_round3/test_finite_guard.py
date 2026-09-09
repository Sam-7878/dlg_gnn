import pytest
import torch
from gog_fraud.evaluation.finite_guard import NonFiniteTensorError, assert_finite_tensor


def test_guard_reports_first_stage_without_repair():
    with pytest.raises(NonFiniteTensorError) as caught:
        assert_finite_tensor(torch.tensor([1.0, float("nan"), float("inf")]),
                             stage="per_node_anomaly_score", dataset="Yelp-Syn",
                             model="DOMINANT", partition_id=7, node_range="10:13")
    diagnostic = caught.value.diagnostic
    assert diagnostic.nan_count == 1 and diagnostic.inf_count == 1
    assert diagnostic.tensor_min == diagnostic.tensor_max == 1.0
