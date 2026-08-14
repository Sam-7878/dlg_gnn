import pytest

from gog_fraud.experiments.round4b_policy import SupportCell


def test_algorithmic_unsupported_requires_reason_and_is_not_execution_failure():
    cell = SupportCell(
        dataset="Reddit-Syn", model="AnomalyDAE",
        exact_backend_available=True, full_graph_feasible=False,
        reason_if_not="nonlinear all-pairs decoder complexity",
        primary_metric_available=False, status="unsupported_algorithmic",
    )
    assert cell.to_dict()["status"] == "unsupported_algorithmic"
    with pytest.raises(ValueError):
        SupportCell("Yelp-Syn", "AnomalyDAE", True, False, None, False, "unsupported_algorithmic")
