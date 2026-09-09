import pandas as pd
import pytest

from gog_fraud.experiments.round5_policy import validate_support_matrix


def test_support_matrix_requires_exact_cartesian_coverage():
    valid = pd.DataFrame([
        {"dataset": dataset, "model": model, "support_status": "supported"}
        for dataset in ("A", "B") for model in ("M1", "M2")
    ])
    validate_support_matrix(valid, ["A", "B"], ["M1", "M2"])
    with pytest.raises(ValueError):
        validate_support_matrix(valid.iloc[:-1], ["A", "B"], ["M1", "M2"])
