import pandas as pd
import pytest

from gog_fraud.experiments.round5_policy import validate_final_raw


def test_supported_pair_requires_exactly_five_successes():
    support = pd.DataFrame([{"dataset":"D","model":"M","support_status":"supported"}])
    raw = pd.DataFrame([{"dataset":"D","model":"M","seed":seed,"status":"success"} for seed in range(42,47)])
    validate_final_raw(raw, support, list(range(42,47)))
    with pytest.raises(ValueError):
        validate_final_raw(raw.iloc[:-1], support, list(range(42,47)))
