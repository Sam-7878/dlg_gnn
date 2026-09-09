import pandas as pd

from gog_fraud.experiments.round5_policy import validate_final_raw


def test_unsupported_pair_has_zero_performance_rows():
    support=pd.DataFrame([{"dataset":"D","model":"M","support_status":"unsupported_operational"}])
    validate_final_raw(pd.DataFrame(columns=["dataset","model","seed","status"]),support,[42,43,44,45,46])
