import pandas as pd

from gog_fraud.pipelines.analyze_sci_round4c import build_round5_protocol


def test_protocol_is_created_only_for_ready_decisions():
    config={"models":["M"],"round5":{"model_seeds":[42,43,44,45,46],"output_root":"out"},
            "training":{"epochs":50,"dlg_l1_epochs":20}}
    support=pd.DataFrame([{"dataset":"D","model":"M","primary_supported":True}])
    assert build_round5_protocol(config,support,"NOT_READY") is None
    assert build_round5_protocol(config,support,"READY_FOR_FULL_RUN")["execution"]["message_backend"] == "sparse_fused"

