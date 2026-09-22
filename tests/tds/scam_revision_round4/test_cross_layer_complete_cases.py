import pandas as pd

from graphrag.scam_revision.round4_final_evidence import complete_cross_layer_cases


def test_complete_cases_keep_identical_real_lineage_support():
    frame = pd.DataFrame([
        {"sample_id": "a", "label": 1, "p_rag": .9, "p_gnn": .8, "onchain_transaction_hash": "real"},
        {"sample_id": "b", "label": 0, "p_rag": .1, "p_gnn": .2, "onchain_transaction_hash": ""},
    ])
    result = complete_cross_layer_cases(frame)
    assert result[["sample_id", "label"]].to_dict("records") == [{"sample_id": "a", "label": 1}]
