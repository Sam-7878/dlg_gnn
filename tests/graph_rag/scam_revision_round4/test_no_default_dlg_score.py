import numpy as np
import pandas as pd

from graphrag.scam_revision.round4_final_evidence import complete_cross_layer_cases


def test_missing_dlg_is_not_defaulted_into_complete_cases():
    frame = pd.DataFrame([
        {"sample_id": "complete", "label": 1, "p_rag": .8, "p_gnn": .7, "onchain_transaction_hash": "h"},
        {"sample_id": "missing", "label": 0, "p_rag": .2, "p_gnn": np.nan, "onchain_transaction_hash": ""},
    ])
    assert complete_cross_layer_cases(frame).sample_id.tolist() == ["complete"]
