import pandas as pd

from graphrag.scam_revision.round4_final_evidence import semantic_alignment_supported


def test_unadjudicated_shuffle_experiment_cannot_complete():
    frame = pd.DataFrame([{
        "seed": 7, "method": "GraphRAG real text", "auc_pr": 0.9,
        "control_status": "unverified_control",
    }])
    result = semantic_alignment_supported(frame)
    assert not result["test_complete"]
    assert not result["claim_supported"]
