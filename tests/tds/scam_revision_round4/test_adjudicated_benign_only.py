import pandas as pd

from graphrag.scam_revision.round4_final_evidence import adjudicated_benign_only


def test_only_double_benign_final_consensus_is_retained():
    frame = pd.DataFrame([
        {"sample_id": "ok", "annotation_1": "BENIGN", "annotation_2": "BENIGN", "final_label": "BENIGN"},
        {"sample_id": "blank", "annotation_1": "", "annotation_2": "", "final_label": ""},
        {"sample_id": "disagree", "annotation_1": "BENIGN", "annotation_2": "SCAM", "final_label": "BENIGN"},
    ])
    assert adjudicated_benign_only(frame).sample_id.tolist() == ["ok"]
