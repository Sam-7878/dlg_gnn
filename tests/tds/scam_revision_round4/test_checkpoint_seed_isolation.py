import pandas as pd

from graphrag.scam_revision.round4_final_evidence import CHECKPOINT_SEEDS, validate_checkpoint_isolation


def test_observed_and_permuted_checkpoints_are_unique():
    rows = []
    for seed in CHECKPOINT_SEEDS:
        for mode in ("observed", "permuted"):
            key = f"{seed}-{mode}"
            rows.append({"seed": seed, "mode": mode, "checkpoint_path": key, "checkpoint_sha256": "sha-" + key,
                         "train_id_hash": "tr", "val_id_hash": "va", "test_id_hash": "te", "label_hash": key})
    assert validate_checkpoint_isolation(pd.DataFrame(rows))["pass"]
    rows[-1]["checkpoint_path"] = rows[0]["checkpoint_path"]
    assert not validate_checkpoint_isolation(pd.DataFrame(rows))["pass"]
