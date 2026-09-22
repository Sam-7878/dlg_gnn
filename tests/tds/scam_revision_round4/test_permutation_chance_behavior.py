import pandas as pd

from graphrag.scam_revision.round4_final_evidence import CHECKPOINT_SEEDS, validate_permutation_chance


def test_every_seed_must_be_near_chance():
    frame = pd.DataFrame([{"seed": seed, "mode": "permuted", "roc_auc": .5, "auc_pr": .3} for seed in CHECKPOINT_SEEDS])
    assert validate_permutation_chance(frame, prevalence=.3)
    frame.loc[frame.seed == 47, "roc_auc"] = .9
    assert not validate_permutation_chance(frame, prevalence=.3)
