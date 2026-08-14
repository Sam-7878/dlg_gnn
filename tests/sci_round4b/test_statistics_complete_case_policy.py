import pandas as pd

from gog_fraud.experiments.round4b_policy import complete_case_blocks


def test_complete_case_policy_does_not_impute_unsupported_as_worst_rank():
    frame = pd.DataFrame([
        {"dataset": "A", "model": "M1", "status": "success", "pr_auc": .8},
        {"dataset": "A", "model": "M2", "status": "success", "pr_auc": .7},
        {"dataset": "B", "model": "M1", "status": "success", "pr_auc": .6},
        {"dataset": "B", "model": "M2", "status": "unsupported_algorithmic", "pr_auc": None},
    ])
    blocks, metadata = complete_case_blocks(frame, models=["M1", "M2"], metric="pr_auc")
    assert list(blocks.index) == ["A"]
    assert metadata["n_blocks"] == 1
    assert metadata["missing_policy"] == "complete_cases_only_no_rank_imputation"
