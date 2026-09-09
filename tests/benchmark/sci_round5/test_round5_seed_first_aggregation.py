import pandas as pd

from gog_fraud.experiments.round5_policy import seed_first_summary


def test_seed_summary_has_one_row_and_sample_std():
    raw = pd.DataFrame([{"dataset":"D","model":"M","seed":seed,"roc_auc":value}
                        for seed,value in zip(range(42,47),[.1,.2,.3,.4,.5])])
    result = seed_first_summary(raw, ["roc_auc"])
    assert len(result) == 1
    assert result.iloc[0].n_seeds == 5
    assert result.iloc[0].roc_auc_std > 0
