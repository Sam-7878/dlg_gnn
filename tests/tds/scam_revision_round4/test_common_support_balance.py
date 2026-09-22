import pandas as pd

from graphrag.scam_revision.round4_final_evidence import balance_diagnostics, common_support_balance_pass


def test_balance_requires_every_covariate_below_point_one():
    balanced = pd.DataFrame({"label": [1, 1, 0, 0], "degree": [1, 2, 1, 2], "time": [3, 4, 3, 4]})
    assert common_support_balance_pass(balance_diagnostics(balanced, ["degree", "time"]))
    imbalanced = balanced.copy(); imbalanced.loc[imbalanced.label == 1, "degree"] += 10
    assert not common_support_balance_pass(balance_diagnostics(imbalanced, ["degree", "time"]))
