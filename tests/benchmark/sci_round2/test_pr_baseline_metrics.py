def test_pr_baseline_gain_and_lift_use_test_prevalence():
    prevalence, pr_auc = .02, .10
    assert pr_auc / prevalence == 5.0
    assert pr_auc - prevalence == .08

