import pandas as pd

from gog_fraud.evaluation.statistics import aggregate_seed_results, friedman_dataset_test, paired_model_tests


def _frame():
    rows = []
    for dataset_index, dataset in enumerate(("A", "B", "C", "D")):
        for model_index, model in enumerate(("M1", "M2", "M3")):
            for seed in (42, 43):
                rows.append({"dataset": dataset, "model": model, "seed": seed, "pr_auc": 0.9 - model_index * 0.2 + dataset_index * 0.01})
    return pd.DataFrame(rows)


def test_seed_aggregation_precedes_friedman_and_pairwise_tests():
    frame = _frame()
    assert len(aggregate_seed_results(frame, metric="pr_auc")) == 12
    result = friedman_dataset_test(frame, metric="pr_auc")
    assert result["n_datasets"] == 4
    assert result["n_models"] == 3
    assert len(paired_model_tests(frame, metric="pr_auc")) == 3

