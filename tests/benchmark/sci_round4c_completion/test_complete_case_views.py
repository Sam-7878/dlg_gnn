import pandas as pd

from gog_fraud.experiments.round4c_completion import complete_case_views


def test_complete_case_views_never_impute_unsupported_cells():
    models = ["A", "AnomalyDAE", "GADNR"]
    datasets = ["Cora", "Fraud"]
    support = pd.DataFrame([
        {"dataset": dataset, "model": model,
         "primary_supported": not (dataset == "Fraud" and model != "A")}
        for dataset in datasets for model in models
    ])
    views = complete_case_views(support, models, datasets, ["Fraud"])
    full = views.loc[views.view_name.eq("full_comparable_subset")].iloc[0]
    scalable = views.loc[views.view_name.eq("scalable_detector_subset")].iloc[0]
    fraud = views.loc[views.view_name.eq("fraud_oriented_comparable_subset")].iloc[0]
    assert full.datasets == "Cora"
    assert scalable.models == "A"
    assert scalable.n_datasets == 2
    assert fraud.models == "A"
