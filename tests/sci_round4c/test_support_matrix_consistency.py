import pandas as pd

from gog_fraud.experiments.round4c_policy import final_support_matrix
from gog_fraud.pipelines.analyze_sci_round4c import _component_summary, _save_figures


def test_support_requires_two_production_successes():
    frame=pd.DataFrame([
        {"dataset":"D","model":"M","seed":42,"status":"success"},
        {"dataset":"D","model":"M","seed":43,"status":"failed_cuda"},
    ])
    row=final_support_matrix(frame).iloc[0]
    assert row.production_tested
    assert not row.primary_supported
    assert row.restriction == "unexpected_failure_or_incomplete"


def test_partial_support_heatmap_treats_unattempted_cells_as_unsupported(tmp_path):
    (tmp_path / "figures").mkdir()
    support = pd.DataFrame([
        {"dataset": "A", "model": "M1", "primary_supported": True},
        {"dataset": "A", "model": "M2", "primary_supported": False},
        {"dataset": "B", "model": "M1", "primary_supported": False},
    ])
    _save_figures(
        tmp_path,
        pd.DataFrame(),
        support,
        pd.DataFrame(),
        pd.DataFrame(),
    )
    assert (tmp_path / "figures" / "06_model_dataset_support_matrix.png").is_file()


def test_local_dominant_allows_fusion_to_recover_nearly_all_local_signal(tmp_path):
    ablation = tmp_path / "ablation"
    ablation.mkdir()
    pd.DataFrame([
        {"dataset": "D", "seed": 42, "variant": variant,
         "pr_auc": pr_auc, "validation_f1": pr_auc}
        for variant, pr_auc in (
            ("DLG-Base", .08), ("DLG-Aug", .09),
            ("DLG-Local", .31), ("DLG-Fusion", .305),
        )
    ]).to_csv(ablation / "component_raw.csv", index=False)
    _, summary = _component_summary(tmp_path)
    assert summary.iloc[0].descriptive_class == "local-dominant"
    assert ">0.02" in summary.iloc[0].classification_basis
