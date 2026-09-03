from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/graphrag/scam_revision_round3"


def test_five_seed_prediction_ids_and_prevalence_match_frozen_test_manifest():
    manifest = pd.read_parquet(RESULTS / "evaluation_sample_manifests/natural_temporal.parquet")
    expected = manifest[manifest.split_name == "test"].sort_values("sample_id")
    files = sorted((RESULTS / "raw_predictions").glob("seed*_natural_temporal.parquet"))
    assert len(files) == 5
    for path in files:
        predictions = pd.read_parquet(path).sort_values("sample_id")
        assert predictions.sample_id.is_unique
        assert predictions.sample_id.tolist() == expected.sample_id.tolist()
        assert predictions.label.astype(int).tolist() == expected.label.astype(int).tolist()
        assert predictions.label.astype(int).mean() == expected.label.astype(int).mean()
        assert predictions.label_manifest_version.nunique() == 1


def test_entity_disjoint_manifests_report_real_support_and_unavailable_metrics():
    support = pd.read_csv(RESULTS / "entity_disjoint_support.csv").set_index("track")
    assert bool(support.loc["campaign_disjoint", "two_class_support"])
    assert bool(support.loc["campaign_disjoint", "paper_eligible"])
    assert bool(support.loc["wallet_disjoint", "two_class_support"])
    assert not bool(support.loc["wallet_disjoint", "paper_eligible"])
    assert not bool(support.loc["domain_disjoint", "two_class_support"])
    results = pd.read_csv(RESULTS / "entity_disjoint_results.csv")
    domain = results[results.track == "domain_disjoint"]
    assert domain.roc_auc.isna().all() and domain.auc_pr.isna().all()
    assert (domain.status == "unavailable_one_class_support").all()
