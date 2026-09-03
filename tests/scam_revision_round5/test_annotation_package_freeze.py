import pandas as pd

from experiments.round5.analysis import ANNOTATION_VERSION, freeze_annotation_package, sha256_file


def test_annotation_package_has_no_model_hints_and_is_hashed(tmp_path):
    source = tmp_path / "source.csv"
    pd.DataFrame([{
        "sample_id": "s1", "campaign_id": "c1", "campaign_time": "2024-01-01",
        "campaign_title/text": "title", "promoted_urls": "[]", "wallets": "[]",
        "domains": "[]", "source_links/identifiers": "source:1",
        "CST_exact_hit": False, "CSDB_exact_hit": False,
        "annotation_1": "BENIGN", "annotation_2": "BENIGN",
        "final_label": "BENIGN", "reason": "prefilled",
    }]).to_csv(source, index=False)
    manifest = freeze_annotation_package(source, tmp_path / "out")
    package = pd.read_csv(tmp_path / "out" / "annotation_package_v1.csv", keep_default_na=False)
    assert manifest["annotation_package_version"] == ANNOTATION_VERSION
    assert manifest["package_sha256"] == sha256_file(tmp_path / "out" / "annotation_package_v1.csv")
    assert not package[["annotation_1", "annotation_2", "final_label", "reason"]].any(axis=None)
