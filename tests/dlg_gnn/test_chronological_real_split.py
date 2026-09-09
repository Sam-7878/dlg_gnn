import json

import _round3_bootstrap  # noqa: F401

from experiments.round3.artifact_paths import ROUND3_RESULTS


def test_current_controlled_dataset_is_not_mislabeled_chronological_real():
    manifest = json.loads((ROUND3_RESULTS / "real_dataset_manifest.json").read_text())
    assert manifest["split_type"] == "synthetic_time_ordered"
    assert manifest["timestamp_source"] == "synthetic_context_schedule"
    assert manifest["paper_eligible"] is False


def test_split_has_class_support_for_controlled_metrics():
    manifest = json.loads((ROUND3_RESULTS / "real_dataset_manifest.json").read_text())
    assert 0 < manifest["test_fraud"] < manifest["test_size"]
