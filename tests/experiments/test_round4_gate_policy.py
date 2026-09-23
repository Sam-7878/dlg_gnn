import json

from gog_fraud.experiments.round4_policy import assess_paper_eligibility, check_main_prerequisites


def _dataset(root, *, leakage="PASS", split_hash="abc", legacy="INCOMPLETE"):
    for name in ("manifests", "mappings", "splits", "labels", "audit"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "labels/label_semantics.json").write_text(json.dumps({"semantic_status": "RESOLVED"}))
    (root / "audit/leakage_audit_all.json").write_text(json.dumps({"status": leakage, "violations": 0}))
    summary = {"dataset_version": "gog-sci-v2.0", "semantic_status": "RESOLVED", "chains": {}}
    for chain in ("ethereum", "bsc", "polygon"):
        (root / f"normalizers/{chain}/holdout").mkdir(parents=True)
        (root / f"relations/{chain}/holdout").mkdir(parents=True)
        manifest = {"manifest_complete": True, "files_failed": 0, "files_succeeded": 1,
                    "records": [{}], "legacy_mapping_status": legacy}
        (root / f"manifests/{chain}.json").write_text(json.dumps(manifest))
        (root / f"mappings/{chain}_raw_to_graph.json").write_text(json.dumps([{}]))
        (root / f"splits/{chain}_holdout_v2.json").write_text(json.dumps({"split_hash": split_hash}))
        (root / f"normalizers/{chain}/holdout/normalizer.json").write_text(json.dumps({"fit_scope": "train_only", "fit_hash": "n"}))
        (root / f"relations/{chain}/holdout/relation_state.json").write_text(json.dumps({"future_nodes_included": 0, "future_relations_included": 0}))
    (root / "manifests/dataset_summary.json").write_text(json.dumps(summary))


def test_v2_complete_and_legacy_partial_authorizes_main(tmp_path):
    root = tmp_path / "data"; _dataset(root)
    lock = tmp_path / "requirements.lock"; lock.write_text("torch==1")
    result = check_main_prerequisites(root, git_clean_at_start=True, dependency_lock=lock)
    assert result.authorized
    assert set(result.evidence["legacy_compatibility"].values()) == {"PARTIAL"}


def test_leakage_incomplete_blocks(tmp_path):
    root = tmp_path / "data"; _dataset(root, leakage="INCOMPLETE")
    lock = tmp_path / "lock"; lock.write_text("x")
    assert not check_main_prerequisites(root, git_clean_at_start=True, dependency_lock=lock).authorized


def test_dirty_git_blocks(tmp_path):
    root = tmp_path / "data"; _dataset(root)
    lock = tmp_path / "lock"; lock.write_text("x")
    assert not check_main_prerequisites(root, git_clean_at_start=False, dependency_lock=lock).authorized


def test_missing_split_hash_blocks(tmp_path):
    root = tmp_path / "data"; _dataset(root, split_hash="")
    lock = tmp_path / "lock"; lock.write_text("x")
    assert not check_main_prerequisites(root, git_clean_at_start=True, dependency_lock=lock).authorized


def test_demo_metrics_block_eligibility():
    record = {"dataset_version": "gog-sci-v2.0", "leakage_audit_status": "PASS",
              "split_hash": "s", "run_manifest": "m", "resolved_config": "c",
              "git_clean_at_start": True, "real_model_inference": True,
              "demo_or_synthetic_metric": True, "sample_count_consistent": True,
              "status": "SUCCESS"}
    eligible, reasons = assess_paper_eligibility(record)
    assert not eligible and "demo/synthetic metric detected" in reasons


def test_complete_real_record_is_eligible_despite_legacy_absence():
    record = {"dataset_version": "gog-sci-v2.0", "leakage_audit_status": "PASS",
              "split_hash": "s", "run_manifest": "m", "resolved_config": "c",
              "git_clean_at_start": True, "real_model_inference": True,
              "demo_or_synthetic_metric": False, "sample_count_consistent": True,
              "status": "SUCCESS", "legacy_compatibility": "PARTIAL"}
    assert assess_paper_eligibility(record) == (True, [])


def test_pilot_scope_is_not_an_eligibility_input():
    """The runner, not the scientific record validator, excludes pilot rows."""
    record = {"dataset_version": "gog-sci-v2.0", "leakage_audit_status": "PASS",
              "split_hash": "s", "run_manifest": "m", "resolved_config": "c",
              "git_clean_at_start": True, "real_model_inference": True,
              "demo_or_synthetic_metric": False, "sample_count_consistent": True,
              "status": "SUCCESS", "experiment_scope": "EXPLORATORY"}
    assert assess_paper_eligibility(record)[0]
