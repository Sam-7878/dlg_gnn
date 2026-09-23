import json

from gog_fraud.pipelines.run_sci_evaluation import _validate_prerequisites


def test_prerequisites_fail_closed_on_incomplete_leakage_audit(tmp_path):
    (tmp_path / "results/manifests").mkdir(parents=True)
    (tmp_path / "results/splits").mkdir(parents=True)
    manifest = {
        "chain": "ethereum", "manifest_complete": True, "files_failed": 0,
        "canonical_root": "/data/GoG",
    }
    (tmp_path / "results/manifests/ethereum.json").write_text(json.dumps(manifest), encoding="utf-8")
    for name, payload in (
        ("ethereum_holdout_v1.json", {}), ("ethereum_rolling5_v1.json", {}),
        ("ethereum_leakage_audit_v1.json", {"status": "INCOMPLETE", "violations": 0}),
    ):
        (tmp_path / "results/splits" / name).write_text(json.dumps(payload), encoding="utf-8")
    config = {"dataset": {"version": "v1", "canonical_root": "/data/GoG", "physical_transaction_root": "/raw", "labels_path": "/labels", "chains": ["ethereum"]}}
    errors, _ = _validate_prerequisites(config, tmp_path / "results", require_clean_git=False, repo_root=tmp_path)
    assert errors == ["sample-level leakage audit not PASS: ethereum"]
