import json

from gog_fraud.experiments.round4c_policy import classify_timeout
from gog_fraud.pipelines.run_sci_round4c import _timeout_record, hashes, result_path


def test_only_predeclared_anomalydae_timeout_is_operationally_unsupported():
    assert classify_timeout("AnomalyDAE") == "unsupported_operational"
    assert classify_timeout("DOMINANT") == "failed_other"


def test_timeout_record_preserves_checkpoint_epoch_progress(tmp_path):
    config = {
        "training": {"epochs": 50},
        "display_names": {"D": "Dataset"},
        "backend": {"message": "sparse_fused"},
    }
    config_hash, backend_hash = hashes(config)
    from gog_fraud.experiments.round4c_policy import cell_key
    key = cell_key("D", "AnomalyDAE", 42, config_hash, backend_hash)
    checkpoint_dir = tmp_path / "checkpoints"
    raw_dir = tmp_path / "raw"
    checkpoint_dir.mkdir()
    raw_dir.mkdir()
    (checkpoint_dir / f"{key}.pt.progress.json").write_text(
        json.dumps({"completed_epochs": 7}), encoding="utf-8"
    )

    _timeout_record(config, tmp_path, "D", "AnomalyDAE", 42, 86400)
    record = json.loads(result_path(config, tmp_path, "D", "AnomalyDAE", 42).read_text())
    assert record["status"] == "unsupported_operational"
    assert record["actual_epochs"] == 7
