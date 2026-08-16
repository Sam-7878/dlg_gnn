import json

import yaml

from gog_fraud.pipelines.run_sci_round4c import result_path
from gog_fraud.pipelines.run_sci_round4c_guard_resume import resume_guard


def test_completed_cumulative_guard_does_not_repeat_full_runtime(tmp_path, monkeypatch):
    output = tmp_path / "out"
    (output / "resources").mkdir(parents=True)
    config = {
        "experiment": {"output_root": str(output)},
        "execution": {"max_run_wall_hours": 24},
        "training": {"epochs": 50},
        "display_names": {"DGraphFin": "DGraphFin"},
        "backend": {"message": "sparse_fused"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    state = resume_guard(config_path, dataset="DGraphFin", seed=42,
                         prior_active_sec=24 * 3600)
    assert state["guard_completed"]
    record = json.loads(result_path(config, output, "DGraphFin", "AnomalyDAE", 42).read_text())
    assert record["status"] == "unsupported_operational"
    assert record["total_wall_sec"] == 24 * 3600


def test_measured_epoch_lower_bound_can_complete_guard_decision_without_wasting_gpu(tmp_path):
    output = tmp_path / "out"
    (output / "resources").mkdir(parents=True)
    (output / "resources" / "anomalydae_dgraphfin_live_calibration.json").write_text(
        json.dumps({"epoch_checkpoints": [
            {"completed_epoch": 1, "estimated_active_elapsed_minutes": 135.0},
            {"completed_epoch": 2, "estimated_active_epoch_minutes": 266.5},
            {"completed_epoch": 3, "estimated_active_epoch_minutes": 293.5},
            {"completed_epoch": 4, "estimated_active_epoch_minutes": 319.8},
        ]}), encoding="utf-8"
    )
    config = {
        "experiment": {"output_root": str(output)},
        "execution": {"max_run_wall_hours": 24},
        "training": {"epochs": 50},
        "display_names": {"DGraphFin": "DGraphFin"},
        "backend": {"message": "sparse_fused"},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    state = resume_guard(config_path, dataset="DGraphFin", seed=42,
                         prior_active_sec=63064)
    assert state["guard_decision_complete"]
    assert not state["guard_completed"]
    assert state["completion_impossible_within_guard"]
    assert state["optimistic_remaining_runtime_sec"] == 46 * 135 * 60
    assert not result_path(config, output, "DGraphFin", "AnomalyDAE", 42).exists()
