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
