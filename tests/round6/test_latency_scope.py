import json

import pandas as pd

from experiments.round6.latency import audit_latency


def test_latency_is_labeled_as_full_panel(tmp_path):
    source = tmp_path / "mc.csv"
    rows = []
    for seed in (7, 17, 27, 37, 47):
        for passes in (1, 10):
            rows.append({
                "seed": seed, "T": passes, "n_test": 3648,
                "latency_ms": 100 + passes, "auc_pr": 0.4, "ece": 0.1,
            })
    pd.DataFrame(rows).to_csv(source, index=False)
    legacy = tmp_path / "e2e.csv"
    legacy.write_text("T,median_total_ms\n1,2.0\n")
    output = tmp_path / "latency.json"
    payload = audit_latency(source, legacy, output, tmp_path / "figure.png")
    assert payload["latency_scope_consistent"] is True
    assert "complete held-out panel" in payload["unit"]
    assert "not single-event" in payload["single_event_or_batch"]
    assert json.loads(output.read_text())["sample_count"]["events_per_seed_T"] == 3648

