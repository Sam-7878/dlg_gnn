import csv

import _round3_bootstrap  # noqa: F401

from experiments.round3.artifact_paths import ROUND3_RESULTS


def test_latency_is_measured_and_component_complete():
    with (ROUND3_RESULTS / "real_e2e_latency.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {int(row["T"]) for row in rows} == {1, 5, 10, 20, 30}
    required = {
        "mean_total_ms", "mean_gnn_ms", "mean_graphrag_ms",
        "mean_risk_encoder_ms", "mean_fusion_ms", "mean_serialization_ms",
    }
    assert required <= rows[0].keys()
    assert all(float(row["mean_total_ms"]) > 0 for row in rows)
    assert all(row["paper_eligible"] == "False" for row in rows)
