import json

from gog_fraud.pipelines.classify_sci_round4c_support import build_ledger


def test_compact_second_seed_trace_inherits_specific_reproduced_gadnr_stage(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    messages = {
        42: "CUDA out of memory. Tried to allocate 2.89 GiB. linalg_inv linalg_solve",
        43: "CUDA out of memory. Tried to allocate 2.89 GiB.",
    }
    for seed, message in messages.items():
        (raw / f"cell{seed}.json").write_text(json.dumps({
            "dataset": "Elliptic", "model": "GADNR", "seed": seed,
            "status": "failed_oom", "failure_message": message,
            "total_wall_sec": 60.0,
        }), encoding="utf-8")
    ledger = build_ledger({"backend": "exact"}, tmp_path)
    rows = ledger["classifications"]
    assert len(rows) == 2
    assert {row["oom_stage"] for row in rows} == {"neighborhood_covariance_inverse"}
    assert {row["evidence_mode"] for row in rows} == {"measured"}
