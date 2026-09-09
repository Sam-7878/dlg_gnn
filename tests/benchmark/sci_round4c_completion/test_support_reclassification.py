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


def test_reddit_single_measured_gadnr_oom_accounts_second_seed_by_policy(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "reddit_gadnr_42.json").write_text(json.dumps({
        "dataset": "Reddit", "model": "GADNR", "seed": 42,
        "status": "failed_oom",
        "failure_message": (
            "CUDA out of memory. Tried to allocate 27.44 GiB in "
            "SAGEConv full_batch_neigh_recon"
        ),
        "total_wall_sec": 53.0,
    }), encoding="utf-8")

    rows = build_ledger({"backend": "exact"}, tmp_path)["classifications"]

    assert {(row["seed"], row["evidence_mode"]) for row in rows} == {
        (42, "measured"), (43, "policy")
    }
    assert {row["final_status"] for row in rows} == {
        "unsupported_resource_exact_implementation"
    }
