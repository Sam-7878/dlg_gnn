import pandas as pd

from gog_fraud.experiments.round4c_completion import (
    account_cells, completion_decision, frozen_support_matrix,
)


def test_measured_and_policy_unsupported_cells_are_accounted_without_erasing_observation():
    raw = pd.DataFrame([
        {"dataset": "D", "model": "M", "seed": 42, "status": "failed_oom",
         "failure_message": "OOM"},
    ])
    ledger = pd.DataFrame([
        {"dataset": "D", "model": "M", "seed": 42,
         "final_status": "unsupported_resource_exact_implementation",
         "evidence_mode": "measured", "restriction_reason": "reproduced materialization",
         "evidence_path": "failures/a.json"},
        {"dataset": "D", "model": "M", "seed": 43,
         "final_status": "unsupported_resource_exact_implementation",
         "evidence_mode": "policy", "restriction_reason": "deterministic N/E complexity",
         "evidence_path": "failures/a.json"},
    ])
    accounting = account_cells(raw, ledger, [("D", "M", 42), ("D", "M", 43)])
    assert accounting.accounted.all()
    assert accounting.loc[accounting.seed.eq(42), "observed_status"].item() == "failed_oom"
    support = frozen_support_matrix(accounting).iloc[0]
    assert support.seed42_status.endswith("_measured")
    assert support.seed43_status.endswith("_by_policy")
    assert not support.primary_supported
    assert not support.production_tested
    assert completion_decision(accounting)[0] == "READY_WITH_RESTRICTIONS"


def test_unattempted_unclassified_cell_keeps_gate_closed():
    accounting = account_cells(
        pd.DataFrame(columns=["dataset", "model", "seed", "status"]),
        pd.DataFrame(columns=["dataset", "model", "seed", "final_status",
                              "evidence_mode", "restriction_reason", "evidence_path"]),
        [("D", "M", 42)],
    )
    assert completion_decision(accounting)[0] == "NOT_READY"

