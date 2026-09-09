import pandas as pd

from gog_fraud.pipelines.analyze_sci_round4c_completion import (
    gadnr_oom_audit, long_limit_preclassification,
)


def test_gadnr_oom_audit_extracts_allocator_evidence_and_specific_stage():
    raw = pd.DataFrame([
        {"dataset": "D", "model": "GADNR", "seed": 42, "status": "failed_oom",
         "failure_message": "Tried to allocate 2.89 GiB; allocated memory 19.01 GiB; "
                            "0.88 GiB is reserved but unallocated; linalg_inv"},
        {"dataset": "D", "model": "GADNR", "seed": 43, "status": "failed_oom",
         "failure_message": "Tried to allocate 2.89 GiB"},
    ])
    audit = gadnr_oom_audit(raw, {"datasets": [{"dataset": "D", "nodes": 10, "edges": 20}]})
    assert set(audit.oom_stage) == {"neighborhood_covariance_inverse"}
    assert audit.iloc[0].requested_allocation_gib == 2.89
    assert audit.iloc[0].N == 10


def test_long_limit_preclassification_is_flag_only():
    preflight = pd.DataFrame([
        {"dataset": "tiny", "N": 1000, "E": 2000, "F": 8},
        {"dataset": "large", "N": 100000, "E": 1000000, "F": 8},
    ])
    result = long_limit_preclassification(preflight)
    assert set(result.preclassification).issubset({
        "likely_supported", "requires_preflight", "likely_unsupported"
    })
    assert not result.preclassification.str.startswith("unsupported_").any()
