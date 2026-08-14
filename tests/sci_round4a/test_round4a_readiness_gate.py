from gog_fraud.experiments.round4a_reconstruction import (
    CRITICAL_GATES,
    decide_round4a_readiness,
    receptive_field_manifest,
)


def test_readiness_fails_closed():
    gates = {name: True for name in CRITICAL_GATES}
    gates["reddit_resource"] = False
    assert decide_round4a_readiness(gates) == "NOT_READY"


def test_receptive_field_manifest_does_not_claim_one_hop_equivalence():
    rows = {row["model"]: row for row in receptive_field_manifest()}
    assert rows["DLG-Aug"]["effective_receptive_field_hops"] == 6
    assert rows["DLG-Aug"]["partition_halo_hops"] == 0
    assert rows["DLG-Aug"]["representation_equivalence_expected"] is True
