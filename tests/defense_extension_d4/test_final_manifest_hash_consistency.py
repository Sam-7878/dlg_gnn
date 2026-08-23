import csv
import json


def test_final_manifest_hash_consistency(d4_root):
    manifest = json.loads((d4_root / "manifests/defense_final_source_of_truth.json").read_text(encoding="utf-8"))
    integrity = json.loads((d4_root / "manifests/bundle_integrity.json").read_text(encoding="utf-8"))
    assert manifest["round5"]["frozen_unchanged"] is True
    assert manifest["round5"]["successful_supported_runs"] == 355
    assert integrity["all_authoritative_hash_checks_pass"] is True
    with (d4_root / "tables/hash_reconciliation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["match"] == "True" for row in rows if row["authoritative"] == "True")
    for dataset in ("DARPA", "LANL"):
        assert all(row["declared_machine_manifest_match"] for row in manifest[dataset]["raw_files_actually_consumed"])
