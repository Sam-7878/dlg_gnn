import csv


def test_no_hardcoded_round5_metadata(d4_root):
    source = open("scripts/defense_extension_real/finalize_round_d4.py", encoding="utf-8").read()
    assert 'manifests/data_freeze.json' in source
    with (d4_root / "tables/dataset_metadata_reconciliation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert all(row["final_match"] == "True" for row in rows)
    assert all(row["authoritative_source"].endswith("data_freeze.json") for row in rows)
