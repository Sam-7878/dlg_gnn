import csv
import json


def test_dataset_characteristics_from_frozen_manifest(d4_root):
    freeze = json.loads(open("outputs/sci_round5_final/manifests/data_freeze.json", encoding="utf-8").read())
    expected = {row["dataset"]: (row["nodes"], row["edges"], row["features"]) for row in freeze["datasets"]}
    with (d4_root / "tables/table_d4_dataset_portfolio.csv").open(newline="", encoding="utf-8") as handle:
        portfolio = list(csv.DictReader(handle))
    primary = [row for row in portfolio if row["role"] == "primary_frozen_performance"]
    assert len(primary) == 10
    for row in primary:
        assert (int(row["N"]), int(row["E"]), int(row["F"])) == expected[row["dataset"]]
        assert row["metadata_source"].endswith("data_freeze.json")
