import csv


def test_report_hashes_match_authoritative_manifest(d4_root):
    with (d4_root / "tables/hash_reconciliation.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    authoritative = [row for row in rows if row["authoritative"] == "True"]
    report_rows = [row for row in rows if "D3 report" in row["artifact"]]
    assert authoritative and all(row["match"] == "True" for row in authoritative)
    assert report_rows and any(row["match"] == "False" for row in report_rows)
