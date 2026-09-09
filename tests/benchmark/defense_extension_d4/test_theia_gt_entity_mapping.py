import csv
import json


def test_theia_gt_entity_mapping(d4_root):
    with (d4_root / "source_audit/theia_ground_truth_entity_mapping.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    gate = json.loads((d4_root / "defense_validation/theia_performance_eligibility.json").read_text(encoding="utf-8"))
    assert rows
    assert {row["mapping_confidence"] for row in rows} <= {"direct", "strong_temporal_identifier_match", "unresolved"}
    assert all(row["mapping_confidence"] == "unresolved" for row in rows)
    assert gate["direct_mappings"] == 0
    assert gate["performance_eligible"] is False
    assert gate["evaluation_role"] == "scalability_only"
