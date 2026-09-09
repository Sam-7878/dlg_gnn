import json


def test_lanl_gt_coverage(d4_root):
    audit = json.loads((d4_root / "defense_validation/lanl_ground_truth_freeze.json").read_text(encoding="utf-8"))
    assert audit["positive_nodes"] == 301
    assert audit["official_redteam_events"] == audit["events_within_30_day_cutoff"] == 749
    assert audit["max_redteam_time_seconds"] <= 30 * 86400
    assert audit["all_positive_ids_exist_in_final_graph"] is True
    assert audit["all_cutoff_destination_ids_accounted"] is True
