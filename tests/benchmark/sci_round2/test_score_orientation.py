from gog_fraud.evaluation.score_semantics import audit_score_orientation


def test_orientation_audit_warns_but_does_not_invert():
    result = audit_score_orientation([0, 0, 1, 1], [9, 8, 2, 1])
    assert result["orientation_warning"]
    assert result["orientation_action"] == "report_only_no_silent_inversion"

