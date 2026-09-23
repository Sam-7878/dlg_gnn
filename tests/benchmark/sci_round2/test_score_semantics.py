from gog_fraud.evaluation.score_semantics import SCORE_SEMANTICS


def test_all_pilot_detectors_have_explicit_unbounded_higher_anomaly_contracts():
    assert len(SCORE_SEMANTICS) == 8
    assert all(item.higher_is_more_anomalous for item in SCORE_SEMANTICS.values())
    assert all(not item.probability_like for item in SCORE_SEMANTICS.values())

