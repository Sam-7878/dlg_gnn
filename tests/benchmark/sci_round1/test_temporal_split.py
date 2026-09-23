from gog_fraud.data.splits.temporal_split import temporal_split
from gog_fraud.data.validation.temporal_leakage import validate_temporal_integrity


def test_chronological_split_and_future_source_detection():
    records = [{"sample_id": str(index), "event_time": index} for index in range(10)]
    split = temporal_split(records, train_ratio=.6, valid_ratio=.2)
    assert max(map(int, split.train_ids)) < min(map(int, split.valid_ids))
    assert max(map(int, split.valid_ids)) < min(map(int, split.test_ids))
    audit = validate_temporal_integrity([
        {"sample_id": "x", "event_time": 5, "feature_source_max_time": 6, "relation_source_max_time": 5}
    ])
    assert not audit.valid
    assert audit.issues[0].source == "feature"
