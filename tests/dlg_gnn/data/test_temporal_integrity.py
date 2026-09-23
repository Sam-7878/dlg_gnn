from gog_fraud.data.splits.rolling_origin import rolling_origin_splits
from gog_fraud.data.splits.temporal_split import temporal_split
from gog_fraud.data.validation.temporal_leakage import validate_temporal_integrity


def records(n=20):
    return [{"sample_id": f"s{i:02d}", "event_time": i, "feature_source_max_time": i, "relation_source_max_time": i} for i in range(n)]


def test_temporal_split_is_stable_and_disjoint():
    first = temporal_split(reversed(records()))
    second = temporal_split(records())
    assert first.split_hash == second.split_hash
    assert not (set(first.train_ids) & set(first.test_ids))
    assert first.train_end < first.valid_end


def test_rolling_origin_expands_training_window():
    folds = rolling_origin_splits(records(30), n_folds=5)
    assert len(folds) == 5
    assert all(len(a.train_ids) < len(b.train_ids) for a, b in zip(folds, folds[1:]))
    assert all(set(f.train_ids).isdisjoint(f.test_ids) for f in folds)


def test_future_feature_relation_and_scaler_are_rejected():
    data = records(3)
    data[1]["feature_source_max_time"] = 99
    report = validate_temporal_integrity(data, scaler_fit_end=10, train_end=2)
    assert not report.valid
    assert {issue.source for issue in report.issues} == {"feature", "scaler"}


def test_entity_overlap_is_reported_not_hidden():
    report = validate_temporal_integrity([], split_entities={"train": ["a", "b"], "valid": ["b"], "test": ["a"]})
    assert report.entity_overlap == {"train_valid": 1, "train_test": 1, "valid_test": 0}
