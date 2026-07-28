from gog_fraud.experiments.result_schema import REQUIRED_RESULT_COLUMNS, validate_result_record


def test_result_schema_accepts_complete_record_and_rejects_bad_counts():
    record = {column: None for column in REQUIRED_RESULT_COLUMNS}
    record.update(split="test", status="success", num_samples=3, num_pos=1, num_neg=2)
    assert validate_result_record(record) == []
    record["num_neg"] = 1
    assert "num_pos + num_neg must equal num_samples" in validate_result_record(record)
