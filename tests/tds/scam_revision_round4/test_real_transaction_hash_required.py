import pandas as pd

from graphrag.scam_revision.round4_final_evidence import TRANSACTION_COLUMNS, valid_real_transactions


def test_real_hash_timestamp_source_are_all_required():
    row = {column: "x" for column in TRANSACTION_COLUMNS}
    row.update({"transaction_hash": "0x" + "a" * 64, "block_timestamp": "2024-01-01T00:00:00Z", "chain": "ethereum", "source": "archive"})
    frame = pd.DataFrame([row, {**row, "transaction_hash": "synthetic-1"}, {**row, "block_timestamp": ""}])
    assert len(valid_real_transactions(frame)) == 1
