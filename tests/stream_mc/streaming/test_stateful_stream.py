from gog_fraud.data.io.streaming_dataset import StatefulTransactionStream


def events():
    return [{"sample_id": f"s{i}", "chain_id": "eth", "contract_id": "c", "event_time": i, "block_number": i, "transaction_index": 0} for i in range(6)]


def test_replay_and_restore_are_equivalent():
    stream = StatefulTransactionStream(reversed(events()))
    iterator = iter(stream)
    prefix = [next(iterator).sample_id for _ in range(2)]
    checkpoint = stream.checkpoint()
    suffix_a = [event.sample_id for event in iterator]
    stream.restore(checkpoint)
    suffix_b = [event.sample_id for event in stream]
    assert prefix == ["s0", "s1"]
    assert suffix_a == suffix_b


def test_malformed_records_are_quarantined():
    stream = StatefulTransactionStream(events() + [{"sample_id": "bad"}])
    assert len(list(stream)) == 6
    assert stream.quarantine[0]["record_index"] == 6


def test_merged_multichain_stream_is_chronological():
    left = StatefulTransactionStream(events()[::2])
    right_data = [{**item, "chain_id": "bsc"} for item in events()[1::2]]
    merged = StatefulTransactionStream.merged([left, StatefulTransactionStream(right_data)])
    assert [event.event_time for event in merged] == list(range(6))
