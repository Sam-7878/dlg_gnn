from gog_fraud.data.io.streaming_dataset import StatefulTransactionStream
from gog_fraud.selection.router import SelectiveRouter, TriageOutput
from gog_fraud.streaming.engine import StatefulStreamingEngine


def test_engine_emits_routing_trace_and_stable_prediction_hash():
    records = [{"sample_id": f"s{i}", "chain_id": "eth", "contract_id": "c", "event_time": i} for i in range(3)]
    def triage(event): return TriageOutput(event.event_time / 2, 0.0, 0.0, 0.0, None, 1)
    router = SelectiveRouter(tau_b=0.2, tau_f=0.8, tau_u=0.1, threshold_version="t1")
    one = StatefulStreamingEngine(stream=StatefulTransactionStream(records), router=router, triage_fn=triage, deep_fn=lambda event: 0.4, model_version="m1")
    two = StatefulStreamingEngine(stream=StatefulTransactionStream(records), router=router, triage_fn=triage, deep_fn=lambda event: 0.4, model_version="m1")
    assert [trace.route for trace in one.run()] == ["benign_direct", "deep_inspection", "fraud_direct"]
    two.run()
    assert one.prediction_hash == two.prediction_hash
