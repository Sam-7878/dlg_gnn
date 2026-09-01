from __future__ import annotations

import torch

from gog_fraud.data.io.streaming_dataset import StreamEvent
from profiling.raw_event_selective_e2e_profiler import decide, graph_from_state, tabular_feature


def test_direct_only_and_no_routing_are_executable_branches() -> None:
    assert decide("direct_only", 0.5, 0.5, 0.1, 0.05, 0.0) is False
    assert decide("no_routing", 0.0, 0.5, 0.1, 0.05, 0.0) is True


def test_raw_state_features_feed_production_shape() -> None:
    state = {"contract_id": "c", "event_time": 2, "nodes": ["a", "b"], "edges": [("a", "b", 2)]}
    graph = graph_from_state(state, 1, torch.device("cpu"))
    assert tuple(graph.x.shape) == (2, 3)
    event = StreamEvent(event_time=2, block_number=1, transaction_index=0, sample_id="s", chain_id="ethereum", contract_id="c", payload={"value": 3.0, "label": 1})
    feature = tabular_feature(event, state, {})
    assert tuple(feature.shape) == (1, 11)
    assert torch.isfinite(graph.x).all()
