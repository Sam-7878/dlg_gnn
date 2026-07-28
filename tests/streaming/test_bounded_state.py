from gog_fraud.data.io.streaming_dataset import StreamEvent
from gog_fraud.streaming.embedding_cache import EmbeddingCache
from gog_fraud.streaming.queue_manager import QueueManager
from gog_fraud.streaming.relation_state import IncrementalRelationState, RelationEdge
from gog_fraud.streaming.subgraph_store import IncrementalSubgraphStore


def event(index):
    return StreamEvent(index, index, 0, f"s{index}", "eth", "c", {"src": f"n{index}", "dst": f"n{index+1}"})


def test_subgraph_is_recent_bounded_and_snapshot_restores():
    store = IncrementalSubgraphStore(temporal_window_seconds=10, max_nodes_per_contract=4, max_edges_per_contract=2)
    for i in range(4): store.apply_event(event(i))
    graph = store.materialize("c", 4)
    assert len(graph["edges"]) == 2
    clone = IncrementalSubgraphStore(temporal_window_seconds=10, max_nodes_per_contract=4, max_edges_per_contract=2)
    clone.restore(store.snapshot())
    assert clone.materialize("c", 4) == graph


def test_embedding_cache_enforces_lru_ttl_and_versions():
    cache = EmbeddingCache(max_entries=2, max_bytes=10000, ttl_seconds=5)
    cache.put("a", [1], now=0, model_version="m1", feature_version="f1")
    cache.put("b", [2], now=0, model_version="m1", feature_version="f1")
    assert cache.get("a", now=1, model_version="m1", feature_version="f1") == [1]
    cache.put("c", [3], now=1, model_version="m1", feature_version="f1")
    assert cache.get("b", now=1, model_version="m1", feature_version="f1") is None
    assert cache.get("a", now=9, model_version="m1", feature_version="f1") is None


def test_queue_keeps_higher_risk_under_backpressure_and_expires():
    manager = QueueManager(limits={"ingest": 1, "direct": 1, "deep": 1, "review": 1})
    assert manager.enqueue("deep", "low", risk=0.1, now=0)
    assert manager.enqueue("deep", "high", risk=0.9, now=0)
    assert manager.dequeue("deep", now=1) == "high"
    manager.enqueue("review", "old", risk=1, ttl_seconds=1, now=0)
    assert manager.dequeue("review", now=2) is None
    assert manager.stats["review"].expired == 1


def test_relation_state_rejects_future_and_expires():
    state = IncrementalRelationState()
    state.apply(RelationEdge("a", "b", "flow", 1.0, 5, expires_at=10), prediction_time=5)
    assert state.edge_count == 1
    try:
        state.apply(RelationEdge("a", "c", "temporal", 1.0, 9), prediction_time=8)
        assert False, "future edge accepted"
    except ValueError:
        pass
    assert state.expire(10) == 1
    assert state.node_count == 0
