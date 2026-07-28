from __future__ import annotations

import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

from gog_fraud.data.io.streaming_dataset import StreamEvent


@dataclass(frozen=True)
class SubgraphDelta:
    contract_id: str
    nodes_added: int
    edges_added: int
    duplicate: bool


@dataclass(frozen=True)
class ExpirationStats:
    contracts: int = 0
    edges: int = 0


class IncrementalSubgraphStore:
    def __init__(self, *, temporal_window_seconds: int, max_nodes_per_contract: int, max_edges_per_contract: int, contract_ttl_seconds: int | None = None) -> None:
        if min(temporal_window_seconds, max_nodes_per_contract, max_edges_per_contract) < 1:
            raise ValueError("subgraph limits must be positive")
        self.window = temporal_window_seconds
        self.max_nodes = max_nodes_per_contract
        self.max_edges = max_edges_per_contract
        self.contract_ttl = contract_ttl_seconds or temporal_window_seconds
        self._events: dict[str, OrderedDict[str, tuple[int, str, str, Mapping[str, Any]]]] = defaultdict(OrderedDict)
        self._last_seen: dict[str, int] = {}

    def apply_event(self, event: StreamEvent) -> SubgraphDelta:
        payload = event.payload if isinstance(event.payload, Mapping) else {}
        src, dst = str(payload.get("src", payload.get("from", ""))), str(payload.get("dst", payload.get("to", "")))
        key = str(payload.get("edge_id", event.sample_id))
        bucket = self._events[event.contract_id]
        duplicate = key in bucket
        bucket[key] = (event.event_time, src, dst, payload)
        bucket.move_to_end(key)
        self._last_seen[event.contract_id] = max(event.event_time, self._last_seen.get(event.contract_id, event.event_time))
        self._trim(event.contract_id, event.event_time)
        return SubgraphDelta(event.contract_id, len({src, dst} - {""}) if not duplicate else 0, 0 if duplicate else 1, duplicate)

    def _trim(self, contract_id: str, watermark: int) -> None:
        bucket = self._events[contract_id]
        cutoff = watermark - self.window
        for key in list(bucket):
            if bucket[key][0] < cutoff:
                del bucket[key]
        while len(bucket) > self.max_edges:
            bucket.popitem(last=False)
        while len({node for _, src, dst, _ in bucket.values() for node in (src, dst) if node}) > self.max_nodes:
            bucket.popitem(last=False)

    def materialize(self, contract_id: str, event_time: int) -> dict[str, Any]:
        self._trim(contract_id, event_time)
        values = sorted(self._events.get(contract_id, {}).values(), key=lambda item: (item[0], str(item[3])))
        nodes = sorted({node for _, src, dst, _ in values for node in (src, dst) if node})
        return {"contract_id": contract_id, "event_time": event_time, "nodes": nodes, "edges": [(src, dst, ts) for ts, src, dst, _ in values]}

    def expire(self, watermark: int) -> ExpirationStats:
        removed_contracts = removed_edges = 0
        for contract_id in list(self._events):
            before = len(self._events[contract_id])
            self._trim(contract_id, watermark)
            removed_edges += before - len(self._events[contract_id])
            if self._last_seen[contract_id] < watermark - self.contract_ttl:
                removed_edges += len(self._events[contract_id])
                del self._events[contract_id]
                del self._last_seen[contract_id]
                removed_contracts += 1
        return ExpirationStats(removed_contracts, removed_edges)

    def snapshot(self) -> dict[str, Any]:
        return {"events": {key: list(value.items()) for key, value in self._events.items()}, "last_seen": dict(self._last_seen)}

    def restore(self, state: Mapping[str, Any]) -> None:
        self._events = defaultdict(OrderedDict, {key: OrderedDict(value) for key, value in state["events"].items()})
        self._last_seen = {key: int(value) for key, value in state["last_seen"].items()}

    @property
    def estimated_bytes(self) -> int:
        return sum(sys.getsizeof(key) + sys.getsizeof(value) for bucket in self._events.values() for key, value in bucket.items())
