from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True, order=True)
class RelationEdge:
    src: str
    dst: str
    relation_type: str
    weight: float
    created_at: int
    expires_at: int | None = None
    cross_chain: bool = False
    source_max_time: int | None = None


class IncrementalRelationState:
    def __init__(self) -> None:
        self._edges: dict[tuple[str, str, str], RelationEdge] = {}
        self._nodes: set[str] = set()

    def apply(self, edge: RelationEdge, *, prediction_time: int) -> None:
        source_time = edge.source_max_time if edge.source_max_time is not None else edge.created_at
        if source_time > prediction_time or edge.created_at > prediction_time:
            raise ValueError("future relation edge is forbidden")
        key = (edge.src, edge.dst, edge.relation_type)
        self._edges[key] = edge
        self._nodes.update((edge.src, edge.dst))

    def expire(self, watermark: int) -> int:
        expired = [key for key, edge in self._edges.items() if edge.expires_at is not None and edge.expires_at <= watermark]
        for key in expired:
            del self._edges[key]
        connected = {node for edge in self._edges.values() for node in (edge.src, edge.dst)}
        self._nodes.intersection_update(connected)
        return len(expired)

    def edges(self) -> tuple[RelationEdge, ...]:
        return tuple(sorted(self._edges.values()))

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def snapshot(self) -> list[dict]:
        return [asdict(edge) for edge in self.edges()]

    def restore(self, state: Iterable[Mapping]) -> None:
        self._edges.clear(); self._nodes.clear()
        for item in state:
            edge = RelationEdge(**item)
            self.apply(edge, prediction_time=max(edge.created_at, edge.source_max_time or edge.created_at))
