from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(order=True)
class _Queued:
    priority: float
    sequence: int
    enqueued_at: float = field(compare=False)
    item: Any = field(compare=False)
    expires_at: float | None = field(compare=False, default=None)


@dataclass
class QueueStats:
    enqueued: int = 0
    dequeued: int = 0
    dropped: int = 0
    expired: int = 0
    backpressure_activations: int = 0
    wait_ms: list[float] = field(default_factory=list)


class QueueManager:
    NAMES = ("ingest", "direct", "deep", "review")

    def __init__(self, *, limits: dict[str, int], overload_policy: str = "risk_priority") -> None:
        if overload_policy not in {"risk_priority", "drop_newest"}:
            raise ValueError("unsupported overload policy")
        self.limits = {name: int(limits.get(name, 0)) for name in self.NAMES}
        if any(limit < 1 for limit in self.limits.values()):
            raise ValueError("all queue limits must be positive")
        self.overload_policy = overload_policy
        self._queues: dict[str, list[_Queued]] = {name: [] for name in self.NAMES}
        self._counter = itertools.count()
        self.stats: dict[str, QueueStats] = {name: QueueStats() for name in self.NAMES}

    def enqueue(self, name: str, item: Any, *, risk: float = 0.0, ttl_seconds: float | None = None, now: float | None = None) -> bool:
        self._check_name(name)
        clock = time.monotonic() if now is None else now
        queue, stats = self._queues[name], self.stats[name]
        entry = _Queued(-float(risk), next(self._counter), clock, item, None if ttl_seconds is None else clock + ttl_seconds)
        if len(queue) >= self.limits[name]:
            stats.backpressure_activations += 1
            if self.overload_policy == "drop_newest":
                stats.dropped += 1
                return False
            worst = max(queue)
            if entry >= worst:
                stats.dropped += 1
                return False
            queue.remove(worst); heapq.heapify(queue); stats.dropped += 1
        heapq.heappush(queue, entry); stats.enqueued += 1
        return True

    def dequeue(self, name: str, *, now: float | None = None) -> Any | None:
        self._check_name(name)
        clock = time.monotonic() if now is None else now
        queue, stats = self._queues[name], self.stats[name]
        while queue:
            entry = heapq.heappop(queue)
            if entry.expires_at is not None and entry.expires_at <= clock:
                stats.expired += 1
                continue
            stats.dequeued += 1
            stats.wait_ms.append(max(0.0, (clock - entry.enqueued_at) * 1000.0))
            return entry.item
        return None

    def depth(self, name: str) -> int:
        self._check_name(name); return len(self._queues[name])

    def snapshot(self) -> dict[str, Any]:
        return {name: [vars(entry).copy() for entry in queue] for name, queue in self._queues.items()}

    def restore(self, state: dict[str, Any]) -> None:
        for name in self.NAMES:
            self._queues[name] = [_Queued(**entry) for entry in state.get(name, [])]
            heapq.heapify(self._queues[name])
        max_sequence = max((entry.sequence for queue in self._queues.values() for entry in queue), default=-1)
        self._counter = itertools.count(max_sequence + 1)

    def _check_name(self, name: str) -> None:
        if name not in self._queues:
            raise KeyError(f"unknown queue: {name}")
