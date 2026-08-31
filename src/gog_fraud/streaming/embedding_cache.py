from __future__ import annotations

import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stale_rejections: int = 0


@dataclass
class _Entry:
    value: Any
    created_at: int
    size: int
    model_version: str
    feature_version: str


class EmbeddingCache:
    def __init__(self, *, max_entries: int, max_bytes: int, ttl_seconds: int) -> None:
        if min(max_entries, max_bytes, ttl_seconds) < 1:
            raise ValueError("cache limits must be positive")
        self.max_entries, self.max_bytes, self.ttl = max_entries, max_bytes, ttl_seconds
        self._items: OrderedDict[str, _Entry] = OrderedDict()
        self._bytes = 0
        self._last_expiry_scan = 0
        self.stats = CacheStats()

    def put(self, key: str, value: Any, *, now: int, model_version: str, feature_version: str) -> None:
        size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
        if size > self.max_bytes:
            self.stats.evictions += 1
            return
        old = self._items.pop(key, None)
        if old:
            self._bytes -= old.size
        self._items[key] = _Entry(value, now, size, model_version, feature_version)
        self._bytes += size
        self._evict(now)

    def get(self, key: str, *, now: int, model_version: str, feature_version: str, allow_stale: bool = False) -> Any | None:
        entry = self._items.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        stale = now - entry.created_at > self.ttl or entry.model_version != model_version or entry.feature_version != feature_version
        if stale and not allow_stale:
            self._remove(key)
            self.stats.misses += 1
            self.stats.stale_rejections += 1
            return None
        self._items.move_to_end(key)
        self.stats.hits += 1
        return entry.value

    def _remove(self, key: str) -> None:
        entry = self._items.pop(key)
        self._bytes -= entry.size

    def _evict(self, now: int) -> None:
        # A full TTL scan on every insertion makes a bounded cache O(events *
        # capacity). Gets still reject stale entries immediately; insertion
        # performs a periodic complete sweep, preserving the TTL invariant
        # while keeping long streaming replays amortized linear.
        scan_interval = min(self.ttl, 1000)
        if now - self._last_expiry_scan >= scan_interval:
            for key in list(self._items):
                if now - self._items[key].created_at > self.ttl:
                    self._remove(key)
                    self.stats.evictions += 1
            self._last_expiry_scan = now
        while len(self._items) > self.max_entries or self._bytes > self.max_bytes:
            _, entry = self._items.popitem(last=False)
            self._bytes -= entry.size
            self.stats.evictions += 1

    def metadata_snapshot(self) -> dict[str, Any]:
        return {"keys": list(self._items), "bytes": self._bytes, "stats": vars(self.stats).copy()}

    @property
    def current_bytes(self) -> int:
        return self._bytes
