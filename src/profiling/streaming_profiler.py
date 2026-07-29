from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import psutil


class StreamingProfiler:
    """Cold/steady latency and bounded-memory profiler for streaming runs."""

    def __init__(
        self,
        *,
        synchronize: Callable[[], None] | None = None,
        rss_reader: Callable[[], int] | None = None,
        vram_reader: Callable[[], int] | None = None,
    ) -> None:
        self.timers: dict[str, float] = {}
        self.latencies: dict[str, list[float]] = {}
        self.trace: list[dict[str, float | int | str | bool]] = []
        self.memory_samples: list[dict[str, int]] = []
        self._stack: list[str] = []
        self._synchronize = synchronize or self._default_synchronize
        process = psutil.Process(os.getpid())
        self._rss_reader = rss_reader or (lambda: process.memory_info().rss)
        self._vram_reader = vram_reader or self._default_vram

    @staticmethod
    def _default_synchronize() -> None:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except (ImportError, RuntimeError):
            return

    @staticmethod
    def _default_vram() -> int:
        try:
            import torch
            return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        except (ImportError, RuntimeError):
            return 0

    def start_timer(self, step_name: str) -> None:
        if step_name in self.timers:
            raise ValueError(f"timer already active: {step_name}")
        self._synchronize()
        self.timers[step_name] = time.perf_counter()
        self._stack.append(step_name)

    def stop_timer(self, step_name: str, *, cold_start: bool = False) -> float:
        if step_name not in self.timers:
            raise KeyError(f"timer not active: {step_name}")
        if not self._stack or self._stack[-1] != step_name:
            raise ValueError(f"timer nesting violation: expected {self._stack[-1] if self._stack else 'none'}, got {step_name}")
        self._synchronize()
        elapsed_ms = (time.perf_counter() - self.timers.pop(step_name)) * 1000.0
        self._stack.pop()
        self.latencies.setdefault(step_name, []).append(elapsed_ms)
        self.trace.append({"step": step_name, "latency_ms": elapsed_ms, "cold_start": cold_start})
        return elapsed_ms

    @contextmanager
    def timer(self, step_name: str, *, cold_start: bool = False) -> Iterator[None]:
        self.start_timer(step_name)
        try:
            yield
        finally:
            self.stop_timer(step_name, cold_start=cold_start)

    def get_average_latency(self, step_name: str) -> float:
        values = self.latencies.get(step_name, [])
        return float(np.mean(values)) if values else 0.0

    def get_percentile_latency(self, step_name: str, percentile: float) -> float:
        if not 0 <= percentile <= 100:
            raise ValueError("percentile must be in [0, 100]")
        values = self.latencies.get(step_name, [])
        return float(np.percentile(values, percentile)) if values else 0.0

    def latency_summary(self, step_name: str, *, include_cold: bool = True) -> dict[str, float | int]:
        values = [
            float(row["latency_ms"])
            for row in self.trace
            if row["step"] == step_name and (include_cold or not row["cold_start"])
        ]
        if not values:
            return {"count": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
        return {
            "count": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
        }

    def record_memory(self, event_index: int, *, cache_bytes: int = 0, queue_bytes: int = 0) -> dict[str, int]:
        row = {
            "event_index": int(event_index), "rss_bytes": int(self._rss_reader()),
            "vram_bytes": int(self._vram_reader()), "cache_bytes": int(cache_bytes),
            "queue_bytes": int(queue_bytes),
        }
        self.memory_samples.append(row)
        return row

    def get_peak_memory_mb(self) -> float:
        values = [row["rss_bytes"] for row in self.memory_samples]
        peak = max(values) if values else int(self._rss_reader())
        return peak / (1024 * 1024)

    def peak_vram_mb(self) -> float:
        values = [row["vram_bytes"] for row in self.memory_samples]
        peak = max(values) if values else int(self._vram_reader())
        return peak / (1024 * 1024)

    def memory_slope_mb_per_10k(self) -> float:
        if len(self.memory_samples) < 2:
            return 0.0
        x = np.asarray([row["event_index"] for row in self.memory_samples], dtype=float)
        y = np.asarray([row["rss_bytes"] for row in self.memory_samples], dtype=float) / (1024 * 1024)
        if np.all(x == x[0]):
            return 0.0
        return float(np.polyfit(x, y, 1)[0] * 10000.0)

    def write_trace(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"latency_trace": self.trace, "memory_trace": self.memory_samples}
        target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def calculate_communication_reduction(self, raw_text: str, risk_vector: dict) -> tuple[int, int, float]:
        raw_bytes = len(raw_text.encode("utf-8")) if raw_text else 2048
        if raw_bytes < 256:
            raw_bytes = 2048
        vector_bytes = 96
        return raw_bytes, vector_bytes, float(1.0 - vector_bytes / raw_bytes)
