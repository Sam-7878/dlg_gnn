import json

import pytest

from profiling.streaming_profiler import StreamingProfiler


def test_nested_timers_percentiles_and_trace(tmp_path):
    profiler = StreamingProfiler(synchronize=lambda: None, rss_reader=lambda: 100, vram_reader=lambda: 0)
    with profiler.timer("outer", cold_start=True):
        with profiler.timer("inner"):
            pass
    assert profiler.latency_summary("outer")["count"] == 1
    assert profiler.latency_summary("outer", include_cold=False)["count"] == 0
    assert profiler.get_percentile_latency("inner", 95) >= 0
    target = tmp_path / "trace.json"
    profiler.write_trace(target)
    assert len(json.loads(target.read_text())["latency_trace"]) == 2


def test_timer_nesting_violation_is_rejected():
    profiler = StreamingProfiler(synchronize=lambda: None)
    profiler.start_timer("outer")
    profiler.start_timer("inner")
    with pytest.raises(ValueError, match="nesting"):
        profiler.stop_timer("outer")
