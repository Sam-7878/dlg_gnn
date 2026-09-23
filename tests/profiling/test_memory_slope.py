import pytest

from profiling.streaming_profiler import StreamingProfiler


def test_memory_slope_peak_and_component_bytes():
    rss = iter([100 * 1024 * 1024, 110 * 1024 * 1024, 120 * 1024 * 1024])
    profiler = StreamingProfiler(
        synchronize=lambda: None,
        rss_reader=lambda: next(rss),
        vram_reader=lambda: 25 * 1024 * 1024,
    )
    profiler.record_memory(0, cache_bytes=10, queue_bytes=20)
    profiler.record_memory(10000, cache_bytes=30, queue_bytes=40)
    profiler.record_memory(20000, cache_bytes=50, queue_bytes=60)
    assert profiler.get_peak_memory_mb() == 120
    assert profiler.peak_vram_mb() == 25
    assert profiler.memory_slope_mb_per_10k() == pytest.approx(10.0)
    assert profiler.memory_samples[-1]["cache_bytes"] == 50
