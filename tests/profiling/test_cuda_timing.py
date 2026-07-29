from profiling.streaming_profiler import StreamingProfiler


def test_synchronizer_wraps_timed_region():
    calls = []
    profiler = StreamingProfiler(synchronize=lambda: calls.append("sync"))
    profiler.start_timer("inference")
    profiler.stop_timer("inference")
    assert calls == ["sync", "sync"]


def test_cpu_fallback_is_safe():
    profiler = StreamingProfiler()
    with profiler.timer("cpu"):
        sum(range(10))
    assert profiler.get_average_latency("cpu") >= 0
