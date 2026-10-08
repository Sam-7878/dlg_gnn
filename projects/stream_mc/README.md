# StreamMC facade

`./projects/stream_mc/reproduce.sh verify` checks the current streaming source hashes and test presence without data or credentials. With pandas installed it also checks fixed event ingestion, checkpoint/replay, bounded subgraph state and a frozen result hash; otherwise this scientific smoke is reported as skipped. In the scientific environment run `PYTHONPATH=src pytest tests/stream_mc/streaming/test_stateful_stream.py tests/stream_mc/streaming/test_bounded_state.py` for broader replay tests. This A07 facade is a scaffold; StreamMC paper results and full load tests are not revalidated here.

The active manuscript and earlier local draft are identified in [paper/README.md](paper/README.md).
