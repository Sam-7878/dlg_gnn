"""Generate pre-event, label-independent context for the Round 4 auxiliary track."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


GENERATOR_VERSION = "round4-context-v1.0"
SOURCE_FIELDS = ("chain_id", "timestamp", "num_nodes", "num_edges")


def generate(input_path: Path, output_path: Path) -> pd.DataFrame:
    source = pd.read_parquet(input_path, columns=["event_id", *SOURCE_FIELDS])
    rows = []
    for event in source.itertuples(index=False):
        dt = datetime.fromtimestamp(int(event.timestamp), tz=timezone.utc)
        density_band = "high" if event.num_edges >= 512 else ("medium" if event.num_edges >= 64 else "low")
        context = (
            f"chain={event.chain_id}; observed_time_utc={dt.isoformat()}; "
            f"historical_nodes={int(event.num_nodes)}; historical_edges={int(event.num_edges)}; "
            f"activity_band={density_band}"
        )
        rows.append({
            "event_id": event.event_id,
            "context_text": context,
            "source_fields": ",".join(SOURCE_FIELDS),
            "source_timestamp": int(event.timestamp),
            "generator_version": GENERATOR_VERSION,
            "label_accessed": False,
        })
    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/benchmark/gog_scimain_v1/transactions.parquet"))
    parser.add_argument("--output", type=Path, default=Path("results/graphrag/round_4/context_provenance.parquet"))
    args = parser.parse_args()
    result = generate(args.input, args.output)
    print(f"wrote {len(result)} context provenance rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
