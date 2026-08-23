#!/usr/bin/env python3
"""
Scan timestamps, topics, and record counts across all extracted THEIA files.
"""
import gzip
import time
from datetime import datetime, timezone
from pathlib import Path
import fastavro

THEIA_DIR = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Data/theia/theia-20260822T022150Z-1-001/theia")

files = sorted(THEIA_DIR.glob("*.bin.*.gz"))
print(f"Scanning {len(files)} files in {THEIA_DIR.name} ...\n")

for fpath in files:
    t0 = time.time()
    min_ts = None
    max_ts = None
    total = 0
    events = 0
    first_few = []

    with gzip.open(fpath, "rb") as f:
        reader = fastavro.reader(f)
        for i, rec in enumerate(reader):
            total += 1
            datum = rec.get("datum", {})
            ts = datum.get("timestampNanos")
            if ts:
                events += 1
                if min_ts is None or ts < min_ts:
                    min_ts = ts
                if max_ts is None or ts > max_ts:
                    max_ts = ts
            if i % 2_000_000 == 0 and i > 0:
                print(f"    [{fpath.name}] {i:,} records ...")

    elapsed = time.time() - t0
    min_dt = datetime.fromtimestamp(min_ts / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if min_ts else "N/A"
    max_dt = datetime.fromtimestamp(max_ts / 1e9, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC') if max_ts else "N/A"
    print(f"File: {fpath.name}")
    print(f"  Size: {fpath.stat().st_size / 1024**2:.1f} MB | Total records: {total:,} | Events: {events:,}")
    print(f"  Start: {min_dt} ({min_ts})")
    print(f"  End:   {max_dt} ({max_ts})")
    print(f"  Scan time: {elapsed:.2f}s\n")
