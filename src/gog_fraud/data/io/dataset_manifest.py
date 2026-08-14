from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class DatasetManifest:
    chain: str
    source_root: str
    collection_start: int | None
    collection_end: int | None
    block_min: int | None
    block_max: int | None
    transactions: int
    contracts: int
    addresses: int
    subgraphs: int
    fraud: int
    benign: int
    unlabeled: int
    positive_ratio: float | None
    duplicates: int
    missing_timestamp: int
    file_hashes: dict[str, str]
    canonical_root: str | None = None
    physical_root: str | None = None
    preprocessing_source: str | None = None
    source_version: str | None = None
    chain_directory: str | None = None
    manifest_complete: bool = False
    truncated: bool = False
    max_files_requested: int | None = None
    files_discovered: int = 0
    files_processed: int = 0
    files_failed: int = 0
    hash_algorithm: str = "sha256"
    hash_verification: str = "full"
    hash_cache_validation: str = "absolute_path+size+mtime_ns+inode"
    label_mapping: dict[str, str] | None = None
    label_categories: dict[str, int] | None = None

    def write(self, json_path: str | Path, csv_path: str | Path) -> None:
        payload = asdict(self)
        jp, cp = Path(json_path), Path(csv_path)
        jp.parent.mkdir(parents=True, exist_ok=True)
        cp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        flat = {
            **payload,
            "file_hashes": json.dumps(payload["file_hashes"], sort_keys=True),
        }
        with cp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat))
            writer.writeheader()
            writer.writerow(flat)


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------

def _hash_file(path: Path) -> str:
    """SHA-256 hash of a single file (streaming, 1 MB chunks)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hash_index(index_path: Path) -> Dict[str, Any]:
    """Load a previously saved metadata-guarded hash index."""
    if index_path.exists():
        try:
            return json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_hash_index(index_path: Path, index: Dict[str, Any]) -> None:
    """Persist the hash index atomically."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = index_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(index, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    tmp.replace(index_path)


# ---------------------------------------------------------------------------
# Progress reporter
# ---------------------------------------------------------------------------

class _ProgressReporter:
    """Simple stderr progress reporter (no external dependencies)."""

    def __init__(self, total: Optional[int], label: str = "", enabled: bool = True):
        self._total = total
        self._label = label
        self._enabled = enabled
        self._done = 0
        self._start = time.monotonic()
        self._last_print = 0.0

    def update(self, n: int = 1) -> None:
        if not self._enabled:
            return
        self._done += n
        now = time.monotonic()
        if now - self._last_print >= 2.0:  # print at most every 2 s
            self._print()
            self._last_print = now

    def close(self) -> None:
        if self._enabled:
            self._print(final=True)
            print("", file=sys.stderr)

    def _print(self, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start
        rate = self._done / elapsed if elapsed > 0 else 0.0
        if self._total:
            pct = 100.0 * self._done / self._total
            eta = (self._total - self._done) / rate if rate > 0 else float("inf")
            msg = (
                f"\r[{self._label}] {self._done}/{self._total} ({pct:.1f}%) "
                f"{rate:.0f} files/s  ETA {eta:.0f}s"
            )
        else:
            msg = f"\r[{self._label}] {self._done} files  {rate:.0f} files/s  {elapsed:.0f}s elapsed"
        print(msg, end="" if not final else "\n", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_dataset_manifest(
    source_root: str | Path,
    *,
    chain: str,
    labels_path: str | Path | None = None,
    max_files: Optional[int] = None,
    progress: bool = False,
    hash_index_path: Optional[str | Path] = None,
    canonical_root: Optional[str | Path] = None,
    preprocessing_source: Optional[str | Path] = None,
    source_version: Optional[str] = None,
    on_file: Optional[Callable[[Path, int, int], None]] = None,
) -> DatasetManifest:
    """Build a DatasetManifest by scanning *source_root*.

    Args:
        source_root: Root directory containing chain sub-directories or files.
        chain: Chain name (e.g., "ethereum", "bsc", "polygon").
        labels_path: Path to the labels CSV (optional).
        max_files: If set, stop scanning after this many files (useful for
            smoke tests and early-exit diagnostics).
        progress: If True, print chain/file progress to stderr every 2 s.
        hash_index_path: Path to a JSON file used as a resumable SHA-256
        index. Cached hashes are reused only when absolute path, size,
            nanosecond mtime, and (when available) inode all match.
        on_file: Optional callback(path, file_index, total_files).
    """
    root = Path(source_root).resolve()
    chain_root = root / chain if (root / chain).exists() else root

    # ── 1. Enumerate files ──────────────────────────────────────────────────
    discovered_files = sorted(p for p in chain_root.rglob("*") if p.is_file())
    files_discovered = len(discovered_files)
    truncated = max_files is not None and max_files < files_discovered
    all_files = discovered_files[:max_files] if max_files is not None else discovered_files

    total_files = len(all_files)
    csv_files = [p for p in all_files if p.suffix.lower() == ".csv"]

    prog = _ProgressReporter(total=total_files, label=f"manifest/{chain}", enabled=progress)

    # ── 2. Load resumable hash index ────────────────────────────────────────
    index_path = Path(hash_index_path) if hash_index_path else None
    hash_index: Dict[str, Any] = _load_hash_index(index_path) if index_path else {}
    failed_files: set[str] = set()

    # ── 3. Scan CSV files ───────────────────────────────────────────────────
    transactions = 0
    missing = 0
    timestamp_min: int | None = None
    timestamp_max: int | None = None
    block_minimum: int | None = None
    block_maximum: int | None = None
    addresses: set[str] = set()
    seen_sample_ids: set[str] = set()
    duplicates = 0

    for file_idx, path in enumerate(all_files):
        prog.update(1)
        if on_file is not None:
            on_file(path, file_idx, total_files)

        if path.suffix.lower() != ".csv":
            continue

        try:
            wanted = {"sampleid", "timestamp", "blocktimestamp", "blocknumber", "from", "to", "address"}
            chunks = pd.read_csv(
                path, dtype=str, chunksize=100_000, encoding="utf-8-sig", encoding_errors="replace",
                usecols=lambda name: str(name).lower().replace("_", "") in wanted,
            )
            for frame in chunks:
                transactions += len(frame)
                columns = {str(name).lower().replace("_", ""): name for name in frame.columns}
                time_column = columns.get("timestamp") or columns.get("blocktimestamp")
                if time_column is None:
                    missing += len(frame)
                else:
                    values = pd.to_numeric(frame[time_column], errors="coerce")
                    missing += int(values.isna().sum())
                    valid = values.dropna()
                    if not valid.empty:
                        chunk_min, chunk_max = int(valid.min()), int(valid.max())
                        timestamp_min = chunk_min if timestamp_min is None else min(timestamp_min, chunk_min)
                        timestamp_max = chunk_max if timestamp_max is None else max(timestamp_max, chunk_max)
                block_column = columns.get("blocknumber")
                if block_column is not None:
                    valid_blocks = pd.to_numeric(frame[block_column], errors="coerce").dropna()
                    if not valid_blocks.empty:
                        chunk_min, chunk_max = int(valid_blocks.min()), int(valid_blocks.max())
                        block_minimum = chunk_min if block_minimum is None else min(block_minimum, chunk_min)
                        block_maximum = chunk_max if block_maximum is None else max(block_maximum, chunk_max)
                for normalized in ("from", "to", "address"):
                    column = columns.get(normalized)
                    if column is not None:
                        addresses.update(frame[column].dropna().astype(str).unique().tolist())
                sample_column = columns.get("sampleid")
                if sample_column is not None:
                    for sample_id in frame[sample_column].dropna().astype(str):
                        if sample_id in seen_sample_ids:
                            duplicates += 1
                        else:
                            seen_sample_ids.add(sample_id)
        except (OSError, ValueError, pd.errors.ParserError):
            failed_files.add(str(path.resolve()))
            continue

    prog.close()

    # ── 4. Label statistics ─────────────────────────────────────────────────
    labels: Counter = Counter()
    label_categories: Counter = Counter()
    if labels_path and Path(labels_path).is_file():
        with Path(labels_path).open(
            encoding="utf-8-sig", errors="replace", newline=""
        ) as handle:
            for row in csv.DictReader(handle):
                row_chain = str(
                    row.get("Chain", row.get("chain", chain))
                ).lower()
                if row_chain != chain.lower():
                    continue
                raw = row.get("Category", row.get("label", ""))
                if raw in (None, ""):
                    labels["unlabeled"] += 1
                    label_categories["UNLABELED"] += 1
                elif str(raw).strip().lower() in {"0", "0.0", "benign", "normal"}:
                    labels["benign"] += 1
                    label_categories[str(raw).strip()] += 1
                else:
                    labels["fraud"] += 1
                    label_categories[str(raw).strip()] += 1

    labeled = labels["fraud"] + labels["benign"]

    # ── 5. File hashes (resumable) ──────────────────────────────────────────
    # A manifest advertised as "full" must cover every discovered file,
    # including auxiliary source metadata such as list.txt.  Restricting this
    # to known data extensions silently left such files outside provenance.
    hash_candidates = list(all_files)
    if progress and hash_candidates:
        print(
            f"\r[manifest/{chain}] hashing {len(hash_candidates)} files ...",
            file=sys.stderr,
            flush=True,
        )

    selected_hashes: Dict[str, str] = {}
    hash_prog = _ProgressReporter(
        total=len(hash_candidates),
        label=f"hash/{chain}",
        enabled=progress and len(hash_candidates) > 50,
    )
    for path in hash_candidates:
        rel = str(path.relative_to(root))
        absolute = str(path.resolve())
        try:
            stat = path.stat()
            cached = hash_index.get(absolute)
            cache_hit = (
                isinstance(cached, dict)
                and cached.get("path") == absolute
                and cached.get("size") == stat.st_size
                and cached.get("mtime_ns") == stat.st_mtime_ns
                and (cached.get("inode") in (None, stat.st_ino))
                and isinstance(cached.get("sha256"), str)
            )
            if cache_hit:
                selected_hashes[rel] = cached["sha256"]
            else:
                digest = _hash_file(path)
                selected_hashes[rel] = digest
                hash_index[absolute] = {
                    "path": absolute,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "inode": stat.st_ino or None,
                    "sha256": digest,
                }
        except OSError:
            failed_files.add(absolute)
        hash_prog.update(1)
    hash_prog.close()

    # Persist updated hash index
    if index_path is not None:
        _save_hash_index(index_path, hash_index)

    # ── 6. Assemble manifest ────────────────────────────────────────────────
    return DatasetManifest(
        chain=chain,
        source_root=str(root),
        collection_start=timestamp_min,
        collection_end=timestamp_max,
        block_min=block_minimum,
        block_max=block_maximum,
        transactions=transactions,
        contracts=len(csv_files),
        addresses=len(addresses),
        subgraphs=len(all_files),
        fraud=labels["fraud"],
        benign=labels["benign"],
        unlabeled=labels["unlabeled"],
        positive_ratio=labels["fraud"] / labeled if labeled else None,
        duplicates=duplicates,
        missing_timestamp=missing,
        file_hashes=selected_hashes,
        canonical_root=str(Path(canonical_root).resolve()) if canonical_root else str(root),
        physical_root=str(root),
        preprocessing_source=str(Path(preprocessing_source).resolve()) if preprocessing_source else None,
        source_version=source_version,
        chain_directory=str(chain_root),
        manifest_complete=not truncated and not failed_files and len(selected_hashes) == len(hash_candidates),
        truncated=truncated,
        max_files_requested=max_files,
        files_discovered=files_discovered,
        files_processed=len(all_files),
        files_failed=len(failed_files),
        hash_verification="full" if len(selected_hashes) == len(hash_candidates) and not failed_files else "incomplete",
        label_mapping={"0": "benign", "nonzero": "fraud", "missing": "unlabeled"},
        label_categories=dict(sorted(label_categories.items())),
    )
