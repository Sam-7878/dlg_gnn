from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


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

    def write(self, json_path: str | Path, csv_path: str | Path) -> None:
        payload = asdict(self)
        jp, cp = Path(json_path), Path(csv_path)
        jp.parent.mkdir(parents=True, exist_ok=True); cp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        flat = {**payload, "file_hashes": json.dumps(payload["file_hashes"], sort_keys=True)}
        with cp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat)); writer.writeheader(); writer.writerow(flat)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_manifest(source_root: str | Path, *, chain: str, labels_path: str | Path | None = None) -> DatasetManifest:
    root = Path(source_root).resolve()
    chain_root = root / chain if (root / chain).exists() else root
    files = sorted(path for path in chain_root.rglob("*") if path.is_file())
    csv_files = [path for path in files if path.suffix.lower() == ".csv"]
    transactions = missing = 0; timestamps: list[int] = []; blocks: list[int] = []; addresses: set[str] = set(); sample_ids: list[str] = []
    for path in csv_files:
        try:
            with path.open(encoding="utf-8-sig", errors="replace", newline="") as handle:
                for index, row in enumerate(csv.DictReader(handle)):
                    transactions += 1; sample_ids.append(str(row.get("sample_id") or f"{path}:{index}"))
                    lowered = {str(key).lower().replace("_", ""): value for key, value in row.items()}
                    raw_time = lowered.get("timestamp") or lowered.get("blocktimestamp")
                    if raw_time in (None, ""): missing += 1
                    else:
                        try: timestamps.append(int(float(raw_time)))
                        except ValueError: missing += 1
                    raw_block = lowered.get("blocknumber")
                    if raw_block not in (None, ""):
                        try: blocks.append(int(float(raw_block)))
                        except ValueError: pass
                    addresses.update(str(lowered.get(name)) for name in ("from", "to", "address") if lowered.get(name))
        except OSError:
            continue
    labels = Counter()
    if labels_path and Path(labels_path).is_file():
        with Path(labels_path).open(encoding="utf-8-sig", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                row_chain = str(row.get("Chain", row.get("chain", chain))).lower()
                if row_chain != chain.lower(): continue
                raw = row.get("Category", row.get("label", ""))
                if raw in (None, ""): labels["unlabeled"] += 1
                elif str(raw).strip() in {"0", "0.0", "benign", "normal"}: labels["benign"] += 1
                else: labels["fraud"] += 1
    labeled = labels["fraud"] + labels["benign"]
    selected_hashes = {str(path.relative_to(root)): _hash(path) for path in files if path.suffix.lower() in {".csv", ".json", ".pt", ".pkl"}}
    return DatasetManifest(chain, str(root), min(timestamps, default=None), max(timestamps, default=None), min(blocks, default=None), max(blocks, default=None), transactions, len(csv_files), len(addresses), len(files), labels["fraud"], labels["benign"], labels["unlabeled"], labels["fraud"] / labeled if labeled else None, len(sample_ids) - len(set(sample_ids)), missing, selected_hashes)
