from __future__ import annotations

import hashlib
import json
import pandas as pd
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from .dataset import FraudDataset

log = logging.getLogger(__name__)


@dataclass(frozen=True, order=True)
class StreamEvent:
    event_time: int
    block_number: int
    transaction_index: int
    sample_id: str
    chain_id: str = field(compare=False)
    contract_id: str = field(compare=False)
    payload: Any = field(compare=False, default=None)

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "StreamEvent":
        required = ("sample_id", "chain_id", "contract_id", "event_time")
        missing = [key for key in required if record.get(key) in (None, "")]
        if missing:
            raise ValueError(f"missing stream fields: {', '.join(missing)}")
        return cls(
            sample_id=str(record["sample_id"]),
            chain_id=str(record["chain_id"]),
            contract_id=str(record["contract_id"]),
            event_time=int(record["event_time"]),
            block_number=int(record.get("block_number", 0)),
            transaction_index=int(record.get("transaction_index", 0)),
            payload=record.get("payload", record),
        )


@dataclass(frozen=True)
class StreamCheckpoint:
    offset: int
    watermark: int | None
    order_hash: str


class StatefulTransactionStream:
    """Deterministic, checkpointable replay over normalized stream events."""

    def __init__(
        self,
        records: Iterable[StreamEvent | Mapping[str, Any]],
        *,
        warmup_until: int | None = None,
        prefetch_size: int = 256,
        delay_probability: float = 0.0,
        max_delay_positions: int = 0,
        seed: int = 42,
        replay_rate: float | None = None,
    ) -> None:
        if prefetch_size < 1:
            raise ValueError("prefetch_size must be positive")
        if not 0.0 <= delay_probability <= 1.0:
            raise ValueError("delay_probability must be within [0, 1]")
        self.quarantine: list[dict[str, Any]] = []
        events: list[StreamEvent] = []
        for index, record in enumerate(records):
            try:
                events.append(record if isinstance(record, StreamEvent) else StreamEvent.from_record(record))
            except (TypeError, ValueError, KeyError) as exc:
                self.quarantine.append({"record_index": index, "error": str(exc)})
        events.sort()
        if delay_probability and max_delay_positions:
            rng = random.Random(seed)
            for index in range(len(events) - 1):
                if rng.random() < delay_probability:
                    target = min(len(events) - 1, index + rng.randint(1, max_delay_positions))
                    events[index], events[target] = events[target], events[index]
        self._events = tuple(events)
        self._offset = 0
        self._watermark: int | None = None
        self.warmup_until = warmup_until
        self.prefetch_size = prefetch_size
        self.replay_rate = replay_rate
        self._order_hash = hashlib.sha256(
            "\n".join(event.sample_id for event in self._events).encode()
        ).hexdigest()
        self.last_event_lag_ms = 0.0

    def __iter__(self) -> Iterator[StreamEvent]:
        while self._offset < len(self._events):
            event = self._events[self._offset]
            start = time.perf_counter()
            self._offset += 1
            self._watermark = event.event_time if self._watermark is None else max(self._watermark, event.event_time)
            yield event
            self.last_event_lag_ms = max(0.0, (time.perf_counter() - start) * 1000.0)
            if self.replay_rate and self.replay_rate > 0:
                time.sleep(max(0.0, 1.0 / self.replay_rate - (time.perf_counter() - start)))

    @property
    def warmup_complete(self) -> bool:
        return self.warmup_until is None or (self._watermark is not None and self._watermark >= self.warmup_until)

    def checkpoint(self) -> StreamCheckpoint:
        return StreamCheckpoint(self._offset, self._watermark, self._order_hash)

    def restore(self, checkpoint: StreamCheckpoint) -> None:
        if checkpoint.order_hash != self._order_hash:
            raise ValueError("checkpoint belongs to a different event sequence")
        if not 0 <= checkpoint.offset <= len(self._events):
            raise ValueError("checkpoint offset is outside the stream")
        self._offset = checkpoint.offset
        self._watermark = checkpoint.watermark

    def remaining(self) -> int:
        return len(self._events) - self._offset

    @staticmethod
    def merged(streams: Iterable["StatefulTransactionStream"], **kwargs: Any) -> "StatefulTransactionStream":
        return StatefulTransactionStream((event for stream in streams for event in stream._events), **kwargs)

class StreamingDataset(FraudDataset):
    """
    Extends FraudDataset to enforce chronological splitting instead of random stratified splits.
    Scans the earliest transaction timestamp for each contract.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chronological_order = []
        
    def prepare_streaming_splits(self, transactions_root: str, train_ratio: float = 0.8):
        """
        Calculates the 80/20 chronological split based on the earliest transaction timestamp.
        """
        log.info("[StreamingDataset] Scanning transaction timestamps for sorting ... This may take a moment.")
        
        tx_dir = Path(transactions_root) / self.cfg.chain if self.cfg.chain else Path(transactions_root)
        
        contract_timestamps = {}
        missing = 0
        
        # For each contract in labels, find its earliest timestamp
        for cid in self.labels.keys():
            # Support both flat style (0xabc.csv) and folder style (0xabc/*.csv)
            flat_file = tx_dir / f"{cid}.csv"
            contract_dir = tx_dir / str(cid)
            
            csv_files = []
            if flat_file.exists():
                csv_files = [flat_file]
            elif contract_dir.exists() and contract_dir.is_dir():
                csv_files = list(contract_dir.glob("*.csv"))
            
            if not csv_files:
                missing += 1
                continue
            
            # Read just the first row of all CSVs associated with the contract
            min_ts = float('inf')
            for csv_f in csv_files:
                try:
                    df = pd.read_csv(csv_f, nrows=1)
                    # Normalize columns to lowercase for check
                    cols_lower = [c.lower() for c in df.columns]
                    
                    if 'timestamp' in cols_lower:
                        idx = cols_lower.index('timestamp')
                        ts = int(df.iloc[0, idx])
                        min_ts = min(min_ts, ts)
                    elif 'block_number' in cols_lower:
                        idx = cols_lower.index('block_number')
                        blk = int(df.iloc[0, idx])
                        min_ts = min(min_ts, blk)
                    elif 'blocknumber' in cols_lower:
                        idx = cols_lower.index('blocknumber')
                        blk = int(df.iloc[0, idx])
                        min_ts = min(min_ts, blk)
                except Exception:
                    continue
            
            if min_ts != float('inf'):
                contract_timestamps[cid] = min_ts
            else:
                missing += 1
                
        log.info(f"[StreamingDataset] Located timestamps for {len(contract_timestamps)} contracts. Missing/Empty: {missing}")
        
        # Sort chronologically
        sorted_contracts = sorted(contract_timestamps.items(), key=lambda x: x[1])
        
        # Build split indices
        n_total = len(sorted_contracts)
        n_train = int(n_total * train_ratio)
        
        train_cids = {x[0] for x in sorted_contracts[:n_train]}
        stream_cids = {x[0] for x in sorted_contracts[n_train:]}
        
        self.chronological_order = [x[0] for x in sorted_contracts[n_train:]]
        
        log.info(f"[StreamingDataset] Chronological Split - Historical Context: {len(train_cids)}, Streaming Sequence: {len(stream_cids)}")
        
        # Assign back to the base datastructures dynamically
        self.train_graphs = [g for g in self.transaction_graphs if getattr(g, "contract_id", "") in train_cids]
        
        # In this dataset, the explicit streaming sequence retains chronological ordering
        stream_unsorted = {getattr(g, "contract_id", ""): g for g in self.transaction_graphs if getattr(g, "contract_id", "") in stream_cids}
        self.stream_graphs = [stream_unsorted[c] for c in self.chronological_order if c in stream_unsorted]
        
        return self.train_graphs, self.stream_graphs

    def get_streaming_graphs(self):
        """ Returns the strictly ordered testing graphs """
        return getattr(self, "stream_graphs", self.test_graphs)
