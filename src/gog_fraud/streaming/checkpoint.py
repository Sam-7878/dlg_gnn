from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PipelineCheckpoint:
    stream: Mapping[str, Any]
    subgraph_store: Mapping[str, Any]
    relation_state: list[Mapping[str, Any]]
    embedding_cache: Mapping[str, Any]
    queues: Mapping[str, Any]
    model_version: str
    threshold_version: str


def save_checkpoint(checkpoint: PipelineCheckpoint, path: str | Path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(checkpoint), sort_keys=True, separators=(",", ":"), default=str)
    target.write_text(payload + "\n", encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def load_checkpoint(path: str | Path) -> tuple[PipelineCheckpoint, str]:
    payload = Path(path).read_text(encoding="utf-8").strip()
    return PipelineCheckpoint(**json.loads(payload)), hashlib.sha256(payload.encode()).hexdigest()
