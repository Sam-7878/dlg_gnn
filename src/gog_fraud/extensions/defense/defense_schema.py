"""Defense extension schema and metadata definitions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch_geometric.data import Data


@dataclass
class DefenseManifest:
    dataset_name: str
    official_source_name: str
    source_citation: str
    source_url_or_doi: str
    provenance_details: str
    num_nodes: int
    num_edges: int
    num_features: int
    num_positives: int
    num_negatives: int
    positive_ratio: float
    time_range_description: str
    node_definition: str
    edge_definition: str
    ground_truth_definition: str
    negative_label_semantics: str
    graph_sha256: str
    feature_sha256: str
    label_sha256: str
    split_strategy: str = "stratified_node_transductive"
    is_temporal: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Compute deterministic SHA-256 hash of a PyTorch tensor."""
    array_bytes = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(array_bytes).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file on disk."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
