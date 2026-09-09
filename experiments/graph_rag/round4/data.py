"""Load the packed chronological GoG dataset as PyG objects."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data


def load_packed(dataset_dir: Path):
    payload = torch.load(dataset_dir / "graph.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((dataset_dir / "real_dataset_manifest.json").read_text())
    graphs = []
    for index, item in enumerate(payload["graphs"]):
        graphs.append(Data(
            x=item["x"], edge_index=item["edge_index"],
            y=torch.tensor(float(item["label"])),
            # Avoid PyG's automatic offsetting of attributes whose names contain
            # ``index``; chain_id is a graph-level categorical value.
            chain_id=torch.tensor(item["chain_index"], dtype=torch.long),
            event_pos=torch.tensor(index, dtype=torch.long),
            timestamp=torch.tensor(item["timestamp"], dtype=torch.long),
        ))
    n_train = manifest["split"]["train"]["n_events"]
    n_validation = manifest["split"]["validation"]["n_events"]
    return payload, manifest, {
        "train": graphs[:n_train],
        "validation": graphs[n_train:n_train + n_validation],
        "test": graphs[n_train + n_validation:],
    }
