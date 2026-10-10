"""Leakage-explicit DGraphFin NPZ loader preserving official arrays."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data


def _split_mask(raw: np.ndarray, old_to_new: torch.Tensor, retained_nodes: torch.Tensor) -> torch.Tensor:
    values = np.asarray(raw)
    if values.dtype == np.bool_ or (values.ndim == 1 and values.size == old_to_new.numel()
                                    and np.isin(values, [0, 1]).all()):
        old_ids = torch.from_numpy(np.flatnonzero(values)).long()
    else:
        old_ids = torch.from_numpy(values.reshape(-1)).long()
    mapped = old_to_new[old_ids]
    mapped = mapped[mapped >= 0]
    mask = torch.zeros(retained_nodes.numel(), dtype=torch.bool)
    mask[mapped] = True
    return mask


def load_dgraphfin_aligned(npz_path: str | Path) -> Data:
    path = Path(npz_path)
    with np.load(path) as source:
        required = {"x", "y", "edge_index", "edge_timestamp", "train_mask", "valid_mask", "test_mask"}
        missing = sorted(required.difference(source.files))
        if missing:
            raise KeyError(f"DGraphFin NPZ missing required arrays: {missing}")
        x = torch.from_numpy(source["x"]).float()
        y = torch.from_numpy(source["y"]).long().reshape(-1)
        raw_edges = torch.from_numpy(source["edge_index"]).long()
        edge_index = raw_edges.t().contiguous() if raw_edges.shape[1] == 2 else raw_edges.contiguous()
        edge_timestamp = torch.from_numpy(source["edge_timestamp"]).long().reshape(-1)
        edge_type = torch.from_numpy(source["edge_type"]).long().reshape(-1) if "edge_type" in source.files else None
        split_arrays = {name: np.array(source[name], copy=True) for name in ("train_mask", "valid_mask", "test_mask")}
    if edge_index.size(1) != edge_timestamp.numel():
        raise ValueError("raw edge_index and edge_timestamp lengths differ")
    if edge_type is not None and edge_type.numel() != edge_timestamp.numel():
        raise ValueError("raw edge_type and edge_timestamp lengths differ")
    known = (y == 0) | (y == 1)
    retained = torch.nonzero(known, as_tuple=False).flatten()
    old_to_new = torch.full((y.numel(),), -1, dtype=torch.long)
    old_to_new[retained] = torch.arange(retained.numel())
    edge_keep = known[edge_index[0]] & known[edge_index[1]]
    filtered_edges = old_to_new[edge_index[:, edge_keep]]
    data = Data(x=x[retained].clone(), y=y[retained].clone(), edge_index=filtered_edges,
                edge_timestamp=edge_timestamp[edge_keep].clone(), num_nodes=int(retained.numel()))
    if edge_type is not None:
        data.edge_type = edge_type[edge_keep].clone()
    data.original_node_id = retained
    data.train_mask = _split_mask(split_arrays["train_mask"], old_to_new, retained)
    data.val_mask = _split_mask(split_arrays["valid_mask"], old_to_new, retained)
    data.test_mask = _split_mask(split_arrays["test_mask"], old_to_new, retained)
    data.eval_mask = data.val_mask | data.test_mask
    if data.edge_index.size(1) != data.edge_timestamp.numel():
        raise AssertionError("filtered edge/timestamp alignment failed")
    if data.edge_index.numel() and int(data.edge_index.max()) >= data.num_nodes:
        raise AssertionError("filtered edge index is out of range")
    if bool((data.train_mask & data.val_mask).any() or (data.train_mask & data.test_mask).any()
            or (data.val_mask & data.test_mask).any()):
        raise AssertionError("official split masks overlap")
    return data
