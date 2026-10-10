"""Strict, independent node view for contract-node benchmarks (not Stream/TDS)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from .crypto_raw import array_hash, content_hash, normalize_contract, sha256


def attach_labels(node_ids, label_csv, chain):
    import csv
    mapping = {}
    with Path(label_csv).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            if row['Chain'].strip().lower() != chain: continue
            contract = normalize_contract(row['Contract'])
            if contract in mapping: raise ValueError('duplicate/conflicting label ID')
            category = int(row['Category'])
            if category < 0: raise ValueError('unknown category')
            mapping[contract] = int(category == 0)
    target, valid = [], []
    for identifier in node_ids:
        prefix, contract = str(identifier).split(':', 1)
        if prefix != chain: raise ValueError('wrong chain identity')
        valid.append(contract in mapping); target.append(mapping.get(contract, -1))
    return np.array(target, dtype=np.int64), np.array(valid, dtype=bool)


def node_view(source, node_ids, dataset_id, *, labels=None, label_valid_mask=None, masks=None):
    """Never mutate graph-level source.y; node labels must be explicit vectors."""
    if not dataset_id or not dataset_id.endswith('_contract_clean_v1'): raise ValueError('explicit A08 dataset identity required')
    x = source.x.detach().clone(); edge = source.edge_index.detach().clone()
    ids = [str(v) for v in node_ids]; n = len(ids)
    if x.ndim != 2 or n != x.shape[0] or int(source.num_nodes) != n or len(set(ids)) != n:
        raise ValueError('node identity/shape mismatch')
    if not torch.isfinite(x).all(): raise ValueError('nonfinite features')
    if edge.dtype != torch.int64 or edge.ndim != 2 or edge.shape[0] != 2: raise ValueError('invalid edge shape/dtype')
    if edge.numel() and (edge.min() < 0 or edge.max() >= n): raise ValueError('edge out of range')
    target = labels
    aliases = []
    for name in ('labels', 'level1_label'):
        value = getattr(source, name, None)
        if value is not None: aliases.append(torch.as_tensor(value))
    if target is None:
        if not aliases: raise ValueError('explicit node labels missing; graph y is never broadcast')
        target = aliases[0]
    target = torch.as_tensor(target)
    if target.ndim != 1 or target.shape != (n,) or target.dtype != torch.int64: raise ValueError('node labels must be int64 N-vector')
    for alias in aliases:
        if alias.ndim != 1 or alias.shape != (n,) or alias.dtype != torch.int64 or not torch.equal(alias, target):
            raise ValueError('label alias conflict')
    valid = torch.ones(n, dtype=torch.bool) if label_valid_mask is None else torch.as_tensor(label_valid_mask)
    if valid.dtype != torch.bool or valid.shape != (n,): raise ValueError('invalid valid mask')
    if not torch.isin(target[valid], torch.tensor([0, 1])).all() or not (target[~valid] == -1).all(): raise ValueError('invalid binary/unknown target')
    result = Data(x=x, edge_index=edge, y=target.clone(), num_nodes=n)
    weight = getattr(source, 'edge_weight', None)
    if weight is not None:
        if weight.ndim != 1 or weight.shape != (edge.shape[1],) or not torch.isfinite(weight).all(): raise ValueError('invalid edge weight')
        result.edge_weight = weight.clone()
    result.node_ids = ids; result.dataset_id = dataset_id; result.prediction_unit = 'contract_node'
    result.label_valid_mask = valid.clone()
    graph_y = getattr(source, 'y', None)
    if graph_y is not None: result.graph_y = graph_y.detach().clone()
    occupied = torch.zeros(n, dtype=torch.bool)
    for name, mask in (masks or {}).items():
        mask = torch.as_tensor(mask)
        if mask.dtype != torch.bool or mask.shape != (n,) or (occupied & mask).any() or (mask & ~valid).any():
            raise ValueError('invalid/overlapping split mask')
        occupied |= mask; setattr(result, name, mask.clone())
    return result


def load_frozen_node_graph(manifest_path, expected_dataset_id, seed):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest['dataset_id'] != expected_dataset_id or manifest['scientific_status'] != 'FROZEN_APPROVED_INPUT':
        raise ValueError('unapproved dataset version')
    path = manifest_path.parent / manifest['artifact_relative_path']
    if sha256(path) != manifest['serialized_file_sha256']: raise ValueError('artifact byte hash mismatch')
    with np.load(path, allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
    if content_hash(arrays) != manifest['scientific_content_hash']: raise ValueError('scientific content mismatch')
    masks = {name: arrays[f'{name}_{seed}'] for name in ('train_mask', 'val_mask', 'test_mask')}
    data = node_view(Data(x=torch.from_numpy(arrays['x']), edge_index=torch.from_numpy(arrays['edge_index']), num_nodes=len(arrays['node_ids'])),
                     arrays['node_ids'], expected_dataset_id, labels=arrays['labels'], label_valid_mask=arrays['label_valid_mask'], masks=masks)
    data.input_manifest_hash = sha256(manifest_path)
    data.split_hash = hashlib.sha256(json.dumps({k: array_hash(v) for k,v in masks.items()}, sort_keys=True).encode()).hexdigest()
    return data
