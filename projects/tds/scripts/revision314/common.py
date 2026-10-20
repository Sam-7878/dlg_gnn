"""Shared paths and identity checks for the Applied Sciences revision."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[4]
PROJECT = ROOT / 'projects/tds'
OUT = PROJECT / 'paper_applsci'
FROZEN = PROJECT / 'results/main_final_v2'
GRAPH = ROOT / 'archive/tds/data/gog_scimain_v1/graph.pt'
GRAPH_HASH = '067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd'
RAW_ROOT = Path(os.environ.get('TDS_RAW_ROOT', str(ROOT.parent.parent/'_data/gog_round7_upstream/Token Data')))
SEEDS = (7, 17, 27, 37, 47)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def load():
    if sha(GRAPH) != GRAPH_HASH:
        raise RuntimeError('Frozen packed graph identity changed')
    payload = torch.load(GRAPH, map_location='cpu', weights_only=False)
    records = []
    graphs = []
    for i, g in enumerate(payload['graphs']):
        chain, contract, _ = g['event_id'].split(':')
        split = 'train' if i < 17021 else 'validation' if i < 20668 else 'test'
        records.append(dict(event_id=g['event_id'], timestamp=g['timestamp'],
                            label=g['label'], chain=chain, contract=contract, split=split,
                            chain_index=g['chain_index'], num_nodes=len(g['x']),
                            num_edges=g['edge_index'].shape[1],
                            max_input_edge_timestamp=g['max_input_edge_timestamp']))
        graphs.append(Data(x=g['x'], edge_index=g['edge_index'],
                           y=torch.tensor(float(g['label'])),
                           chain_id=torch.tensor(g['chain_index']),
                           event_pos=torch.tensor(i), timestamp=torch.tensor(g['timestamp'])))
    frame = pd.DataFrame(records)
    assert len(frame) == 24316 and frame.event_id.is_unique
    assert list(frame.groupby('split').label.sum().sort_index()) == [107, 6783, 308]
    assert frame.timestamp.is_monotonic_increasing
    for split in ['validation', 'test']:
        pred = pd.read_csv(FROZEN/f'raw_predictions/tgat/seed7_{split}.csv')
        expected = frame.loc[frame.split == split, ['event_id', 'timestamp', 'label']].reset_index(drop=True)
        assert pred[['event_id', 'timestamp', 'label']].equals(expected), split
    return payload, frame, graphs


def summary_features(g):
    """Capped topology degrees PLUS observed full-history node features.

    Degree features in packed x are computed before edge truncation. Including
    their summaries gives the tabular controls access to that same information.
    """
    n = g.num_nodes
    e = g.edge_index.shape[1]
    indeg = torch.bincount(g.edge_index[1], minlength=n).float()
    outdeg = torch.bincount(g.edge_index[0], minlength=n).float()
    v = [float(n), float(e), float(e / max(1, n * (n - 1)))]
    for a in (indeg, outdeg, indeg + outdeg):
        v.extend([float(a.mean()), float(a.quantile(.5)), float(a.max())])
    for col in range(g.x.shape[1]):
        a = g.x[:, col]
        v.extend([float(a.mean()), float(a.quantile(.5)), float(a.max())])
    v.extend(np.eye(3)[int(g.chain_id)].tolist())
    return v


def aligned_prediction(path, frame):
    p = pd.read_csv(path)
    expected = frame.loc[frame.split == 'test', ['event_id', 'timestamp', 'label']].reset_index(drop=True)
    assert p[['event_id', 'timestamp', 'label']].equals(expected), path
    return p.p_mean.to_numpy(float)
