"""A08 provider CSV observables. This module has no label or old graph input.

Prediction unit: token contract. Transaction address observations are static;
zero address (mint/burn endpoint) is retained as an observed endpoint.
"""
from __future__ import annotations
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

FEATURE_NAMES = ('log_transaction_rows', 'log_unique_senders', 'log_unique_receivers',
                 'log_unique_addresses', 'log_unique_directed_pairs',
                 'self_transfer_fraction', 'reciprocal_pair_fraction',
                 'repeated_pair_fraction')
ADDRESS = re.compile(r'^0x[0-9a-f]{40}$')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def array_hash(array):
    a = np.ascontiguousarray(array)
    if a.dtype.kind == 'O': raise ValueError('object arrays forbidden')
    h = hashlib.sha256(json.dumps({'dtype': a.dtype.str, 'shape': list(a.shape)}, sort_keys=True).encode())
    h.update(a.tobytes()); return h.hexdigest()


def normalize_contract(value):
    value = str(value).strip().lower()
    if not ADDRESS.fullmatch(value): raise ValueError(f'invalid contract address: {value}')
    return value


def transaction_features(stream):
    """Only from/to fields are parsed; missing endpoints cause explicit failure.

    Retain all rows including repeated transfers, zero-address and self-transfer.
    Reciprocal fraction excludes self pairs. Empty CSV is a valid zero-activity
    contract (no exclusion based on labels, activity or performance).
    """
    df = pd.read_csv(stream, usecols=['from', 'to'], dtype=str, keep_default_na=False)
    s = df['from'].str.strip().str.lower(); t = df['to'].str.strip().str.lower()
    if not s.str.fullmatch(r'0x[0-9a-f]{40}').all() or not t.str.fullmatch(r'0x[0-9a-f]{40}').all():
        raise ValueError('missing or malformed transaction endpoint; no silent row deletion')
    count = len(df)
    if not count: return np.zeros(8, dtype=np.float64), {'rows': 0, 'addresses': 0, 'pairs': 0}
    ids, unique = pd.factorize(pd.concat([s, t], ignore_index=True), sort=True)
    a, b = ids[:count].astype(np.int64), ids[count:].astype(np.int64)
    n = len(unique); pairs = np.unique(a*n+b)
    u, v = pairs//n, pairs % n
    nonself = u != v
    reciprocal = np.isin(v[nonself]*n+u[nonself], pairs).sum()
    raw = np.array([np.log1p(count), np.log1p(np.unique(a).size),
                    np.log1p(np.unique(b).size), np.log1p(n), np.log1p(pairs.size),
                    np.mean(a == b), reciprocal/max(1, int(nonself.sum())),
                    1-pairs.size/count], dtype=np.float64)
    return raw, {'rows': count, 'addresses': n, 'pairs': int(pairs.size)}


def build_observables(transaction_zip, chain, progress=None):
    """Read the CSV ZIP directly; never use JSON/PT/features caches or labels."""
    with zipfile.ZipFile(transaction_zip) as z:
        members = {}
        for member in z.infolist():
            if member.is_dir(): continue
            if not member.filename.endswith('.csv'): raise ValueError('unexpected non-CSV raw payload')
            contract = normalize_contract(Path(member.filename).stem)
            if contract in members: raise ValueError('duplicate contract CSV')
            members[contract] = member
        if not members: raise ValueError('empty transaction archive')
        contracts = sorted(members)
        rows, audit = [], []
        for i, contract in enumerate(contracts):
            member = members[contract]
            with z.open(member) as stream:
                # Hash the exact decompressed provider bytes while parsing once.
                content = stream.read()
            feature, counts = transaction_features(io.BytesIO(content))
            rows.append(feature)
            audit.append({'contract': contract, 'member': member.filename,
                          'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(), **counts})
            if progress and (i % 100 == 0 or i+1 == len(contracts)): progress(i+1, len(contracts))
    raw = np.stack(rows)
    # Transductive label-blind fit on the full observed contract population.
    mean = raw.mean(0); std = raw.std(0); scale = np.where(std > 0, std, 1.)
    x = ((raw-mean)/scale).astype(np.float32)
    if not np.isfinite(x).all(): raise ValueError('nonfinite model features')
    return {'node_ids': np.array([f'{chain}:{c}' for c in contracts]), 'raw_features': raw, 'x': x,
            'preprocessing_mean': mean, 'preprocessing_scale': scale}, audit


def deterministic_knn(x, node_ids, k=5, block_size=256, metric='euclidean'):
    """Exact blocked distances, stable-ID ties; binary undirected union.

    No random noise, labels, node-ID feature or full NxN allocation. Stable
    ordering is explicit, including ties at kth boundary. Distances use float64.
    """
    x = np.asarray(x, dtype=np.float64)
    ids = np.asarray(node_ids)
    if x.ndim != 2 or len(ids) != len(x) or len(set(ids.tolist())) != len(ids) or not np.isfinite(x).all():
        raise ValueError('invalid relation input')
    if len(x) < 2 or np.unique(x, axis=0).shape[0] < 2: raise ValueError('degenerate informative relation')
    if metric == 'cosine':
        norm = np.linalg.norm(x, axis=1)
        if np.any(norm == 0): raise ValueError('zero-norm cosine feature')
        x = x/norm[:, None]
        if np.allclose(x, x[:1], atol=1e-12, rtol=0): raise ValueError('cosine directions fully degenerate')
    elif metric != 'euclidean': raise ValueError('unapproved metric')
    order = np.argsort(ids, kind='stable'); sorted_x = x[order]
    n = len(x); k = min(k, n-1)
    if k < 1: raise ValueError('k must be positive')
    # Differences computed directly to avoid negative roundoff in squared norms.
    src, dst = [], []; boundary_ties = 0
    for start in range(0, n, block_size):
        stop = min(start+block_size, n)
        if metric == 'euclidean':
            dist = np.zeros((stop-start, n), dtype=np.float64)
            for j in range(x.shape[1]): dist += (sorted_x[start:stop, j, None]-sorted_x[None, :, j])**2
        else: dist = 1-sorted_x[start:stop] @ sorted_x.T
        dist[np.arange(stop-start), np.arange(start, stop)] = np.inf
        nearest = np.argsort(dist, axis=1, kind='stable')[:, :k]
        values = np.take_along_axis(dist, nearest, axis=1)
        boundary_ties += int(np.sum((dist == values[:, -1, None]).sum(1) > 1))
        src.append(np.repeat(order[start:stop], k)); dst.append(order[nearest].ravel())
    a, b = np.concatenate(src), np.concatenate(dst)
    pairs = np.unique(np.concatenate([a*n+b, b*n+a]))
    edge = np.stack([pairs//n, pairs % n]).astype(np.int64)
    return edge, {'metric': metric, 'k': k, 'kth_boundary_tie_rows': boundary_ties,
                  'tie_rule': 'ascending stable contract ID', 'directedness': 'undirected union',
                  'weights': 'binary unit', 'self_loops': False, 'duplicate_rule': 'binary union'}


def save_arrays(path, arrays):
    path = Path(path)
    if path.exists(): raise FileExistsError('immutable output exists: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def content_hash(arrays):
    return hashlib.sha256(json.dumps({k: array_hash(v) for k, v in sorted(arrays.items())}, sort_keys=True).encode()).hexdigest()
