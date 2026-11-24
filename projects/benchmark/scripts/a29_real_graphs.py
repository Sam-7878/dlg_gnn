"""A29-08: build frozen real-label graphs from the downloaded source files.

Downloads live in local_storage/benchmark/a29/raw_data/<name>/ (not redistributed).
Output: local_storage/benchmark/a29/real_graphs/<Name>.npz with x, edge_index, y,
eval_mask, plus results/a29/real_label/DATASETS_A29.json (hashes, counts, provenance).
Edges are stored in both directions, coalesced, explicit self-loops removed.
"""
import io
import zipfile

import numpy as np
import scipy.io
import scipy.sparse as sp

from a29_common import LOCAL, RESULTS, dump, pin_3090, sha

RAW = LOCAL / 'raw_data'
OUT = LOCAL / 'real_graphs'
OUT.mkdir(parents=True, exist_ok=True)
REC = RESULTS / 'real_label'
REC.mkdir(parents=True, exist_ok=True)


def tensor_hash(a):
    import hashlib
    a = np.asarray(a); h = hashlib.sha256(); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def clean_edges(rows, cols, n):
    m = sp.coo_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n)).tocsr()
    m = ((m + m.T) > 0).astype(np.float32).tocoo()
    keep = m.row != m.col
    ei = np.vstack([m.row[keep], m.col[keep]]).astype(np.int64)
    order = np.lexsort((ei[1], ei[0]))
    return ei[:, order]


def from_pygod_pt(path):
    pin_3090()
    import torch
    from torch_geometric.data import Data
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
    from torch_geometric.data.storage import GlobalStorage
    torch.serialization.add_safe_globals([Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith('.pt')][0]
        g = torch.load(io.BytesIO(z.read(name)), weights_only=True)
    x = g.x.numpy().astype(np.float32)
    y = g.y.numpy().reshape(-1).astype(np.int64)
    ei = g.edge_index.numpy()
    return x, ei[0], ei[1], y, None


def from_mat(path, key_x='features'):
    with zipfile.ZipFile(path) as z:
        name = [n for n in z.namelist() if n.endswith('.mat')][0]
        m = scipy.io.loadmat(io.BytesIO(z.read(name)))
    x = m[key_x]
    x = (x.toarray() if sp.issparse(x) else np.asarray(x)).astype(np.float32)
    homo = sp.coo_matrix(m['homo'])
    y = np.asarray(m['label']).reshape(-1).astype(np.int64)
    return x, homo.row, homo.col, y


SOURCES = {
    'Weibo': dict(file='weibo/weibo.pt.zip', url='https://github.com/pygod-team/data/raw/main/weibo.pt.zip',
                  origin='Weibo user-post graph distributed with PyGOD/BOND (Liu et al., NeurIPS 2022 Datasets and Benchmarks); '
                         'labels mark suspicious users from the original study (Zhao et al., 2020).',
                  domain='social media / spam accounts'),
    'Reddit-Real': dict(file='reddit/reddit.pt.zip', url='https://github.com/pygod-team/data/raw/main/reddit.pt.zip',
                        origin='Reddit user-subreddit post graph distributed with PyGOD/BOND; labels mark users banned by the platform '
                               '(Kumar et al., KDD 2019). Unrelated to the Reddit-Syn primary graph.',
                        domain='social media / banned users'),
    'Tolokers': dict(file='tolokers/tolokers.npz', url='https://github.com/yandex-research/heterophilous-graphs/raw/main/data/tolokers.npz',
                     origin='Toloka crowdsourcing worker graph (Platonov et al., ICLR 2023); edge = two workers on the same task; '
                            'label = worker banned on a project.',
                     domain='crowdsourcing / fraudulent workers'),
    'YelpChi': dict(file='yelpchi/YelpChi.zip', url='https://github.com/YingtongDou/CARE-GNN/raw/master/data/YelpChi.zip',
                    origin='Yelp hotel/restaurant reviews (Rayana and Akoglu, KDD 2015) as prepared by CARE-GNN (Dou et al., CIKM 2020); '
                           'homogeneous union relation; label = filtered (spam) review.',
                    domain='e-commerce / review spam'),
    'Amazon-Fraud': dict(file='amazon/Amazon.zip', url='https://github.com/YingtongDou/CARE-GNN/raw/master/data/Amazon.zip',
                         origin='Amazon musical-instrument users (McAuley and Leskovec, WWW 2013) as prepared by CARE-GNN; homogeneous union '
                                'relation; label = fraudulent user. Nodes 0-3304 carry no label in CARE-GNN and are excluded from evaluation.',
                         domain='e-commerce / fraudulent users'),
}

records = {}
for name, src in SOURCES.items():
    path = RAW / src['file']
    if name in ('Weibo', 'Reddit-Real'):
        x, r, c, y, _ = from_pygod_pt(path)
    elif name == 'Tolokers':
        a = np.load(path, allow_pickle=False)
        x = a['node_features'].astype(np.float32); y = a['node_labels'].reshape(-1).astype(np.int64)
        r, c = a['edges'][:, 0], a['edges'][:, 1]
    else:
        x, r, c, y = from_mat(path)
    n = x.shape[0]
    ei = clean_edges(np.asarray(r), np.asarray(c), n)
    eval_mask = np.ones(n, dtype=bool)
    if name == 'Amazon-Fraud':
        eval_mask[:3305] = False
    assert set(np.unique(y[eval_mask])) <= {0, 1}, (name, np.unique(y))
    np.savez_compressed(OUT / f'{name}.npz', x=x, edge_index=ei, y=y, eval_mask=eval_mask)
    deg = np.bincount(ei[0], minlength=n)
    records[name] = {**src, 'source_file_sha256': sha(path), 'frozen_file': str((OUT / f'{name}.npz').name),
                     'frozen_file_sha256': sha(OUT / f'{name}.npz'), 'N': int(n), 'E_directed': int(ei.shape[1]),
                     'F': int(x.shape[1]), 'evaluable_nodes': int(eval_mask.sum()), 'positives': int(y[eval_mask].sum()),
                     'prevalence': float(y[eval_mask].mean()), 'isolated_nodes': int((deg == 0).sum()),
                     'feature_hash': tensor_hash(x), 'edge_hash': tensor_hash(ei), 'label_hash': tensor_hash(y),
                     'edge_processing': 'symmetrised, coalesced, explicit self-loops removed; features unchanged (float32)',
                     'redistribution': 'not redistributed; obtain from the source URL under its terms'}
    print(name, {k: records[name][k] for k in ('N', 'E_directed', 'F', 'evaluable_nodes', 'positives', 'prevalence', 'isolated_nodes')}, flush=True)
dump(REC / 'DATASETS_A29.json', records)
