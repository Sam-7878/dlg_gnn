"""A29-05a: dense PyGOD reference vs production exact-Gram path at real scale.

For each graph and family (DOMINANT, DLG-Base):
  dense run A, dense run B (same initial weights; GPU noise floor),
  Gram run (initial weights copied from the dense model).
All runs: seed 42, 50 epochs, primary hyper-parameters, float32, RTX 3090.
  python a29_dense_vs_gram.py GRAPH [GRAPH ...]
"""
from __future__ import annotations

import copy
import sys
import time
import traceback

from a29_common import LOCAL, RESULTS, dump, environment, pin_3090

pin_3090()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.utils import to_dense_adj  # noqa: E402
from pygod.detector import DOMINANT as PyGODDOMINANT  # noqa: E402
from a24_inputs import load_input  # noqa: E402
from a24_run_studies import metrics  # noqa: E402
from gog_fraud.models.pygod.dlg import DLG  # noqa: E402
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGBase, SharedDOMINANT  # noqa: E402

OUT = RESULTS / 'dense_vs_gram'
OUT.mkdir(parents=True, exist_ok=True)
SEED, EPOCHS = 42, 50
HOST_DENSE_BUDGET = 12e9  # host RAM allowed for PyGOD's CPU-side to_dense_adj (WSL2 has 20 GB); same as A29-05b
# No spill: cap PyTorch's allocator at physical VRAM so 'does not fit on a 24 GB card' is an OOM, not a slow spill into
# shared system memory over the eGPU link (set once per process, before any allocation).
torch.cuda.set_per_process_memory_fraction(1.0, 0)
HP = dict(hid_dim=64, num_layers=4, dropout=0., weight_decay=0., lr=0.004, weight=0.5, batch_size=0, verbose=0)


def peak():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


def reset():
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def dense_run(family, data, state0):
    reset()
    torch.manual_seed(SEED)
    det = (PyGODDOMINANT(epoch=EPOCHS, gpu=0, **HP) if family == 'DOMINANT'
           else DLG(epoch=EPOCHS, gpu=0, **HP))
    det.num_nodes, det.in_dim = data.x.shape
    model = det.init_model(**det.kwargs)
    det.model = model
    if state0 is not None:
        model.load_state_dict(state0)
    state = copy.deepcopy(model.state_dict())
    n = data.num_nodes
    s = data.clone()
    s.s = to_dense_adj(s.edge_index, max_num_nodes=n)[0]
    s.batch_size, s.n_id = n, torch.arange(n)
    opt = torch.optim.Adam(model.parameters(), lr=HP['lr'], weight_decay=0.)
    losses = []
    t = time.perf_counter()
    model.train()
    for _ in range(EPOCHS):
        loss, _score = det.forward_model(s)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
    fit = time.perf_counter() - t
    model.eval()
    with torch.no_grad():
        _, score = det.forward_model(s)
    alloc, res = peak()
    del s, det, opt
    return {'losses': losses, 'scores': score.numpy().astype(np.float64), 'fit_seconds': fit,
            'peak_allocated_bytes': alloc, 'peak_reserved_bytes': res}, state, model


def gram_run(family, data, state0):
    reset()
    torch.manual_seed(SEED)
    cls = SharedDOMINANT if family == 'DOMINANT' else SharedDLGBase
    prod = cls(epoch=EPOCHS, gpu=0, message_backend='sparse_fused', reconstruction_backend='exact_sparse',
               gradient_checkpointing=False, score_chunk_size=8192, **HP)
    prod.num_nodes, prod.in_dim = data.x.shape
    model = prod.init_model(**prod.kwargs)
    model.load_state_dict(state0)
    prod.init_model = lambda **kw: model
    d = data.clone()
    t = time.perf_counter()
    prod.fit(d)
    fit = time.perf_counter() - t
    score = prod.decision_function(d)
    alloc, res = peak()
    return {'losses': [float(x) for x in prod.loss_history_], 'scores': score.detach().cpu().numpy().astype(np.float64),
            'fit_seconds': fit, 'peak_allocated_bytes': alloc, 'peak_reserved_bytes': res}


def topk_overlap(a, b, idx, frac):
    k = int(np.ceil(frac * len(idx)))
    ta = set(idx[np.argsort(-a[idx], kind='stable')[:k]])
    tb = set(idx[np.argsort(-b[idx], kind='stable')[:k]])
    return len(ta & tb) / k


def compare(x, y, yl, masks):
    test = np.flatnonzero(masks['test_mask'])
    lx, ly = np.array(x['losses']), np.array(y['losses'])
    mx, _ = metrics(yl, x['scores'], masks)
    my, _ = metrics(yl, y['scores'], masks)
    return {'max_rel_loss_diff': float(np.max(np.abs(lx - ly) / np.maximum(np.abs(lx), 1e-30))),
            'final_rel_loss_diff': float(abs(lx[-1] - ly[-1]) / max(abs(lx[-1]), 1e-30)),
            'score_spearman': float(spearmanr(x['scores'], y['scores']).statistic),
            'score_max_rel_diff': float(np.max(np.abs(x['scores'] - y['scores'])) / max(np.max(np.abs(x['scores'])), 1e-30)),
            'top1pct_overlap_test': topk_overlap(x['scores'], y['scores'], test, 0.01),
            'top5pct_overlap_test': topk_overlap(x['scores'], y['scores'], test, 0.05),
            'delta_AP': my['pr_auc'] - mx['pr_auc'], 'delta_ROC': my['roc_auc'] - mx['roc_auc'],
            'AP_x': mx['pr_auc'], 'AP_y': my['pr_auc']}


def one(name):
    data, masks, identity = load_input(name)
    yl = data.y.numpy().reshape(-1).copy()
    d = Data(x=data.x.clone().float(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
    out = {'dataset': name, 'N': identity['N'], 'E': identity['E'], 'F': identity['F'], 'seed': SEED, 'epochs': EPOCHS,
           'dense_adjacency_bytes_float32': 4 * identity['N'] ** 2, 'families': {},
           'memory_cap': 'torch.cuda.set_per_process_memory_fraction(1.0): no spill past physical VRAM'}
    for family in ('DOMINANT', 'DLG-Base'):
        rec = {}
        try:
            if 4 * identity['N'] ** 2 > HOST_DENSE_BUDGET:
                raise MemoryError('arithmetic')
            a, state0, _ = dense_run(family, d, None)
            b, _, _ = dense_run(family, d, state0)
            rec['dense_status'] = 'COMPLETED'
        except (torch.OutOfMemoryError, MemoryError) as e:
            if isinstance(e, MemoryError) and str(e) == 'arithmetic':
                rec['dense_status'] = 'NOT_ATTEMPTED_ARITHMETIC'
                rec['dense_error'] = (f'dense float32 adjacency alone needs 4N^2 = {4 * identity["N"] ** 2 / 1e9:.1f} GB in host RAM '
                                      'before the GPU copy; above the 12 GB host budget (WSL2 has 20 GB)')
            else:
                rec['dense_status'] = 'GPU_OOM' if isinstance(e, torch.OutOfMemoryError) else 'HOST_OOM'
                rec['dense_error'] = str(e)[:500]
            a = b = None
            torch.manual_seed(SEED)
            tmp = (PyGODDOMINANT(epoch=EPOCHS, gpu=0, **HP) if family == 'DOMINANT' else DLG(epoch=EPOCHS, gpu=0, **HP))
            tmp.num_nodes, tmp.in_dim = d.x.shape
            state0 = copy.deepcopy(tmp.init_model(**tmp.kwargs).state_dict())
        torch.cuda.empty_cache()
        try:
            g = gram_run(family, d, state0)
            rec['gram_status'] = 'COMPLETED'
        except Exception:
            rec['gram_status'] = 'ERROR'; rec['gram_error'] = traceback.format_exc()[-1500:]; g = None
        for tag, r in (('dense_A', a), ('dense_B', b), ('gram', g)):
            if r is not None:
                rec[tag] = {k: r[k] for k in ('fit_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes')}
                rec[tag]['loss_first_last'] = [r['losses'][0], r['losses'][-1]]
                np.save(LOCAL / f'dvg_{name}_{family}_{tag}_scores.npy', r['scores'])
        if a is not None:
            rec['noise_floor_denseA_vs_denseB'] = compare(a, b, yl, masks[SEED])
            if g is not None:
                rec['denseA_vs_gram'] = compare(a, g, yl, masks[SEED])
        if g is not None:
            gm, _ = metrics(yl, g['scores'], masks[SEED])
            rec['gram_AP'], rec['gram_ROC'] = gm['pr_auc'], gm['roc_auc']
        out['families'][family] = rec
        print(name, family, {k: v for k, v in rec.items() if k in ('dense_status', 'gram_status', 'noise_floor_denseA_vs_denseB', 'denseA_vs_gram')}, flush=True)
    out['environment'] = environment()
    dump(OUT / f'{name}.json', out)


if __name__ == '__main__':
    for g in sys.argv[1:]:
        if (OUT / f'{g}.json').exists():
            continue
        one(g)
