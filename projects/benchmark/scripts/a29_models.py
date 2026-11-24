"""A29 detector factory and single-run executor (primary configuration by default).

Mirrors gog_fraud.pipelines.run_sci_round4c._instantiate, i.e. the configuration
that produced the primary runs; `hp` overrides are used only by the declared
A29-04 tuning grid.
"""
from __future__ import annotations

import gc
import json
import random
import time

from a29_common import LOCAL, ROOT, dump, pin_3090, sha

pin_3090()
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Data  # noqa: E402

GNN = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']
FEATURE = ['IForest', 'ECOD']
VRAM = None


def vram():
    global VRAM
    if VRAM is None:
        VRAM = torch.cuda.get_device_properties(0).total_memory
    return VRAM


def make(model: str, epoch: int = 50, **hp):
    from gog_fraud.models.pygod.shared_reconstruction import (SharedAnomalyDAE, SharedDLGBase, SharedDLGFull,
                                                              SharedDOMINANT)
    from gog_fraud.models.pygod.sparse_message import AutoSparseFusedGCN
    common = {'epoch': epoch, 'gpu': 0, 'batch_size': 0, 'verbose': 0}
    if model == 'AnomalyDAE':
        return SharedAnomalyDAE(**common, reconstruction_backend='chunked_exact', score_chunk_size=256, **hp)
    if model in ('DOMINANT', 'DLG-Base', 'DLG-Aug'):
        cls = {'DOMINANT': SharedDOMINANT, 'DLG-Base': SharedDLGBase, 'DLG-Aug': SharedDLGFull}[model]
        kw = dict(common, message_backend='sparse_fused', reconstruction_backend='exact_sparse',
                  gradient_checkpointing=False, score_chunk_size=8192, **hp)
        if model == 'DLG-Aug':
            kw.setdefault('l1_epochs', 20)
        return cls(**kw)
    if model == 'CoLA':
        from pygod.detector import CoLA as cls
    elif model == 'OCGNN':
        from pygod.detector import OCGNN as cls
    elif model == 'GADNR':
        from gog_fraud.models.pygod.gadnr import GADNR as cls
    else:
        raise ValueError(model)
    return cls(**common, num_neigh=-1, backbone=AutoSparseFusedGCN, **hp)


def metrics(y, s, ms):
    from gog_fraud.data.benchmark_metrics import evaluate_scores
    v = evaluate_scores(y, s, np.arange(len(y)), ms['val_mask'], ms['test_mask'])
    op = v['operating_point']
    out = {'pr_auc': v['ap'], 'roc_auc': v['roc_auc'], 'test_f1': v['thresholded']['f1'] if v['thresholded'] else None,
           'test_N': v['test_total'], 'test_positive': v['test_positives'], 'test_prevalence': v['test_prevalence']}
    for q, b in v['alert_budgets'].items():
        key = '1' if q == '0.01' else '5'
        out.update({f'P_at_{key}pct': b['precision'], f'R_at_{key}pct': b['recall'],
                    f'lift_{key}pct': b['precision'] / v['test_prevalence'], f'K_{key}pct': b['k']})
    return out


def val_ap(y, s, ms):
    from sklearn.metrics import average_precision_score
    v = ms['val_mask']
    return float(average_precision_score(y[v], s[v]))


def run(model, data: Data, y, masks_seed, seed, folder, epoch=50, hp=None, save_scores=True, extra=None, test=True,
        detector=None):
    """Fit once (labels never passed), score, and return a record. Never raises.

    test=False computes validation AP only (configuration search); test metrics
    are then computed later, once, for the selected configuration.
    """
    hp = hp or {}
    folder.mkdir(parents=True, exist_ok=True)
    rec = {'model': model, 'seed': seed, 'epoch': epoch, 'hp': hp, **(extra or {}),
           'gpu_name': torch.cuda.get_device_name(0), 'cuda_visible_devices': __import__('os').environ.get('CUDA_VISIBLE_DEVICES')}
    from gog_fraud.evaluation.reproducibility import seed_everything
    seed_everything(seed, deterministic=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        if model in FEATURE:
            x = data.x.numpy()
            if model == 'IForest':
                from sklearn.ensemble import IsolationForest
                det = IsolationForest(n_estimators=100, max_samples=256, max_features=1., bootstrap=False,
                                      contamination='auto', random_state=seed, n_jobs=4)
                det.fit(x); fit = time.perf_counter() - t0; t1 = time.perf_counter(); s = -det.score_samples(x)
            else:
                from pyod.models.ecod import ECOD
                det = ECOD(contamination=.1, n_jobs=1)
                det.fit(x); fit = time.perf_counter() - t0; t1 = time.perf_counter(); s = np.asarray(det.decision_scores_)
            score_s = time.perf_counter() - t1
        else:
            det = detector if detector is not None else make(model, epoch=epoch, **hp)
            d = Data(x=data.x.clone().float(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
            det.fit(d)
            torch.cuda.synchronize(); fit = time.perf_counter() - t0
            t1 = time.perf_counter()
            s = det.decision_function(d)
            torch.cuda.synchronize(); score_s = time.perf_counter() - t1
            s = s.detach().cpu().numpy() if torch.is_tensor(s) else np.asarray(s)
            rec['loss_history_tail'] = [float(v) for v in list(getattr(det, 'loss_history_', []))[-3:]]
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        if s.size != len(y) or not np.isfinite(s).all():
            raise FloatingPointError('non-finite score or wrong length')
        rec.update(status='success', fit_seconds=fit, score_seconds=score_s)
        if test:
            rec.update(metrics(y, s, masks_seed))
        rec['val_ap'] = val_ap(y, s, masks_seed)
        rec['_det'] = det if model not in FEATURE else None
        rec['score_sd'] = float(s.std()); rec['score_unique_frac'] = float(len(np.unique(s)) / len(s))
        if save_scores:
            np.savez_compressed(folder / 'scores.npz', scores=s, labels=y, **masks_seed)
            rec['raw_score_sha256'] = sha(folder / 'scores.npz')
    except torch.OutOfMemoryError as e:
        rec.update(status='GPU_OOM', error=str(e)[:600])
    except Exception as e:
        import traceback
        rec.update(status='ERROR', error=traceback.format_exc()[-1500:])
    context_lost = False
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            rec['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            rec['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            rec['exceeds_physical_vram'] = bool(rec['peak_reserved_bytes'] > vram())
        except Exception as e:  # sticky CUDA error: the context is unusable for any later cell
            context_lost = True
            if rec.get('status') == 'success':
                rec['status'] = 'ERROR'
            if 'out of memory' in str(e).lower() or rec.get('status') == 'GPU_OOM':
                rec['status'] = 'GPU_OOM'
            rec['cuda_context_lost'] = True
            rec['error'] = (rec.get('error') or '') + ' | post-run synchronize: ' + str(e)[:400]
    rec['wall_seconds'] = time.perf_counter() - t0
    dump(folder / 'result.json', {k: v for k, v in rec.items() if k != '_det'})
    if context_lost:
        # record written; exit so the next cell starts in a fresh process instead of inheriting a dead context
        print(rec.get('graph', ''), model, seed, rec['status'], 'cuda_context_lost', flush=True)
        __import__('os')._exit(3)
    gc.collect(); torch.cuda.empty_cache()
    return rec


def load_real(name):
    """A29-08 frozen graph + 50/50 stratified validation/test over evaluable nodes per seed."""
    from sklearn.model_selection import train_test_split
    a = np.load(LOCAL / 'real_graphs' / f'{name}.npz', allow_pickle=False)
    y = a['y'].astype(np.int64)
    ev = np.flatnonzero(a['eval_mask'])
    data = Data(x=torch.from_numpy(np.ascontiguousarray(a['x'])), edge_index=torch.from_numpy(np.ascontiguousarray(a['edge_index'])), num_nodes=len(y))
    masks = {}
    for s in (42, 43, 44, 45, 46):
        va, te = train_test_split(ev, test_size=.5, random_state=s, stratify=y[ev])
        n = len(y)
        m = {'train_mask': np.ones(n, bool), 'val_mask': np.zeros(n, bool), 'test_mask': np.zeros(n, bool)}
        m['val_mask'][va] = True; m['test_mask'][te] = True
        masks[s] = m
    return data, y, masks


def load_primary(name):
    from a24_inputs import load_input
    data, masks, identity = load_input(name)
    y = data.y.numpy().reshape(-1).astype(np.int64)
    return Data(x=data.x.float(), edge_index=data.edge_index, num_nodes=data.num_nodes), y, masks
