"""A29-05b: peak GPU memory and epoch time versus graph size N.

Synthetic undirected random graph, mean degree 10 (both directions stored),
64 standard-normal features, generator seed 0. Three epochs per fit (DLG-Aug:
3 local + 3 global). One process per model; after an OOM or a spill past
physical VRAM, larger N are skipped for that model.
  python a29_scaling_sweep.py MODEL
  python a29_scaling_sweep.py summarize
"""
from __future__ import annotations

import sys
import time

from a29_common import RESULTS, dump, environment, pin_3090, read_json

OUT = RESULTS / 'scaling_sweep'
OUT.mkdir(parents=True, exist_ok=True)
NS = [5000, 10000, 20000, 50000, 100000, 200000]
MODELS = ['DOMINANT-Gram', 'DLG-Base-Gram', 'DLG-Aug-Gram', 'AnomalyDAE-rowblock', 'DOMINANT-dense']
HOST_DENSE_BUDGET = 12e9  # bytes of host RAM we allow PyGOD's CPU-side to_dense_adj to use (WSL2 has 20 GB)


def graph(n):
    import numpy as np
    import torch
    from torch_geometric.data import Data
    rng = np.random.default_rng(0)
    m = 5 * n
    a, b = rng.integers(0, n, m), rng.integers(0, n, m)
    keep = a != b
    a, b = a[keep], b[keep]
    ei = np.unique(np.vstack([np.r_[a, b], np.r_[b, a]]).T, axis=0).T
    x = rng.standard_normal((n, 64)).astype(np.float32)
    return Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(np.ascontiguousarray(ei.astype(np.int64))), num_nodes=n)


def one(model):
    pin_3090()
    import torch
    from pygod.detector import DOMINANT
    from a29_models import make
    vram = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(1.0, 0)  # no spill past physical VRAM: a too-large N is an OOM
    res = {'model': model, 'environment': environment(), 'points': [],
           'memory_cap': 'torch.cuda.set_per_process_memory_fraction(1.0): no spill past physical VRAM'}
    stop = None
    for n in NS:
        pt = {'N': n}
        if stop:
            pt.update(status='SKIPPED', reason=f'larger N after {stop}')
            res['points'].append(pt); continue
        if model == 'DOMINANT-dense' and 4 * n * n > HOST_DENSE_BUDGET:
            pt.update(status='NOT_ATTEMPTED_ARITHMETIC', reason=f'dense float32 adjacency alone needs 4N^2 = {4*n*n/1e9:.1f} GB '
                      'in host RAM before the GPU copy; above the 12 GB host budget')
            res['points'].append(pt); stop = stop or 'arithmetic bound'; continue
        d = graph(n)
        pt['E'] = int(d.edge_index.shape[1])
        torch.manual_seed(0)
        if model == 'DOMINANT-dense':
            det = DOMINANT(epoch=3, gpu=0, batch_size=0, verbose=0)
        else:
            base = model.split('-Gram')[0].split('-rowblock')[0]
            det = make(base, epoch=3, **({'l1_epochs': 3} if base == 'DLG-Aug' else {}))
        steps = []
        step = torch.optim.Adam.step

        def timed(opt, *a, **k):
            torch.cuda.synchronize(); t = time.perf_counter(); out = step(opt, *a, **k); torch.cuda.synchronize()
            steps.append(time.perf_counter()); return out
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.optim.Adam.step = timed
        t0 = time.perf_counter()
        try:
            det.fit(d)
            torch.cuda.synchronize()
            pt['fit_seconds'] = time.perf_counter() - t0
            pt['optimizer_steps'] = len(steps)
            pt['mean_epoch_seconds'] = float((steps[-1] - steps[0]) / max(len(steps) - 1, 1)) if len(steps) > 1 else None
            pt['status'] = 'COMPLETED'
        except torch.OutOfMemoryError as e:
            pt.update(status='GPU_OOM', error=str(e)[:300]); stop = f'GPU OOM at N={n}'
        except Exception as e:
            pt.update(status='ERROR', error=repr(e)[:500]); stop = f'error at N={n}'
        finally:
            torch.optim.Adam.step = step
        res['vram_bytes'] = vram
        try:
            pt['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            pt['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        except Exception as e:  # sticky CUDA error after an OOM: keep the record, stop this model
            pt['cuda_context_lost'] = True; pt['status'] = 'GPU_OOM' if 'memory' in str(e).lower() else pt['status']
            res['points'].append(pt); print(model, pt, flush=True)
            for m in NS[NS.index(n) + 1:]:
                res['points'].append({'N': m, 'status': 'SKIPPED', 'reason': f'larger N after CUDA context loss at N={n}'})
            dump(OUT / f'{model}.json', res)
            __import__('os')._exit(3)
        if pt['peak_reserved_bytes'] > vram:
            pt['exceeds_physical_vram'] = True
            stop = stop or f'reserved memory passed physical VRAM at N={n} (driver system-memory fallback)'
        res['points'].append(pt)
        print(model, pt, flush=True)
        dump(OUT / f'{model}.json', res)   # incremental: a later crash cannot erase earlier points
        del det, d
        torch.cuda.empty_cache()
    res['vram_bytes'] = vram
    dump(OUT / f'{model}.json', res)


def summarize():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, slopes = [], {}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    style = {'DOMINANT-dense': ('#c0392b', 's', '--'), 'DOMINANT-Gram': ('#2c3e50', 'o', '-'),
             'DLG-Base-Gram': ('#16a085', '^', '-'), 'DLG-Aug-Gram': ('#8e44ad', 'v', '-'),
             'AnomalyDAE-rowblock': ('#d35400', 'D', ':')}
    vram = None
    for m in MODELS:
        p = OUT / f'{m}.json'
        if not p.exists():
            continue
        r = read_json(p); vram = r['vram_bytes']
        ok = [q for q in r['points'] if q.get('status') == 'COMPLETED' and not q.get('exceeds_physical_vram')]
        for q in r['points']:
            rows.append({'model': m, **{k: q.get(k) for k in ('N', 'E', 'status', 'peak_allocated_bytes', 'peak_reserved_bytes',
                                                               'mean_epoch_seconds', 'fit_seconds', 'reason', 'exceeds_physical_vram')}})
        if len(ok) >= 2:
            n = np.array([q['N'] for q in ok], float); mem = np.array([q['peak_allocated_bytes'] for q in ok], float)
            slopes[m] = float(np.polyfit(np.log(n), np.log(mem), 1)[0])
            c, mk, ls = style[m]
            ax[0].loglog(n, mem / 2**30, marker=mk, ls=ls, color=c, label=f'{m} (slope {slopes[m]:.2f})', ms=4)
            t = np.array([q['mean_epoch_seconds'] or np.nan for q in ok], float)
            ax[1].loglog(n, t, marker=mk, ls=ls, color=c, ms=4)
            bad = [q for q in r['points'] if q.get('status') in ('GPU_OOM', 'NOT_ATTEMPTED_ARITHMETIC') or q.get('exceeds_physical_vram')]
            if bad:
                ax[0].plot([bad[0]['N']], [vram / 2**30], marker='x', color=c, ms=8)
    if vram:
        ax[0].axhline(vram / 2**30, color='grey', lw=0.8, ls='--'); ax[0].text(5200, vram / 2**30 * 1.08, '24 GB VRAM', fontsize=7, color='grey')
    ax[0].set_xlabel('nodes N (mean degree 10)'); ax[0].set_ylabel('peak allocated GPU memory (GiB)')
    ax[1].set_xlabel('nodes N (mean degree 10)'); ax[1].set_ylabel('seconds per training epoch')
    ax[0].legend(fontsize=6, loc='lower right'); fig.tight_layout()
    fig.savefig(OUT / 'fig_memory_scaling.pdf'); fig.savefig(OUT / 'fig_memory_scaling.png', dpi=200)
    from a29_common import writecsv
    writecsv(OUT / 'scaling_points.csv', rows)
    dump(OUT / 'scaling_slopes.json', slopes)
    print(slopes)


if __name__ == '__main__':
    summarize() if sys.argv[1] == 'summarize' else one(sys.argv[1])
