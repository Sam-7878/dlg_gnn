"""A29-06: GADNR retries without the 300 s cap, with telemetry.

  python a29_gadnr_retry.py all            # driver: one subprocess per cell, 12 h cap
  python a29_gadnr_retry.py cell NAME SEED # one cell (called by the driver)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import traceback

from a29_common import LOCAL, RESULTS, dump, environment, pin_3090, read_json

CAP = 12 * 3600
CELLS = ['Elliptic', 'Flickr-Syn', 'Yelp-Syn', 'DGraphFin', 'Reddit-Syn']
OUT = RESULTS / 'gadnr_retry'
RUNS = LOCAL / 'gadnr_retry'


class Exceeds(Exception):
    pass


class Telemetry:
    def __init__(self, path):
        import psutil
        self.path, self.proc = path, psutil.Process(os.getpid())
        self.t0 = time.perf_counter()
        self.rec = {'phase': 'start', 'phases': {}, 'epoch_seconds': [], 'peak_rss_bytes': 0, 'heartbeat_utc': None}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        while not self._stop.wait(1.0):
            try:
                self.rec['peak_rss_bytes'] = max(self.rec['peak_rss_bytes'], self.proc.memory_info().rss)
            except Exception:
                pass
            self.beat()

    def beat(self):
        with self._lock:
            try:
                self._beat()
            except OSError:
                pass  # telemetry must never abort the measured run

    def _beat(self):
        self.rec['elapsed_seconds'] = time.perf_counter() - self.t0
        self.rec['heartbeat_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        try:
            import torch
            self.rec['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            self.rec['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        except Exception:
            pass
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.rec, indent=1, default=str))
        tmp.replace(self.path)

    def phase(self, name, seconds=None):
        self.rec['phase'] = name
        if seconds is not None:
            self.rec['phases'][name] = seconds
        self.beat()

    def close(self):
        self._stop.set()
        self.beat()


def cell(name: str, seed: int):
    pin_3090()
    import random
    import numpy as np
    import torch
    from torch_geometric.data import Data
    from a24_inputs import load_input
    from a24_run_studies import metrics
    from gog_fraud.models.pygod.gadnr import GADNR
    from gog_fraud.models.pygod.sparse_message import AutoSparseFusedGCN
    from gog_fraud.evaluation.reproducibility import seed_everything

    folder = RUNS / f'{name}__seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    tel = Telemetry(folder / 'heartbeat.json')
    tel.rec.update(dataset=name, seed=seed, cap_seconds=CAP, environment=environment())
    t = time.perf_counter()
    data, masks, identity = load_input(name)
    tel.phase('load_input', time.perf_counter() - t)
    tel.rec['input'] = {k: identity[k] for k in ('N', 'E', 'F', 'feature_hash', 'edge_hash', 'source_sha256')}
    y = data.y.numpy().reshape(-1).copy()
    d = Data(x=data.x.clone().float(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
    del data
    seed_everything(seed, deterministic=True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    det = GADNR(epoch=50, gpu=0, batch_size=0, verbose=0, num_neigh=-1, backbone=AutoSparseFusedGCN)
    torch.cuda.reset_peak_memory_stats()

    orig_pg = det.process_graph

    def timed_pg(data):
        tel.phase('process_graph')
        s = time.perf_counter()
        out = orig_pg(data)
        tel.phase('process_graph_done', time.perf_counter() - s)
        tel.rec['fit_started_epochs_at'] = tel.rec['elapsed_seconds']
        return out
    det.process_graph = timed_pg

    last = [None]
    step = torch.optim.Adam.step
    fit_start = [None]

    def timed_step(opt, *a, **kw):
        out = step(opt, *a, **kw)
        torch.cuda.synchronize()
        now = time.perf_counter()
        prev = last[0] if last[0] is not None else fit_start[0]
        tel.rec['epoch_seconds'].append(now - prev)
        last[0] = now
        done = len(tel.rec['epoch_seconds'])
        tel.phase(f'epoch_{done}')
        mean = sum(tel.rec['epoch_seconds']) / done
        projected = (now - tel.t0) + (50 - done) * mean
        tel.rec['projected_total_seconds'] = projected
        if projected > CAP:
            raise Exceeds(f'projected {projected/3600:.2f} h > cap {CAP/3600:.0f} h after {done} epochs (mean {mean:.1f} s/epoch)')
        return out

    # epochs are timed from the end of process_graph; patch forward to mark the first start
    orig_fwd = det.forward_model

    def fwd(data):
        if fit_start[0] is None:
            fit_start[0] = time.perf_counter()
            tel.phase('first_forward')
        return orig_fwd(data)
    det.forward_model = fwd

    status, err = 'COMPLETED', None
    torch.optim.Adam.step = timed_step
    try:
        s0 = time.perf_counter()
        det.fit(d)
        tel.rec['fit_seconds'] = time.perf_counter() - s0
    except Exceeds as e:
        status, err = 'EXCEEDS_CAP_PROJECTED', str(e)
    except torch.OutOfMemoryError as e:
        status, err = 'GPU_OOM', repr(e)[:2000]
    except MemoryError as e:
        status, err = 'HOST_OOM', repr(e)[:2000]
    except Exception as e:  # CUDA driver OOM (e.g. inside torch.inverse) arrives as AcceleratorError, not OutOfMemoryError
        status = 'GPU_OOM' if 'out of memory' in str(e).lower() else 'OTHER_ERROR'
        err = traceback.format_exc()[-4000:]
    finally:
        torch.optim.Adam.step = step
    if status == 'COMPLETED':
        tel.phase('scoring')
        s0 = time.perf_counter()
        s = det.decision_function(d)
        s = s.detach().cpu().numpy() if hasattr(s, 'detach') else np.asarray(s)
        tel.rec['score_seconds'] = time.perf_counter() - s0
        summary, full = metrics(y, s, masks[seed])
        np.savez_compressed(folder / 'scores.npz', scores=s, labels=y, **masks[seed])
        dump(folder / 'metrics.json', full)
        tel.rec['metrics'] = summary
    tel.rec['status'] = status
    tel.rec['error'] = err
    tel.phase('finished')
    tel.close()
    dump(folder / 'result.json', tel.rec)
    print(name, seed, status, err or '', flush=True)


def driver(seeds=(42,), cells=CELLS):
    OUT.mkdir(parents=True, exist_ok=True)
    for name in cells:
        for seed in seeds:
            folder = RUNS / f'{name}__seed{seed}'
            if (folder / 'result.json').exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            log = (folder / 'stdout.log').open('a')
            started = time.time()
            cmd = [sys.executable, '-u', __file__, 'cell', name, str(seed)]
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=CAP + 1800).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
            if not (folder / 'result.json').exists():
                hb = read_json(folder / 'heartbeat.json') if (folder / 'heartbeat.json').exists() else {}
                status = 'TIMEOUT_OBSERVED' if rc == 'timeout' else ('HOST_OOM' if rc in (-9, 137) else 'OTHER_ERROR')
                hb.update(status=status, returncode=rc, wall_seconds=time.time() - started,
                          note='process ended without writing result.json; last heartbeat preserved')
                dump(folder / 'result.json', hb)
            r = read_json(folder / 'result.json')
            print(name, seed, r['status'], flush=True)
    summarize()


def summarize():
    """All retry records (every cell, seeds 42-46 where run) plus the earlier attempts kept beside them."""
    summary = []
    for name in CELLS:
        for seed in (42, 43, 44, 45, 46):
            folder = RUNS / f'{name}__seed{seed}'
            p = folder / 'result.json'
            if p.exists():
                r = read_json(p)
                rec = {k: r.get(k) for k in ('dataset', 'seed', 'status', 'error', 'phases', 'epoch_seconds',
                                             'projected_total_seconds', 'elapsed_seconds', 'fit_seconds',
                                             'score_seconds', 'peak_rss_bytes', 'cuda_peak_allocated_bytes',
                                             'cuda_peak_reserved_bytes', 'metrics', 'input', 'returncode', 'phase',
                                             'host_memory_cap', 'recorded_by')}
                rec['earlier_attempts'] = [{'path': str(d.relative_to(RUNS)),
                                            'note': read_json(d / 'NOTE.json') if (d / 'NOTE.json').exists() else None,
                                            'status': read_json(d / 'result.json').get('status') if (d / 'result.json').exists() else None}
                                           for d in sorted(folder.glob('attempt*')) if d.is_dir()]
                summary.append(rec)
    dump(OUT / 'gadnr_retry_summary.json', summary)


if __name__ == '__main__':
    if sys.argv[1] == 'cell':
        cell(sys.argv[2], int(sys.argv[3]))
    elif sys.argv[1] == 'all':
        driver()
    elif sys.argv[1] == 'summarize':
        summarize()
    elif sys.argv[1] == 'seeds':
        driver(seeds=(43, 44, 45, 46), cells=sys.argv[2:])
