"""A29-05c: measured resources for {Cora-Syn, Elliptic, Reddit-Syn, DGraphFin} x 7 models, seed 42,
primary configuration and full epoch budget. Replaces the analytic `est.` column of Table 6.

  python a29_resource_matrix.py all          # driver, one subprocess per cell, 3 h cap
  python a29_resource_matrix.py cell G M     # one cell
  python a29_resource_matrix.py summarize    # merges GADNR telemetry from A29-06
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time

from a29_common import LOCAL, RESULTS, dump, environment, read_json, writecsv

OUT = RESULTS / 'resource_matrix'
RUNS = LOCAL / 'resource_matrix'
GRAPHS = ['Cora-Syn', 'Elliptic', 'DGraphFin', 'Reddit-Syn']
MODELS = ['DOMINANT', 'DLG-Base', 'DLG-Aug', 'OCGNN', 'CoLA', 'AnomalyDAE', 'GADNR']
CAP = 3 * 3600


def cell(g, m):
    import psutil
    from a29_models import load_primary, run
    peak = [0]; stop = threading.Event(); proc = psutil.Process()

    def sample():
        while not stop.wait(0.5):
            peak[0] = max(peak[0], proc.memory_info().rss)
    threading.Thread(target=sample, daemon=True).start()
    data, y, masks = load_primary(g)
    r = run(m, data, y, masks[42], 42, RUNS / g / m, save_scores=False, extra={'graph': g})
    r.pop('_det', None)
    stop.set()
    r['host_peak_rss_bytes'] = peak[0]
    r['environment'] = environment()
    dump(RUNS / g / m / 'result.json', r)
    print(g, m, r['status'], r.get('fit_seconds'), r.get('peak_allocated_bytes'), flush=True)


def driver():
    for g in GRAPHS:
        for m in MODELS:
            if m == 'GADNR' and g != 'Cora-Syn':
                continue  # measured by the A29-06 retry with a 12 h cap
            f = RUNS / g / m
            if (f / 'result.json').exists():
                continue
            f.mkdir(parents=True, exist_ok=True)
            t = time.time()
            try:
                rc = subprocess.run([sys.executable, '-u', __file__, 'cell', g, m], stdout=(f / 'stdout.log').open('w'),
                                    stderr=subprocess.STDOUT, timeout=CAP).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
            if not (f / 'result.json').exists():
                dump(f / 'result.json', {'graph': g, 'model': m, 'seed': 42,
                                         'status': 'TIMEOUT_OBSERVED' if rc == 'timeout' else ('HOST_OOM' if rc in (-9, 137) else 'ERROR'),
                                         'returncode': rc, 'wall_seconds': time.time() - t, 'cap_seconds': CAP})
            print(g, m, read_json(f / 'result.json')['status'], flush=True)


def summarize():
    rows = []
    for g in GRAPHS:
        for m in MODELS:
            p = RUNS / g / m / 'result.json'
            if m == 'GADNR' and g != 'Cora-Syn':
                q = LOCAL / 'gadnr_retry' / f'{g}__seed42' / 'result.json'
                if q.exists():
                    h = read_json(q)
                    rows.append({'graph': g, 'model': m, 'status': h.get('status'), 'source': 'A29-06 retry (12 h cap)',
                                 'fit_seconds': h.get('fit_seconds'), 'score_seconds': h.get('score_seconds'),
                                 'peak_allocated_bytes': h.get('cuda_peak_allocated_bytes'),
                                 'peak_reserved_bytes': h.get('cuda_peak_reserved_bytes'),
                                 'host_peak_rss_bytes': h.get('peak_rss_bytes'),
                                 'mean_epoch_seconds': (sum(h['epoch_seconds']) / len(h['epoch_seconds'])) if h.get('epoch_seconds') else None,
                                 'note': (h.get('error') or '')[:200]})
                else:
                    rows.append({'graph': g, 'model': m, 'status': 'NOT_RUN'})
                continue
            if not p.exists():
                rows.append({'graph': g, 'model': m, 'status': 'NOT_RUN'}); continue
            r = read_json(p)
            rows.append({'graph': g, 'model': m, 'status': r.get('status'), 'source': 'A29-05c',
                         **{k: r.get(k) for k in ('fit_seconds', 'score_seconds', 'peak_allocated_bytes', 'peak_reserved_bytes',
                                                  'host_peak_rss_bytes', 'exceeds_physical_vram', 'pr_auc')},
                         'note': (r.get('error') or '')[:200]})
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union: GADNR rows carry mean_epoch_seconds
    writecsv(OUT / 'resource_matrix.csv', [{k: r.get(k) for k in keys} for r in rows])
    dump(OUT / 'resource_matrix.json', rows)
    for r in rows:
        print(r['graph'], r['model'], r['status'], r.get('fit_seconds'), r.get('peak_allocated_bytes'))


if __name__ == '__main__':
    {'all': driver, 'summarize': summarize}.get(sys.argv[1], lambda: cell(sys.argv[2], sys.argv[3]))()
