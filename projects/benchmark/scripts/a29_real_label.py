"""A29-08 real-label extension: 7 primary detectors + IForest + ECOD, seeds 42-46.

  python a29_real_label.py run GRAPH
  python a29_real_label.py summarize
"""
from __future__ import annotations

import sys

from a29_common import LOCAL, RESULTS, dump, read_json, writecsv

OUT = RESULTS / 'real_label'
RUNS = LOCAL / 'real_label'
GRAPHS = ['Weibo', 'Reddit-Real', 'Tolokers', 'Amazon-Fraud', 'YelpChi']
ORDER = ['IForest', 'ECOD', 'DOMINANT', 'DLG-Base', 'DLG-Aug', 'OCGNN', 'CoLA', 'AnomalyDAE', 'GADNR']
SEEDS = [42, 43, 44, 45, 46]


def run_graph(g):
    from a29_models import load_real, run
    data, y, masks = load_real(g)
    for m in ORDER:
        for s in SEEDS:
            folder = RUNS / g / m / str(s)
            if (folder / 'result.json').exists():
                continue
            r = run(m, data, y, masks[s], s, folder, extra={'graph': g})
            r.pop('_det', None)
            print(g, m, s, r['status'], 'AP', r.get('pr_auc'), 'ROC', r.get('roc_auc'), 'fit', round(r.get('fit_seconds') or 0, 1), flush=True)


def run_cell(g, m, s):
    from a29_models import load_real, run
    data, y, masks = load_real(g)
    r = run(m, data, y, masks[s], s, RUNS / g / m / str(s), extra={'graph': g})
    print(g, m, s, r['status'], 'AP', r.get('pr_auc'), 'ROC', r.get('roc_auc'), 'fit', round(r.get('fit_seconds') or 0, 1), flush=True)


def drive(g, models, cap_hours):
    """One subprocess per missing cell with a wall-clock cap (same 12 h rule as A29-06 by default).

    A cell that dies without writing result.json gets one with the observed status. After a TIMEOUT the
    remaining seeds of that model are recorded as NOT_ATTEMPTED (one capped seed already shows the budget is exceeded).
    """
    import subprocess
    import time
    from a29_common import dump
    for m in models:
        timed_out = None
        for s in SEEDS:
            f = RUNS / g / m / str(s)
            if (f / 'result.json').exists():
                continue
            f.mkdir(parents=True, exist_ok=True)
            if timed_out is not None:
                dump(f / 'result.json', {'graph': g, 'model': m, 'seed': s, 'status': 'NOT_ATTEMPTED',
                                         'reason': f'seed {timed_out} exceeded the {cap_hours} h cap'})
                continue
            t = time.time()
            try:
                rc = subprocess.run([sys.executable, '-u', __file__, 'cell', g, m, str(s)], timeout=cap_hours * 3600).returncode
            except subprocess.TimeoutExpired:
                rc = 'timeout'
            if not (f / 'result.json').exists():
                status = 'TIMEOUT_OBSERVED' if rc == 'timeout' else ('HOST_OOM' if rc in (-9, 137) else 'CRASH')
                dump(f / 'result.json', {'graph': g, 'model': m, 'seed': s, 'status': status, 'returncode': rc,
                                         'wall_seconds': time.time() - t, 'cap_seconds': cap_hours * 3600})
                print(g, m, s, status, flush=True)
            if rc == 'timeout':
                timed_out = s


def summarize():
    import numpy as np
    rows, agg = [], []
    for g in GRAPHS:
        for m in ORDER:
            rs = [read_json(p) for p in sorted((RUNS / g / m).glob('*/result.json'))] if (RUNS / g / m).exists() else []
            rows += [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in rs]
            ok = [r for r in rs if r.get('status') == 'success']
            rec = {'graph': g, 'model': m, 'n_success': len(ok), 'n_attempted': len(rs),
                   'statuses': ';'.join(sorted({r['status'] for r in rs}))}
            if len(ok) == 5:
                for k in ('pr_auc', 'roc_auc', 'P_at_1pct', 'P_at_5pct', 'lift_1pct', 'lift_5pct', 'test_prevalence',
                          'fit_seconds', 'peak_allocated_bytes'):
                    v = np.array([r[k] for r in ok], float)
                    rec[k + '_mean'] = float(v.mean()); rec[k + '_sd'] = float(v.std(ddof=1))
            agg.append(rec)
    writecsv(OUT / 'real_label_runs.csv', rows)
    writecsv(OUT / 'real_label_summary.csv', agg)
    dump(OUT / 'real_label_summary.json', agg)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run_graph(sys.argv[2])
    elif sys.argv[1] == 'cell':
        run_cell(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    elif sys.argv[1] == 'drive':   # drive GRAPH CAP_HOURS MODEL [MODEL ...]
        drive(sys.argv[2], sys.argv[4:], float(sys.argv[3]))
    else:
        summarize()
