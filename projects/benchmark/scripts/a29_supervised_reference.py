"""A29-12: supervised reference line (labels ARE used for fitting; never mixed into unsupervised ranks).

HistGradientBoostingClassifier (sklearn defaults, random_state=seed) on X and on
[X, mean of 1-hop neighbour X]; fitted on labelled nodes disjoint from the test split
(train split when disjoint, otherwise the validation half); evaluated on the same test
splits as the unsupervised detectors.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '8')
from a29_common import RESULTS, dump, writecsv  # noqa: E402
from a29_models import load_primary, load_real, metrics  # noqa: E402

import numpy as np  # noqa: E402

OUT = RESULTS / 'supervised_reference'
OUT.mkdir(parents=True, exist_ok=True)
PRIMARY = ['Elliptic', 'DGraphFin', 'Ethereum', 'BSC', 'Polygon', 'LANL-RedTeam']
REAL = ['Weibo', 'Reddit-Real', 'Tolokers', 'Amazon-Fraud', 'YelpChi']


def neighbour_mean(x, ei, n):
    import scipy.sparse as sp
    a = sp.coo_matrix((np.ones(ei.shape[1], np.float32), (ei[1], ei[0])), shape=(n, n)).tocsr()
    deg = np.asarray(a.sum(1)).ravel(); deg[deg == 0] = 1
    return (a @ x) / deg[:, None]


def main(graphs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows = []
    for g in graphs:
        data, y, masks = load_real(g) if g in REAL else load_primary(g)
        x = data.x.numpy().astype(np.float32)
        xn = np.hstack([x, neighbour_mean(x, data.edge_index.numpy(), data.num_nodes)])
        for seed in (42, 43, 44, 45, 46):
            m = masks[seed]
            fitpop = m['train_mask'] & ~m['test_mask'] & ~m['val_mask']
            source = 'train split'
            if fitpop.sum() == 0 or len(np.unique(y[fitpop])) < 2:
                fitpop, source = m['val_mask'].copy(), 'validation half'
            fitpop &= np.isin(y, (0, 1))
            for name, feats in (('HGB-X', x), ('HGB-X+NeighbourMean', xn)):
                clf = HistGradientBoostingClassifier(random_state=seed)
                clf.fit(feats[fitpop], y[fitpop])
                s = clf.predict_proba(feats)[:, 1]
                r = {'graph': g, 'model': name, 'seed': seed, 'fit_population': source, 'n_fit': int(fitpop.sum()),
                     'fit_positives': int(y[fitpop].sum()), **metrics(y, s, m)}
                rows.append(r)
                print(g, name, seed, source, 'AP', round(r['pr_auc'], 4), 'ROC', round(r['roc_auc'], 4), flush=True)
    writecsv(OUT / f'supervised_runs_{"_".join(graphs) if len(graphs) < 3 else "all"}.csv', rows)
    agg = []
    for g in graphs:
        for name in ('HGB-X', 'HGB-X+NeighbourMean'):
            rr = [r for r in rows if r['graph'] == g and r['model'] == name]
            a = np.array([r['pr_auc'] for r in rr]); c = np.array([r['roc_auc'] for r in rr]); p = np.array([r['test_prevalence'] for r in rr])
            agg.append({'graph': g, 'model': name, 'fit_population': rr[0]['fit_population'], 'AP_mean': a.mean(), 'AP_sd': a.std(ddof=1),
                        'ROC_mean': c.mean(), 'ROC_sd': c.std(ddof=1), 'test_prevalence_mean': p.mean(),
                        'P_at_1pct_mean': float(np.mean([r['P_at_1pct'] for r in rr]))})
    dump(OUT / f'supervised_summary_{"all" if len(graphs) >= 3 else "_".join(graphs)}.json', agg)


if __name__ == '__main__':
    main(sys.argv[1:] or PRIMARY + REAL)
