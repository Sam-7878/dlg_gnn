"""Audit the primary AP Friedman views with two block-permutation samplers."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta, chi2, f, friedmanchisquare, rankdata

def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))

def run(root):
    registry = root / 'projects/benchmark/publication/a16/canonical_selected_run_registry.csv'
    source = root / 'projects/benchmark/publication/a24/stats_master_a24.csv'
    by_cell = {}
    for row in rows(registry):
        cell = by_cell.setdefault((row['dataset'], row['model']), {})
        assert row['seed'] not in cell
        cell[row['seed']] = float(row['pr_auc'])
    old_rng = np.random.default_rng(20261024)
    independent_rng = np.random.default_rng(20261024)
    draws = 1_000_000
    results = []
    for view in rows(source):
        datasets, models = view['datasets'].split(';'), view['models'].split(';')
        n, k = len(datasets), len(models)
        ap = np.array([[np.mean(list(by_cell[d, m].values())) for m in models] for d in datasets])
        assert all(set(by_cell[d, m]) == {'42','43','44','45','46'} for d in datasets for m in models)
        ranks = rankdata(-ap, axis=1, method='average')
        tied_blocks = sum(len(np.unique(row)) != k for row in ranks)
        tie_term = sum(int(np.sum(counts**3 - counts)) for counts in
                       (np.unique(row, return_counts=True)[1] for row in ranks))
        correction = 1 - tie_term / (n * (k**3 - k))
        # Twice the ranks are integers even for tied average ranks.
        scaled = (2*ranks).astype(np.int64)
        cutoff = np.square(scaled.sum(axis=0)).sum()
        q = (3 * cutoff / (n*k*(k+1)) - 3*n*(k+1)) / correction
        assert np.isclose(q, float(view['Q']), atol=1e-12, rtol=0)
        assert np.isclose(q, friedmanchisquare(*ap.T).statistic, atol=1e-12, rtol=0)
        original = independent = 0
        for _ in range(10):
            ix = np.argsort(old_rng.random((n,100_000,k)), axis=2)
            sums = np.take_along_axis(scaled[:,None,:], ix, axis=2).sum(axis=0)
            original += np.count_nonzero(np.square(sums).sum(axis=1) >= cutoff)
            # Independently shuffle each trial's model labels inside each dataset.
            sums = np.zeros((100_000,k), dtype=np.int64)
            for row in scaled:
                sums += independent_rng.permuted(np.broadcast_to(row, sums.shape), axis=1)
            independent += np.count_nonzero(np.square(sums).sum(axis=1) >= cutoff)
        original, independent = int(original), int(independent)
        assert original == int(view['exceedances'])
        tail = (original+1)/(draws+1)
        reference = float(chi2.sf(q,k-1))
        id_f = (n-1)*q/(n*(k-1)-q)
        intervals = {}
        for name, count in [('original', original), ('independent', independent)]:
            low = beta.ppf(.025, count, draws-count+1) if count else 0.
            high = beta.ppf(.975, count+1, draws-count) if count < draws else 1.
            intervals[name] = [float(low), float(high)]
        results.append(dict(view=view['view_id'], datasets=datasets, models=models, n=n,k=k,
                            Q=q, W=q/(n*(k-1)), ties=tied_blocks, correction=correction,
                            doubled_ranks=scaled.tolist(), original_b=original,
                            independent_b=independent, draws=draws, seed=20261024, MC_p=tail,
                            independent_p=(independent+1)/(draws+1), chi2_p=reference,
                            ratio=reference/tail, ID_F=id_f, ID_p=float(f.sf(id_f,k-1,(n-1)*(k-1))),
                            MC_tail_CI95=intervals))
        print(results[-1]['view'], original, independent, 'tied blocks', tied_blocks, flush=True)
    return {'numpy':np.__version__, 'scipy':scipy.__version__, 'views':results}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.repository),indent=2)+'\n')
