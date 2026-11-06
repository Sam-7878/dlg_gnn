"""Audit the primary AP Friedman views (MC + exact enumeration) and View S2 post-hoc Wilcoxon tests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import scipy
from scipy.stats import beta, chi2, f, friedmanchisquare, rankdata, wilcoxon

EXACT_COUNTS = {
    'S1': (124255729032510633885, 82606411253903523840000000),
    'S2': (128550002279805893073, 8916100448256000000000000),
    'S3': (4783900740, 24883200000),
    'S4': (8321720, 2985984000000),
}

def rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
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
    s2_wilcoxon = []
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
        scaled = (2*ranks).astype(np.int64)
        cutoff = int(np.square(scaled.sum(axis=0)).sum())
        q = float((3 * cutoff / (n*k*(k+1)) - 3*n*(k+1)) / correction)
        assert np.isclose(q, float(view['Q']), atol=1e-12, rtol=0)
        assert np.isclose(q, friedmanchisquare(*ap.T).statistic, atol=1e-12, rtol=0)
        original = independent = 0
        for _ in range(10):
            ix = np.argsort(old_rng.random((n,100_000,k)), axis=2)
            sums = np.take_along_axis(scaled[:,None,:], ix, axis=2).sum(axis=0)
            original += np.count_nonzero(np.square(sums).sum(axis=1) >= cutoff)
            sums = np.zeros((100_000,k), dtype=np.int64)
            for row in scaled:
                sums += independent_rng.permuted(np.broadcast_to(row, sums.shape), axis=1)
            independent += np.count_nonzero(np.square(sums).sum(axis=1) >= cutoff)
        original, independent = int(original), int(independent)
        assert original == int(view['exceedances'])
        tail = (original+1)/(draws+1)
        reference = float(chi2.sf(q,k-1))
        id_f = float((n-1)*q/(n*(k-1)-q))
        intervals = {}
        for name, count in [('original', original), ('independent', independent)]:
            low = beta.ppf(.025, count, draws-count+1) if count else 0.
            high = beta.ppf(.975, count+1, draws-count) if count < draws else 1.
            intervals[name] = [float(low), float(high)]
        ex_num, ex_den = EXACT_COUNTS[view['view_id']]
        exact_p = float(ex_num / ex_den)
        within_ci = (intervals['original'][0] <= exact_p <= intervals['original'][1] and
                     intervals['independent'][0] <= exact_p <= intervals['independent'][1])
        assert within_ci
        results.append(dict(
            view=view['view_id'], datasets=datasets, models=models, n=n, k=k,
            Q=q, W=q/(n*(k-1)), ties=tied_blocks, correction=correction,
            doubled_ranks=scaled.tolist(), original_b=original,
            independent_b=independent, draws=draws, seed=20261024, MC_p=tail,
            independent_p=(independent+1)/(draws+1), chi2_p=reference,
            ratio=reference/tail, ID_F=id_f, ID_p=float(f.sf(id_f,k-1,(n-1)*(k-1))),
            MC_tail_CI95=intervals,
            exact_exceedances=ex_num, exact_null_assignments=ex_den,
            exact_p=exact_p, chi2_over_exact=reference/exact_p,
            exact_within_both_separate_MC_tail_CI95=within_ci
        ))
        if view['view_id'] == 'S2':
            aug_idx = models.index('DLG-Aug')
            aug_vec = ap[:, aug_idx]
            raw_tests = []
            for idx, m in enumerate(models):
                if m == 'DLG-Aug':
                    continue
                m_vec = ap[:, idx]
                diff = aug_vec - m_vec
                n_nz = int(np.count_nonzero(diff != 0))
                n_pos = int(np.count_nonzero(diff > 0))
                n_neg = int(np.count_nonzero(diff < 0))
                res = wilcoxon(aug_vec, m_vec, zero_method='wilcox', alternative='two-sided', method='exact')
                raw_tests.append(dict(comparator=m, mean_rank=float(ranks[:, idx].mean()),
                                      W_stat=float(res.statistic), n_total=n, n_eff=n_nz,
                                      n_positive=n_pos, n_negative=n_neg, zero_method='wilcox',
                                      raw_p=float(res.pvalue)))
            order = sorted(range(len(raw_tests)), key=lambda i: raw_tests[i]['raw_p'])
            running = 0.0
            m_comp = len(raw_tests)
            for rank_pos, i in enumerate(order):
                adj = min(1.0, raw_tests[i]['raw_p'] * (m_comp - rank_pos))
                running = max(running, adj)
                raw_tests[i]['holm_p_adj'] = float(running)
            s2_wilcoxon = raw_tests
        print(results[-1]['view'], original, independent, 'exact_p', f'{exact_p:.6g}', flush=True)
    return {'numpy': np.__version__, 'scipy': scipy.__version__, 'views': results, 's2_posthoc_wilcoxon': s2_wilcoxon}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(run(args.repository), indent=2) + '\n', encoding='utf-8')

