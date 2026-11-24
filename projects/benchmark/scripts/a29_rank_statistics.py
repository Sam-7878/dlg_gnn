"""A29-07: robustness of the primary rank claims (no new training).

Inputs: the frozen primary registry (five-seed AP means), view memberships from
statistics_a12.json. Outputs: results/a29/rank_statistics/*.
"""
import itertools
import subprocess

import numpy as np
from scipy.stats import chi2, f as fdist, rankdata

from a29_common import REG, RESULTS, ROOT, dump, read_json, rows, writecsv

OUT = RESULTS / 'rank_statistics'
OUT.mkdir(parents=True, exist_ok=True)
EXACT_SRC = ROOT / 'projects/benchmark/publication/a27/exact_null.cpp'
EXACT_BIN = ROOT / 'local_storage/benchmark/a29/exact_null'
B, MC_SEED, BAYES_SEED = 1_000_000, 20261024, 20261123

reg = rows(REG)
means = {}
for r in reg:
    means.setdefault((r['dataset'], r['model']), []).append(float(r['pr_auc']))
assert all(len(v) == 5 for v in means.values())
means = {k: float(np.mean(v)) for k, v in means.items()}
views = {f'S{i}': (v['included_datasets'], v['models'])
         for i, v in enumerate(read_json(ROOT / 'projects/benchmark/analysis/statistics_a12.json')['views'], 1)}


def matrix(ds, ms):
    return np.array([[means[d, m] for m in ms] for d in ds])


def friedman(r):
    n, k = r.shape
    ties = sum(sum(c ** 3 - c for c in np.unique(row, return_counts=True)[1]) for row in r)
    corr = 1 - ties / (n * (k ** 3 - k))
    q = (12 / (n * k * (k + 1)) * (r.sum(0) ** 2).sum() - 3 * n * (k + 1)) / corr
    return float(q), float(corr)


def mc_p(r, q, corr, seed=MC_SEED):
    n, k = r.shape
    rng = np.random.default_rng(seed)
    b = 0
    for _ in range(B // 100_000):
        ix = np.argsort(rng.random((n, 100_000, k)), axis=2)
        s = np.take_along_axis(r[:, None, :], ix, axis=2).sum(0)
        qp = (12 / (n * k * (k + 1)) * (s ** 2).sum(1) - 3 * n * (k + 1)) / corr
        b += int((qp >= q - 1e-12).sum())
    return b, (b + 1) / (B + 1)


def exact_p(r):
    """Exact tail for tie-free integer rank blocks via the A27 enumerator."""
    if not np.allclose(r, np.round(r)):
        return None
    n, k = r.shape
    if not EXACT_BIN.exists():
        subprocess.run(['g++', '-O2', '-std=c++17', '-o', str(EXACT_BIN), str(EXACT_SRC)], check=True)
    threshold = int(round((r.sum(0) ** 2).sum()))
    out = subprocess.run([str(EXACT_BIN)], input=f'{n} {k} {threshold}\n', capture_output=True, text=True, check=True).stdout.split()
    return int(out[0]) / int(out[1])


def analyse(label, ds, ms, a, note=''):
    r = rankdata(-a, axis=1, method='average')
    n, k = r.shape
    q, corr = friedman(r)
    b, p = mc_p(r, q, corr)
    ex = exact_p(r) if corr == 1 else None
    F = (n - 1) * q / (n * (k - 1) - q) if n * (k - 1) - q > 0 else float('inf')
    rec = {'analysis': label, 'n': n, 'k': k, 'datasets': ';'.join(ds), 'models': ';'.join(ms), 'Q': q,
           'Kendall_W': q / (n * (k - 1)), 'tied_blocks': int(sum(len(set(row)) < k for row in r)),
           'MC_b': b, 'p_MC': p, 'p_exact': ex, 'p_chi2': float(chi2.sf(q, k - 1)),
           'p_F': float(fdist.sf(F, k - 1, (n - 1) * (k - 1))), 'note': note}
    for m, v in zip(ms, r.mean(0)):
        rec['mean_rank_' + m] = float(v)
    print(label, {m: round(float(v), 3) for m, v in zip(ms, r.mean(0))}, 'Q', round(q, 4), 'W', round(rec['Kendall_W'], 4), 'p_MC', p, 'p_exact', ex, flush=True)
    return rec


results = []
# 1. S2 without the two near-chance graphs.
ds2, ms2 = views['S2']
results.append(analyse('S2_full', ds2, ms2, matrix(ds2, ms2), 'reference: primary view'))
keep = [d for d in ds2 if d not in ('Elliptic', 'DGraphFin')]
results.append(analyse('S2_minus_Elliptic_DGraphFin', keep, ms2, matrix(keep, ms2),
                       'drops the two graphs where every detector is near test prevalence'))
# 2. Practical-equivalence grids: AP rounded to an epsilon grid before ranking.
for eps in (0.005, 0.01):
    a = np.round(matrix(ds2, ms2) / eps) * eps
    results.append(analyse(f'S2_epsilon_{eps}', ds2, ms2, a, f'AP rounded to a {eps} grid; equal values share average ranks'))
# 3. The three Gram models only.
gram = ['DOMINANT', 'DLG-Base', 'DLG-Aug']
for v in ('S1', 'S2', 'S4'):
    ds, _ = views[v]
    results.append(analyse(f'{v}_gram_models_only', ds, gram, matrix(ds, gram), 'k=3 sensitivity (CoLA/OCGNN/AnomalyDAE/GADNR removed)'))
keys = list(dict.fromkeys(k for r in results for k in r))
writecsv(OUT / 'rank_sensitivity.csv', [{k: r.get(k, '') for k in keys} for r in results])

# 4. Nemenyi critical differences.
Q_ALPHA = {3: 2.343, 5: 2.728, 7: 2.949}  # Demsar (2006), alpha = 0.05
cd = []
for v in ('S1', 'S2', 'S4'):
    ds, ms = views[v]
    r = rankdata(-matrix(ds, ms), axis=1, method='average')
    n, k = r.shape
    crit = Q_ALPHA[k] * np.sqrt(k * (k + 1) / (6 * n))
    mr = dict(zip(ms, r.mean(0)))
    groups = []
    order = sorted(ms, key=mr.get)
    for i, m in enumerate(order):
        g = [x for x in order[i:] if mr[x] - mr[m] <= crit + 1e-12]
        if len(g) > 1 and not any(set(g) <= set(h) for h in groups):
            groups.append(g)
    cd.append({'view': v, 'n': n, 'k': k, 'q_alpha': Q_ALPHA[k], 'critical_difference': float(crit),
               'mean_ranks': {m: float(mr[m]) for m in order}, 'not_separated_groups': groups})
dump(OUT / 'nemenyi_cd.json', cd)


# 5. Bayesian signed-rank test with a region of practical equivalence (Benavoli et al. 2017).
def bayes_signed_rank(z, rope, draws=50_000, seed=BAYES_SEED):
    rng = np.random.default_rng(seed)
    z = np.concatenate([[0.0], np.asarray(z, float)])  # prior pseudo-observation at 0
    s = z[:, None] + z[None, :]
    left, mid, right = (s < -2 * rope), (np.abs(s) <= 2 * rope), (s > 2 * rope)
    w = rng.dirichlet(np.ones(len(z)), size=draws)
    tl = np.einsum('di,ij,dj->d', w, left, w)
    tm = np.einsum('di,ij,dj->d', w, mid, w)
    tr = np.einsum('di,ij,dj->d', w, right, w)
    win = np.argmax(np.stack([tl, tm, tr], 1), 1)
    return {'P_left_better': float((win == 0).mean()), 'P_rope': float((win == 1).mean()), 'P_right_better': float((win == 2).mean())}


bayes = []
for a_, b_ in [('DLG-Aug', 'DLG-Base'), ('DLG-Aug', 'DOMINANT'), ('DLG-Base', 'DOMINANT')]:
    z = [means[d, a_] - means[d, b_] for d in ds2]
    res = bayes_signed_rank(z, 0.01)
    res = {'comparison': f'{a_} - {b_}', 'view': 'S2', 'n': len(z), 'rope_AP': 0.01,
           f'P({a_} better)': res['P_right_better'], 'P(practically equivalent)': res['P_rope'],
           f'P({b_} better)': res['P_left_better'], 'datasets_within_rope': int(sum(abs(x) <= 0.01 for x in z)),
           'median_difference': float(np.median(z))}
    bayes.append(res)
    print(res, flush=True)
dump(OUT / 'bayesian_signed_rank.json', bayes)

# 6. Critical-difference figure for S1 and S2.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(6.8, 3.6))
for ax, rec in zip(axes, [c for c in cd if c['view'] in ('S1', 'S2')]):
    mr = rec['mean_ranks']
    k = rec['k']
    ax.set_xlim(1, k)
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.hlines(0.75, 1, k, color='k', lw=0.8)
    for t in range(1, k + 1):
        ax.vlines(t, 0.72, 0.78, color='k', lw=0.8)
        ax.text(t, 0.84, str(t), ha='center', va='bottom', fontsize=7)
    order = list(mr)
    half = (len(order) + 1) // 2
    for i, m in enumerate(order):
        left = i < half
        y = 0.6 - 0.14 * (i if left else i - half)
        xt = 0.88 if left else k + 0.12  # inverted axis: rank 1 on the right
        ax.plot([mr[m], mr[m]], [0.75, y], color='k', lw=0.6)
        ax.plot([mr[m], xt], [y, y], color='k', lw=0.6)
        ax.text(xt + (-0.05 if left else 0.05), y, f'{m} ({mr[m]:.2f})', fontsize=7,
                ha='left' if left else 'right', va='center')
    for j, g in enumerate(rec['not_separated_groups']):
        lo, hi = min(mr[m] for m in g), max(mr[m] for m in g)
        ax.hlines(0.68 - 0.035 * j, lo - 0.03, hi + 0.03, color='#c0392b', lw=2.2)
    ax.hlines(0.95, 1, 1 + rec['critical_difference'], color='#2c3e50', lw=1.5)
    ax.text(1 + rec['critical_difference'] / 2, 0.97, f"CD = {rec['critical_difference']:.2f}", ha='center', va='bottom', fontsize=7)
    ax.set_title(f"View {rec['view']}: {rec['n']} graphs, {rec['k']} detectors (lower rank = better AP)", fontsize=8)
    ax.axis('off')
fig.tight_layout()
fig.savefig(OUT / 'fig_cd_diagram.pdf')
fig.savefig(OUT / 'fig_cd_diagram.png', dpi=200)
print('done')
