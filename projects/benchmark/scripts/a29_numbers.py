"""Numbers, generated tables and result paragraphs for the A29 manuscript.

Every number printed in an A29 paragraph or table is read here from results/a29
(or from the frozen A28 tables); build_all() is consumed by a29_build_manuscript.py.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from a29_common import EVIDENCE_TAG, LOCAL, REV, RESULTS, ROOT, read_json

A28 = ROOT / 'projects/benchmark/paper/a28/applsci/DLG-Benchmark_A28_AppliedSciences.tex'
FIG = {
    'fig_ap_lift.pdf': RESULTS / 'figures/fig_ap_lift.pdf',
    'fig_support_matrix.pdf': RESULTS / 'figures/fig_support_matrix.pdf',
    'fig_cd_diagram.pdf': RESULTS / 'rank_statistics/fig_cd_diagram.pdf',
    'fig_memory_scaling.pdf': RESULTS / 'scaling_sweep/fig_memory_scaling.pdf',
}
GIB = 2 ** 30
TAG = 'benchmark-a24-supplementary-2026-10-30-r1'


def f4(x):
    return '---' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.4f}'


def f2(x):
    return '---' if x is None else f'{x:.2f}'


def gib(b):
    return '---' if b in (None, '', 0) else f'{float(b) / GIB:.2f}'


def ms(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    return (float(v.mean()), float(v.std(ddof=1))) if len(v) > 1 else ((float(v[0]), 0.0) if len(v) else (None, None))


def pm(vals, d=4):
    m, s = ms(vals)
    return '---' if m is None else f'{m:.{d}f} $\\pm$ {s:.{d}f}'


def table(label, caption, colspec, header, rows, size='\\footnotesize', notes=None, place='htbp'):
    body = ' \\\\\n'.join(' & '.join(r) for r in rows)
    out = (f'\\begin{{table}}[{place}]\\centering{size}\\setlength{{\\tabcolsep}}{{3pt}}\n\\caption{{{caption}}}\n\\label{{{label}}}\n'
           f'\\begin{{tabularx}}{{\\linewidth}}{{{colspec}}}\\toprule\n{" & ".join(header)} \\\\ \\midrule\n{body} \\\\\n\\bottomrule\\end{{tabularx}}')
    if notes:
        out += '\n\\begin{tablenotes}\\footnotesize\n\\item ' + notes + '\n\\end{tablenotes}'
    return out + '\n\\end{table}\n'


def csv_rows(p):
    return list(csv.DictReader(open(p)))


# ---------------------------------------------------------------- static items
KEYWORDS = ('anomaly detection; graph neural networks; fraud detection; cybersecurity; benchmark; '
            'GPU memory; alert budget; reproducibility')
FEATURED = ('Helps fraud and security teams choose a graph anomaly detector in two steps: first check that the detector fits '
            'in GPU memory for the target graph, then check how many true anomalies appear among its top 1--5\\% of alerts.')


def data_availability():
    return ('Aggregate results, analysis scripts, dataset reconciliation tables and reproduction code for the primary benchmark are '
            f'archived under the annotated tag \\nolinkurl{{{TAG}}} (scientific commit \\nolinkurl{{9e7e3a940d7e754b497043cf3959557c82773928}}) at '
            f'\\url{{https://github.com/Sam-7878/dlg_gnn/releases/tag/{TAG}}}. The scripts, pre-registered protocol and aggregate '
            f'results of the extension studies in this version are archived under the annotated tag \\nolinkurl{{{EVIDENCE_TAG}}} at '
            f'\\url{{https://github.com/Sam-7878/dlg_gnn/tree/{EVIDENCE_TAG}}} (directories \\nolinkurl{{projects/benchmark/results/a29}}, '
            f'\\nolinkurl{{projects/benchmark/scripts}} and \\nolinkurl{{projects/benchmark/publication/{REV}}}, which lists the '
            'checksums of the submitted files). '
            'Per-node scores, labels, split masks and node identifiers are retained by the authors and not redistributed. '
            'Per-node arrays are unavailable for 310 earlier runs (275 primary and 35 LANL) whose scripts stored only summary files. '
            'Third-party datasets must be obtained from their providers under their terms; the source address and file hash of each '
            'extension graph are listed in the repository.')


# ---------------------------------------------------------------- primary table (frozen A28 numbers)
def primary_ap():
    from a29_figures import table2
    return table2()


def best_of(vals):
    ok = {m: v for m, v in vals.items() if v is not None}
    m = max(ok, key=ok.get)
    return m, ok[m]


# ---------------------------------------------------------------- A29-12 supervised reference
def supervised():
    agg = read_json(RESULTS / 'supervised_reference/supervised_summary_all.json')
    by = {}
    for r in agg:
        by.setdefault(r['graph'], {})[r['model']] = r
    return by


def supervised_table(real_best):
    prim = primary_ap()
    sup = supervised()
    rows = []
    for g in ['Elliptic', 'DGraphFin', 'Ethereum', 'BSC', 'Polygon', 'LANL-RedTeam', 'Weibo', 'Reddit-Real', 'Tolokers',
              'Amazon-Fraud', 'YelpChi']:
        if g not in sup:
            continue
        a, b = sup[g]['HGB-X'], sup[g]['HGB-X+NeighbourMean']
        if g in prim:
            m, v = best_of(prim[g][1])
            pi = prim[g][0]
        elif g in real_best:
            m, v, pi = real_best[g]
        else:
            continue
        rows.append([g, f4(pi), f'{f4(v)} ({m})', f4(a['AP_mean']), f4(b['AP_mean']), a['fit_population'].replace('validation half', 'val.\\ half').replace('train split', 'train')])
    return table('tab:supervised', 'Supervised reference line versus the best unsupervised detector (mean test AP over five seeds). '
                 'The gradient-boosting model is trained \\emph{with labels} on nodes disjoint from the test set and is never part of '
                 'the unsupervised comparison; it shows how much the labels can be predicted from the same inputs. LANL uses the '
                 'historical input protocol for the unsupervised value. Splits are random, not chronological, which favours the '
                 'supervised model on temporal data such as Elliptic.',
                 '>{\\raggedright\\arraybackslash}Xrrrrl',
                 ['Graph', '$\\pi_{\\mathrm{test}}$', 'Best unsupervised AP', 'HGB on $X$', 'HGB on $X$ + nbr.\\ mean', 'Fit set'], rows)


# ---------------------------------------------------------------- A29-04 soundness (CoLA, OCGNN)
def cola_runs():
    rs = read_json(RESULTS / 'baseline_sanity/cola_runs.json')
    out = {}
    for r in rs:
        out.setdefault((r['graph'], r['variant']), []).append(r)
    return out


def cola_table():
    runs = cola_runs()
    names = {'PyGOD-CoLA-primary': 'PyGOD CoLA, primary settings', 'PyGOD-CoLA-paperlike': 'PyGOD CoLA, paper-like settings',
             'CoLA-RWR-reference': 'Original-design reference'}
    rows = []
    for g in ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn']:
        for v in names:
            rr = [r for r in runs.get((g, v), []) if r.get('status') == 'success']
            if not rr:
                continue
            rows.append([g, names[v], pm([r['pr_auc'] for r in rr]), pm([r['roc_auc'] for r in rr]),
                         f2(np.mean([r['loss_first_last'][1] for r in rr if r.get('loss_first_last')]))])
    return table('tab:cola_sanity', 'CoLA soundness check (five seeds, mean $\\pm$ sample SD; same graphs, seeds and test nodes). '
                 'Paper-like settings: 100 epochs, learning rate $10^{-3}$, scores averaged over 256 rounds. The reference follows the '
                 'original design: four-node random-walk subgraphs, the target node\'s features hidden, 256 scoring rounds. '
                 'Final loss is the training loss in the last epoch (0.693 = chance for this binary task).',
                 'l>{\\raggedright\\arraybackslash}Xccc', ['Graph', 'Variant', 'AP', 'ROC-AUC', 'Final loss'], rows, size='\\scriptsize')


def ocgnn_table():
    rs = read_json(RESULTS / 'baseline_sanity/ocgnn_runs.json')
    rows = []
    for g in ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn']:
        rr = [r for r in rs if r.get('graph') == g and r.get('status') == 'success']
        if not rr:
            continue
        rows.append([g, pm([r['pr_auc'] for r in rr]), pm([r['roc_auc'] for r in rr]),
                     f2(np.mean([r['frac_inside_sphere'] for r in rr])), f'{np.mean([r["dist_cv"] for r in rr]):.3g}',
                     f'{np.mean([r["score_sd"] for r in rr]):.3g}'])
    return table('tab:ocgnn_sanity', 'OCGNN hypersphere diagnostics (primary settings, five seeds). ``Inside\'\' is the share of nodes '
                 'whose embedding lies inside the learned sphere; CV is the coefficient of variation of the squared distance to the '
                 'centre. A share near 1 with a very small spread means the embeddings have collapsed onto the centre, so the '
                 'anomaly scores carry little ranking information.',
                 'lccccc', ['Graph', 'AP', 'ROC-AUC', 'Inside', 'Distance CV', 'Score SD'], rows, size='\\scriptsize')


# ---------------------------------------------------------------- A29-06 GADNR retry
GADNR_CELLS = ['Elliptic', 'Flickr-Syn', 'Yelp-Syn', 'DGraphFin', 'Reddit-Syn']
STATUS_TXT = {'COMPLETED': 'completed', 'GPU_OOM': 'GPU out of memory', 'HOST_OOM': 'host RAM exhausted',
              'NOT_ATTEMPTED': 'not attempted after a time-out', 'CRASH': 'process crashed', 'ERROR': 'error',
              'EXCEEDS_CAP_PROJECTED': 'exceeds 12 h (projected)', 'TIMEOUT_OBSERVED': 'exceeds 12 h (observed)',
              'OTHER_ERROR': 'other error'}


def gadnr():
    out = {}
    for g in GADNR_CELLS:
        p = LOCAL / 'gadnr_retry' / f'{g}__seed42' / 'result.json'
        if p.exists():
            out[g] = read_json(p)
    return out


def gadnr_table():
    rows = []
    for g, r in gadnr().items():
        ep = r.get('epoch_seconds') or []
        rows.append([g, STATUS_TXT.get(r['status'], r['status']), f2((r.get('phases') or {}).get('process_graph_done')),
                     f'{len(ep)}', f2(float(np.mean(ep))) if ep else '---', gib(r.get('cuda_peak_allocated_bytes')),
                     gib(r.get('peak_rss_bytes')), f4((r.get('metrics') or {}).get('pr_auc'))])
    return table('tab:gadnr_retry', 'GADNR retry without the earlier 300 s limit (seed 42, primary settings, RTX 3090, 12 h cap). '
                 'Preprocessing is the Python construction of neighbour lists; GPU peak is the PyTorch allocated peak, which can exceed '
                 'the 24 GB card because the Windows driver lets allocations spill into shared system memory.',
                 'l>{\\raggedright\\arraybackslash}Xrrrrrr',
                 ['Graph', 'Outcome', 'Prep.\\ s', 'Epochs', 's/epoch', 'GPU GiB', 'RAM GiB', 'AP'], rows, size='\\scriptsize')


# ---------------------------------------------------------------- A29-08 real-label extension
REAL = ['Weibo', 'Reddit-Real', 'Tolokers', 'Amazon-Fraud', 'YelpChi']
REAL_MODELS = ['IForest', 'ECOD', 'DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']
GRAPH_MODELS = REAL_MODELS[2:]


def real_runs():
    out = {}
    for g in REAL:
        for m in REAL_MODELS:
            ps = sorted((LOCAL / 'real_label' / g / m).glob('*/result.json'))
            rs = [read_json(p) for p in ps]
            out[(g, m)] = [r for r in rs if r.get('status') == 'success']
            out[(g, m, 'all')] = rs
    return out


def real_mean(runs, g, m, key='pr_auc'):
    rr = runs[(g, m)]
    return float(np.mean([r[key] for r in rr])) if len(rr) == 5 else None


def real_best(runs):
    best = {}
    for g in REAL:
        vals = {m: real_mean(runs, g, m) for m in REAL_MODELS}
        vals = {m: v for m, v in vals.items() if v is not None}
        if not vals:
            continue
        m = max(vals, key=vals.get)
        pi = float(np.mean([r['test_prevalence'] for r in runs[(g, m)]]))
        best[g] = (m, vals[m], pi)
    return best


def real_table(runs):
    rows = []
    short = {'AnomalyDAE': 'AnomDAE', 'DLG-Base': 'Base', 'DLG-Aug': 'Aug', 'DOMINANT': 'DOMIN.'}
    for g in REAL:
        vals = {m: real_mean(runs, g, m) for m in REAL_MODELS}
        if all(v is None for v in vals.values()):
            continue
        some = next(m for m in REAL_MODELS if runs[(g, m)])
        pi = float(np.mean([r['test_prevalence'] for r in runs[(g, some)]]))
        top = max(v for v in vals.values() if v is not None)
        cells = []
        for m in REAL_MODELS:
            v = vals[m]
            if v is None:
                st = ';'.join(sorted({r['status'] for r in runs[(g, m, 'all')]})) or 'not run'
                cells.append('OOM' if 'OOM' in st else 'N/A')
            else:
                cells.append(('\\textbf{%s}' if abs(v - top) < 1e-12 else '%s') % f4(v))
        gm = [vals[m] for m in GRAPH_MODELS if vals[m] is not None]
        fm = [vals[m] for m in ('IForest', 'ECOD') if vals[m] is not None]
        gain = f'{max(gm) - max(fm):+.4f}' if gm and fm else '---'
        rows.append([g, f4(pi)] + cells + [gain])
    head = ['Graph', '$\\pi_{\\mathrm{test}}$', 'IForest', 'ECOD'] + [short.get(m, m) for m in GRAPH_MODELS] + ['Graph gain']
    return ('\\begin{table}[htbp]\\centering\\scriptsize\\setlength{\\tabcolsep}{1.6pt}\n'
            '\\caption{Average Precision on the five graphs with real anomaly labels (extension study; mean test AP over five seeds). '
            'IForest and ECOD use node features only. Graph gain is the best graph detector minus the best feature-only detector. '
            'The best value in each row is in bold. Standard deviations, ROC-AUC and top-1\\% precision are in Appendix Table~\\ref{tab:real_detail}.}\n'
            '\\label{tab:real_label}\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'r' * (len(head) - 1) + '}\\toprule\n'
            + ' & '.join(head) + ' \\\\ \\midrule\n' + ' \\\\\n'.join(' & '.join(r) for r in rows) +
            ' \\\\\n\\bottomrule\\end{tabular*}\n\\end{table}\n')


def real_detail_table(runs):
    rows = []
    for g in REAL:
        for m in REAL_MODELS:
            rr = runs[(g, m)]
            if len(rr) != 5:
                continue
            rows.append([g, m, pm([r['pr_auc'] for r in rr]), pm([r['roc_auc'] for r in rr]), pm([r['P_at_1pct'] for r in rr], 3),
                         pm([r['lift_1pct'] for r in rr], 2)])
    return table('tab:real_detail', 'Real-label extension, all metrics (five seeds, mean $\\pm$ sample SD).', 'llcccc',
                 ['Graph', 'Detector', 'AP', 'ROC-AUC', 'P@1\\%', 'Lift@1\\%'], rows, size='\\scriptsize')


def real_dataset_table():
    d = read_json(RESULTS / 'real_label/DATASETS_A29.json')
    cite = {'Weibo': '\\cite{zhao2020errorbounded, liu2022bond}', 'Reddit-Real': '\\cite{kumar2019jodie, liu2022bond}',
            'Tolokers': '\\cite{platonov2023heterophily}', 'YelpChi': '\\cite{rayana2015collective, dou2020care}',
            'Amazon-Fraud': '\\cite{mcauley2013amateurs, dou2020care}'}
    rows = []
    for g in REAL:
        r = d[g]
        rows.append([g + ' ' + cite[g], f"{r['N']:,}", f"{r['E_directed']:,}", str(r['F']), f"{r['positives']:,}",
                     f"{r['prevalence']:.4f}", r['domain']])
    return table('tab:real_datasets', 'Extension graphs with real anomaly labels and their own edges. Edges are stored in both '
                 'directions without self-loops; features are used as distributed. Amazon-Fraud nodes 0--3,304 carry no label and '
                 'stay in the graph but not in the evaluation. The distributed Weibo file holds 347 positive users (4.1\\%), fewer '
                 'than the 868 listed in the BOND paper, while its edge count (407,963 directed entries before symmetrisation) '
                 'matches BOND; we use the file unchanged.',
                 'lrrrrr>{\\raggedright\\arraybackslash}X', ['Graph', 'Nodes', 'Edges', 'Feat.', 'Positives', 'Prevalence', 'Domain / label'],
                 rows, size='\\scriptsize')


# ---------------------------------------------------------------- A29-04 equal-budget tuning
TUNE_GRAPHS = ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn', 'Amazon-Syn', 'BSC', 'Polygon']
MODELS7 = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']


def tuning():
    out = {}
    for g in TUNE_GRAPHS:
        for m in MODELS7:
            sel = LOCAL / 'baseline_sanity/tune' / g / m / 'selection.json'
            rs = [read_json(p) for p in sorted((LOCAL / 'baseline_sanity/tune' / g / m / 'selected').glob('*/result.json'))]
            rs = [r for r in rs if r.get('status') == 'success']
            if sel.exists() and len(rs) == 5:
                out[(g, m)] = {'config': read_json(sel)['selected']['config'], 'ap': float(np.mean([r['pr_auc'] for r in rs])),
                               'device': rs[0].get('gpu_name')}
    return out


def tuning_failures():
    """Search configurations that did not finish (recorded, then skipped by the validation-AP selection)."""
    out = []
    for g in TUNE_GRAPHS:
        for m in MODELS7:
            sel = LOCAL / 'baseline_sanity/tune' / g / m / 'selection.json'
            if sel.exists():
                out += [(g, m, t['config'], t['status']) for t in read_json(sel)['trials'] if t['status'] != 'success']
    return out


def tuning_stats(tu):
    from scipy.stats import rankdata, chi2
    prim = primary_ap()
    graphs = [g for g in TUNE_GRAPHS if all((g, m) in tu for m in MODELS7)]
    res = {'graphs': graphs}
    if len(graphs) < 3:
        return res
    for label, ms in (('all7', MODELS7), ('gram3', ['DOMINANT', 'DLG-Base', 'DLG-Aug'])):
        a = np.array([[tu[(g, m)]['ap'] for m in ms] for g in graphs])
        r = rankdata(-a, axis=1)
        n, k = r.shape
        q = 12 / (n * k * (k + 1)) * (r.sum(0) ** 2).sum() - 3 * n * (k + 1)
        res[label] = {'mean_ranks': dict(zip(ms, r.mean(0).round(3).tolist())), 'Q': float(q), 'W': float(q / (n * (k - 1))),
                      'p_chi2': float(chi2.sf(q, k - 1))}
    deltas = {m: [tu[(g, m)]['ap'] - prim[g][1][m] for g in graphs if prim[g][1][m] is not None] for m in MODELS7}
    res['improved'] = {m: int(sum(d > 0.005 for d in v)) for m, v in deltas.items()}
    res['worse'] = {m: int(sum(d < -0.005 for d in v)) for m, v in deltas.items()}
    res['n_cells'] = sum(len(v) for v in deltas.values())
    return res


def tuning_table(tu):
    prim = primary_ap()
    rows = []
    for g in TUNE_GRAPHS:
        for m in MODELS7:
            if (g, m) not in tu:
                continue
            c = tu[(g, m)]['config']
            cfg = f"h{c['hid_dim']}, lr {c['lr']:g}, " + (f"$\\beta$ {c['beta']:g}" if 'beta' in c else f"{c['epoch']} ep")
            p = prim[g][1][m]
            rows.append([g, m, f4(p), f4(tu[(g, m)]['ap']), f'{tu[(g, m)]["ap"] - p:+.4f}' if p is not None else '---', cfg])
    return table('tab:tuning', 'Equal-budget tuning on six small graphs (12 configurations per detector chosen by seed-42 '
                 'validation AP; selected configuration then run on seeds 42--46 and evaluated on test once). Default AP is the '
                 'frozen primary value. Validation labels are used for selection, so this is semi-supervised model selection and '
                 'is not mixed with the unsupervised primary results.',
                 'll' + 'r' * 3 + '>{\\raggedright\\arraybackslash}X', ['Graph', 'Detector', 'Default AP', 'Tuned AP', '$\\Delta$', 'Selected'],
                 rows, size='\\scriptsize')


# ---------------------------------------------------------------- pending markers (draft builds only)
PENDING = []
PREVIEW = False  # True: render the real-label story from incomplete results (inspection only)


def pending(name):
    PENDING.append(name)
    return '\\textbf{[PENDING: ' + name.replace('_', '\\_') + ']}'


def p3(p):
    """p-value in plain notation for running text."""
    if p is None:
        return '---'
    if p < 1e-3:
        m, e = f'{p:.1e}'.split('e')
        return f'{m} \\times 10^{{{int(e)}}}'
    return f'{p:.2f}' if p >= 0.01 else f'{p:.3f}'


def oxford(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else (' and '.join(xs) if len(xs) == 2 else ', '.join(xs[:-1]) + ' and ' + xs[-1])


# ---------------------------------------------------------------- A29-07 rank statistics
def rank_rows():
    return csv_rows(RESULTS / 'rank_statistics/rank_sensitivity.csv')


def rank_numbers():
    rr = {r['analysis']: r for r in rank_rows()}
    bay = {r['comparison']: r for r in read_json(RESULTS / 'rank_statistics/bayesian_signed_rank.json')}
    pe = lambda r: float(r['p_exact']) if r['p_exact'] else float(r['p_MC'])
    ab, ad, bd = bay['DLG-Aug - DLG-Base'], bay['DLG-Aug - DOMINANT'], bay['DLG-Base - DOMINANT']
    return {'GRAM3_P_S1': p3(pe(rr['S1_gram_models_only'])), 'GRAM3_P_S2': p3(pe(rr['S2_gram_models_only'])),
            'GRAM3_P_S4': p3(pe(rr['S4_gram_models_only'])),
            'BAYES_AUG_BASE': f"{ab['P(practically equivalent)']:.2f}", 'BAYES_BASE_DOM': f"{bd['P(practically equivalent)']:.2f}",
            'BAYES_AUG_DOM_ROPE': f"{ad['P(practically equivalent)']:.2f}", 'BAYES_AUG_DOM_AUG': f"{ad['P(DLG-Aug better)']:.2f}",
            'BAYES_AUG_DOM_DOM': f"{ad['P(DOMINANT better)']:.2f}"}


def friedman_a28():
    tex = A28.read_text()
    i = tex.index('\\label{tab:friedman_stats}')
    block = tex[i:tex.index('\\end{table}', i)]
    out = {}
    for m in re.finditer(r'^(S[1-4]) & (\d+) & (\d+) & ([\d.]+) & ([\d.]+) & \d+ & [^&]+ & ([^&]+) &', block, re.M):
        out[m.group(1)] = {'n': int(m.group(2)), 'k': int(m.group(3)), 'Q': float(m.group(4)), 'W': float(m.group(5)),
                           'p_exact': float(m.group(6))}
    assert set(out) == {'S1', 'S2', 'S3', 'S4'}, out
    return out


def friedman_compact_table():
    fr = friedman_a28()
    cd = {r['view']: r for r in read_json(RESULTS / 'rank_statistics/nemenyi_cd.json')}
    desc = {'S1': 'All seven detectors ran', 'S2': 'All 13 graphs, five detectors', 'S3': 'Finance and blockchain graphs',
            'S4': 'Injected-anomaly graphs'}
    rows = []
    for v in ('S1', 'S2', 'S3', 'S4'):
        r = fr[v]
        top = ''
        if v in cd:
            mr = cd[v]['mean_ranks']
            best = sorted(mr.items(), key=lambda t: t[1])[:3]
            top = ', '.join(f'{m} {x:.2f}' for m, x in best)
        rows.append([v, desc[v], str(r['n']), str(r['k']), f"{r['Q']:.2f}", f"{r['W']:.2f}", f"${p3(r['p_exact'])}$", top or '---'])
    return table('tab:friedman_compact', 'Friedman tests of detector ranks across graphs. $n$: graphs; $k$: detectors; $W$: Kendall\'s '
                 'agreement (0--1); exact $p$ from full enumeration. The last column lists the three best average ranks '
                 '(1 = best) where the Nemenyi analysis was run.',
                 'l>{\\raggedright\\arraybackslash}Xrrrrr>{\\raggedright\\arraybackslash}X',
                 ['View', 'Graphs', '$n$', '$k$', '$Q$', '$W$', 'Exact $p$', 'Best average ranks'], rows, size='\\scriptsize')


def rank_sensitivity_table():
    names = {'S2_full': 'S2 as published', 'S2_minus_Elliptic_DGraphFin': 'S2 without Elliptic and DGraphFin',
             'S2_epsilon_0.005': 'S2, AP within 0.005 tied', 'S2_epsilon_0.01': 'S2, AP within 0.01 tied',
             'S1_gram_models_only': 'S1, top three only', 'S2_gram_models_only': 'S2, top three only',
             'S4_gram_models_only': 'S4, top three only'}
    rows = []
    for r in rank_rows():
        p = r['p_exact'] or r['p_MC']
        tag = '' if r['p_exact'] else '$^\\dagger$'
        rows.append([names.get(r['analysis'], r['analysis']), r['n'], r['k'], f"{float(r['Q']):.2f}", f"{float(r['Kendall_W']):.2f}",
                     f"${p3(float(p))}${tag}"] + [f"{float(r['mean_rank_' + m]):.2f}" for m in ('DOMINANT', 'DLG-Base', 'DLG-Aug')])
    return table('tab:rank_sensitivity', 'Sensitivity of the rank analysis (mean ranks, 1 = best). ``Tied\'\' rounds AP to the given grid '
                 'so that near-equal values share an average rank. $^\\dagger$ Monte Carlo $p$ ($10^6$ permutations) where ties '
                 'prevent exact enumeration.', '>{\\raggedright\\arraybackslash}Xrrrrrrrr',
                 ['Analysis', '$n$', '$k$', '$Q$', '$W$', '$p$', 'DOMINANT', 'DLG-Base', 'DLG-Aug'], rows, size='\\scriptsize')


# ---------------------------------------------------------------- A29-04 texts
def cola_text():
    runs = cola_runs()
    get = lambda g, v, k: float(np.mean([r[k] for r in runs.get((g, v), []) if r.get('status') == 'success']))
    P, R = 'PyGOD-CoLA-primary', 'CoLA-RWR-reference'
    losses = [r['loss_first_last'][1] for (g, v), rr in runs.items() if v == P for r in rr if r.get('loss_first_last')]
    return (f'Its training loss fell to {min(losses):.2f}--{max(losses):.2f} (0.69 would be chance), so the task it learns is easy, '
            f'yet its ranking stayed close to random on Cora-Syn and CiteSeer-Syn (ROC-AUC {get("Cora-Syn", P, "roc_auc"):.2f} and '
            f'{get("CiteSeer-Syn", P, "roc_auc"):.2f}; Table~\\ref{{tab:cola_sanity}}). Longer training with scores averaged over 256 '
            f'rounds did not change this. A compact re-implementation of the original design reached AP {get("Cora-Syn", R, "pr_auc"):.4f} '
            f'(ROC-AUC {get("Cora-Syn", R, "roc_auc"):.2f}) on Cora-Syn and {get("CiteSeer-Syn", R, "pr_auc"):.4f} '
            f'({get("CiteSeer-Syn", R, "roc_auc"):.2f}) on CiteSeer-Syn, against {get("Cora-Syn", P, "pr_auc"):.4f} and '
            f'{get("CiteSeer-Syn", P, "pr_auc"):.4f} for the library version. On PubMed-Syn the AP of the two was similar '
            f'({get("PubMed-Syn", R, "pr_auc"):.4f} and {get("PubMed-Syn", P, "pr_auc"):.4f}), but the ROC-AUC of the reference was '
            f'higher ({get("PubMed-Syn", R, "roc_auc"):.2f} against {get("PubMed-Syn", P, "roc_auc"):.2f}).')


def ocgnn_text():
    rs = [r for r in read_json(RESULTS / 'baseline_sanity/ocgnn_runs.json') if r.get('status') == 'success']
    by = {}
    for r in rs:
        by.setdefault(r['graph'], []).append(r)
    inside = [float(np.mean([r['frac_inside_sphere'] for r in v])) for v in by.values()]
    cv = [float(np.mean([r['dist_cv'] for r in v])) for v in by.values()]
    return (f'With primary settings, {min(inside) * 100:.0f}--{max(inside) * 100:.0f}\\% of nodes lay inside the learned sphere on '
            f'Cora-, CiteSeer- and PubMed-Syn, and the distances to its centre varied by at most {max(cv) * 100:.1f}\\% '
            f'(coefficient of variation; Table~\\ref{{tab:ocgnn_sanity}}): the node embeddings had collapsed onto the centre.')


def tune_text(tu):
    st = tuning_stats(tu)
    if len(st['graphs']) < len(TUNE_GRAPHS):
        return pending(f"TUNE_TEXT ({len(st['graphs'])}/{len(TUNE_GRAPHS)} graphs complete)")
    imp, wor, n = sum(st['improved'].values()), sum(st['worse'].values()), st['n_cells']
    g3, a7 = st['gram3'], st['all7']
    best7 = sorted(a7['mean_ranks'].items(), key=lambda t: t[1])
    return (f'Tuning raised AP by more than 0.005 in {imp} of {n} detector--graph cells and lowered it by more than 0.005 in {wor}; '
            'with small validation sets the chosen configuration does not always carry over to the test nodes. After tuning, the '
            f'three reconstruction models still could not be separated (Friedman $p = {p3(g3["p_chi2"])}$, $\\chi^2$ approximation), '
            f'and the best average ranks over all seven detectors were {oxford(f"{m} ({x:.2f})" for m, x in best7[:3])}.'
            + fail_sentence() + rerun_sentence())


def fail_sentence():
    f = tuning_failures()
    if not f:
        return ''
    by = {}
    for g, m, c, st in f:
        by.setdefault((m, STATUS_TXT.get(st, st)), []).append((g, c))
    parts = [f'{len(v)} {m} configuration{"s" if len(v) > 1 else ""} on {oxford(sorted(set(g for g, _ in v)))} '
             f'({st}; hidden size {oxford(sorted(set(str(c["hid_dim"]) for _, c in v)))})' for (m, st), v in sorted(by.items())]
    return (' Of the ' + f'{sum(len(read_json(p)["trials"]) for p in (LOCAL / "baseline_sanity/tune").glob("*/*/selection.json")):,}'
            + ' search runs, ' + oxford(parts) + ' did not finish; they are recorded and were not eligible for selection.')


def rerun_sentence():
    """Runs whose first attempt failed while two memory-spilling jobs ran at once (records kept as attempt1_*), and how the
    repeat run, made with one GPU job at a time, ended."""
    root = LOCAL / 'baseline_sanity/tune'
    first = sorted(p for p in root.rglob('result.json') if any(x.startswith('attempt1_concurrent_spill_') for x in p.parts))
    if not first:
        return ''
    graphs = sorted({p.relative_to(root).parts[0] for p in first})
    again = []
    for p in first:
        rel = [x for x in p.relative_to(root).parts if not x.startswith('attempt1_concurrent_spill_')]
        st = read_json(root.joinpath(*rel))['status']  # the repeat must exist once the tuning text is rendered
        if st != 'success':
            again.append((rel[0], st))
    txt = (f' The first attempt at {len(first)} GADNR runs on {oxford(graphs)} ran out of memory while a second job was also '
           'spilling into shared system memory, so these runs were repeated with one GPU job at a time')
    if not again:
        return txt + '; all of them then finished, and the first attempts are kept in the evidence record.'
    ag = sorted({g for g, _ in again})
    return (txt + f'. {NUMWORD.get(len(first) - len(again), len(first) - len(again)).capitalize()} of them then finished; the other '
            f'{NUMWORD.get(len(again), len(again))}, on {oxford(ag)}, ran out of memory again when running alone and {"is" if len(again) == 1 else "are"} among the '
            f'unfinished runs above. The first attempts are kept in the evidence record.')


# ---------------------------------------------------------------- A29-06 GADNR texts
def gadnr_text():
    g = gadnr()
    if len(g) < len(GADNR_CELLS):
        return pending(f'GADNR_RETRY_TEXT ({len(g)}/{len(GADNR_CELLS)} cells)')
    done = [(n, r) for n, r in g.items() if r['status'] == 'COMPLETED']
    sent = []
    for name, r in done:
        m = r.get('metrics') or {}
        spill = (r.get('cuda_peak_reserved_bytes') or 0) > 24 * GIB  # same rule as spilled()
        sent.append(f'On {name} it completed in {(r.get("fit_seconds") or 0) / 60:.0f} minutes with AP {f4(m.get("pr_auc"))} '
                    f'($\\pi_{{\\mathrm{{test}}}} = {f4(m.get("test_prevalence"))}$)'
                    + (f', but only by spilling past the 24 GiB card ({gib(r.get("cuda_peak_reserved_bytes"))} GiB reserved)' if spill else '') + '.')
    for name, r in g.items():
        if r['status'] != 'COMPLETED':
            sent.append(f'On {name} ' + gadnr_stop(r) + '.')
    head = (f'Without the earlier 300 s limit and with a 12 h cap per graph (seed 42), GADNR completed on '
            f'{NUMWORD[len(done)] if done else "none"} of the {NUMWORD[len(g)]} graphs. GPU memory in this paragraph is the '
            'reserved peak, which includes memory held by the allocator; Table~\\ref{tab:gadnr_retry} lists the allocated peak. ')
    return head + ' '.join(sent) + gadnr_seed_sentence()


def gadnr_stop(r):
    """Why a GADNR retry cell did not finish: status, the stage it was in, and the memory it held at that point."""
    st, ph = r['status'], r.get('phase') or ''
    cpu = ph in ('load_input', 'process_graph')  # GADNR builds its neighbourhood structures on the CPU before training
    rss, gpu = (r.get('peak_rss_bytes') or 0) / 1e9, (r.get('cuda_peak_reserved_bytes') or 0) / GIB
    prep = ((r.get('phases') or {}).get('process_graph_done') or 0) / 3600
    if cpu:
        where = 'while still preparing the graph on the CPU, before training began'
    else:
        where = ('in the first training epoch' if not r.get('epoch_seconds') else 'during training') + \
                (f', after {prep:.1f} h of preparing the graph on the CPU' if prep >= 0.5 else '')
    if st == 'GPU_OOM':
        return f'it ran out of GPU memory {where} ({gpu:.1f} GiB reserved, beyond the 24 GiB card)'
    cap = r.get('host_memory_cap')
    if st == 'HOST_OOM' and cap:
        return (f'the host ran out of RAM {where}: an uncapped first attempt filled the memory of the machine within minutes and made it '
                f'unresponsive, and a repeat limited to {cap["MemoryMax"].rstrip("G")} GiB of RAM (plus {cap["MemorySwapMax"].rstrip("G")} GiB swap) '
                'was stopped by the kernel when it reached that limit')
    if st == 'HOST_OOM':
        return f'the host ran out of RAM {where} (peak {rss:.1f} GB of about 22 GB)'
    if st == 'TIMEOUT_OBSERVED':
        return (f'it reached the 12 h cap {where}' + (f' (host RAM {rss:.1f} GB, with swapping)' if cpu and rss > 15 else '')
                + ('' if cpu else f' ({gpu:.1f} GiB GPU memory reserved)'))
    err = (r.get('error') or '').lower()
    if 'illegal memory access' in err:  # observed message only; the cause (e.g. spill exhaustion or a kernel fault) is not established
        return (f'the GPU reported an illegal memory access {where} ({gpu:.1f} GiB reserved'
                + (', beyond the 24 GiB card' if gpu > 24 else '') + '; the cause was not established)')
    return f'it stopped with {STATUS_TXT.get(st, st)} {where}'


def gadnr_seed_sentence(g='Elliptic'):
    """Seeds 43-46 on the one large graph where seed 42 completed; earlier attempts (attempt1_*) are reported, not dropped."""
    base = LOCAL / 'gadnr_retry'
    first = read_json(base / f'{g}__seed42' / 'result.json') if (base / f'{g}__seed42' / 'result.json').exists() else {}
    if first.get('status') != 'COMPLETED':
        return ''
    recs = {s: base / f'{g}__seed{s}' / 'result.json' for s in (43, 44, 45, 46)}
    if not all(p.exists() for p in recs.values()):
        return ' ' + pending(f'GADNR {g} seeds 43--46')
    rs = {s: read_json(p) for s, p in recs.items()}
    ok = [s for s, r in rs.items() if r['status'] == 'COMPLETED']
    aps = [first['metrics']['pr_auc']] + [rs[s]['metrics']['pr_auc'] for s in ok]
    earlier = sorted(s for s in recs if any((base / f'{g}__seed{s}').glob('attempt1_*/result.json')))
    txt = f' On {g} we also ran seeds 43--46: '
    txt += (('all four' if len(ok) == 4 else NUMWORD[len(ok)]) + ' completed' + (f' (AP over the {NUMWORD[len(aps)]} completed seeds {np.mean(aps):.4f} $\\pm$ {np.std(aps, ddof=1):.4f})' if len(aps) > 1 else '')
            if ok else 'none completed')
    bad = [s for s in rs if s not in ok]
    if bad:
        txt += '; seed' + ('s ' if len(bad) > 1 else ' ') + oxford(str(s) for s in bad) + ' stopped with ' + oxford(sorted(set(STATUS_TXT.get(rs[s]['status'], rs[s]['status']) for s in bad)))
    txt += '.'
    if earlier:
        txt += (f' A first attempt at seed{"s" if len(earlier) > 1 else ""} {oxford(str(s) for s in earlier)} ran out of GPU memory, '
                'while a second memory-heavy job was running at the same time; they were repeated with one GPU job at a time, and the '
                'repeats are reported.')
    return txt


# ---------------------------------------------------------------- A29-05a/b/c
def dvg():
    out = {}
    for p in sorted((RESULTS / 'dense_vs_gram').glob('*.json')):
        out[p.stem] = read_json(p)
    return out


DVG_ORDER = ['Cora-Syn', 'CiteSeer-Syn', 'Polygon', 'BSC', 'BitcoinOTC', 'Ethereum', 'Amazon-Syn', 'PubMed-Syn', 'Elliptic']
def rng(a, b, d):
    x, y = f'{a:.{d}f}', f'{b:.{d}f}'
    return x if x == y else f'{x}--{y}'


def sci_tex(x):
    if x == 0:
        return '0'
    m, e = f'{x:.0e}'.split('e')
    return f'${m} \\times 10^{{{int(e)}}}$'


def spilled(rec):
    return (rec.get('peak_reserved_bytes') or 0) > 24 * GIB


DENSE_TXT = {'COMPLETED': None, 'GPU_OOM': 'GPU OOM', 'HOST_OOM': 'host OOM', 'NOT_ATTEMPTED_ARITHMETIC': 'too large'}


def dense_vs_gram_table():
    d = dvg()
    rows = []
    for g in DVG_ORDER:
        if g not in d:
            continue
        for fam in ('DOMINANT', 'DLG-Base'):
            r = d[g]['families'].get(fam, {})
            c, nf = r.get('denseA_vs_gram'), r.get('noise_floor_denseA_vs_denseB')
            dA = r.get('dense_A') or {}
            dense_mem = (gib(dA.get('peak_allocated_bytes')) + ('$^\\ast$' if spilled(dA) else '')) if c else DENSE_TXT.get(r.get('dense_status'), r.get('dense_status'))
            sci = lambda x: '---' if x is None else (f'{x:.0e}' if x else '0')
            rows.append([g if fam == 'DOMINANT' else '', f"{d[g]['N']:,}" if fam == 'DOMINANT' else '', fam,
                         sci(c and c['max_rel_loss_diff']), sci(nf and nf['max_rel_loss_diff']),
                         f"{1 - c['score_spearman']:.0e}" if c else '---', f"{c['top1pct_overlap_test']:.2f}" if c else '---',
                         f"{c['delta_AP']:+.1e}" if c else '---', dense_mem, gib((r.get('gram') or {}).get('peak_allocated_bytes'))])
    if not rows:
        return pending('dense_vs_gram table')
    return table('tab:dense_vs_gram', 'Dense PyGOD loss versus the memory-saving loss on nine benchmark graphs (seed 42, 50 epochs, identical '
                 'initial weights). Loss diff.: largest relative difference in the training loss over all epochs; noise: the same '
                 'quantity for two dense runs from the same start, i.e. what GPU arithmetic alone produces. $1-\\rho$: one minus the '
                 'Spearman correlation of the final anomaly scores (0 = identical ranking); Top-1\\%: overlap of the top-1\\% test alert lists. Memory is the peak allocated '
                 'GPU memory in GiB; $^\\ast$ the run exceeded the 24 GB card and completed only by spilling into system '
                 'memory; ``GPU OOM\'\' means it failed even then.',
                 'lrlrrrrrrr', ['Graph', '$N$', 'Model', 'Loss diff.', 'Noise', '$1-\\rho$', 'Top-1\\%', '$\\Delta$AP', 'Dense GiB',
                                'Saving GiB'], rows, size='\\scriptsize')


def dvg_numbers():
    d = dvg()
    comp = [(g, f, r) for g in d for f, r in d[g]['families'].items() if r.get('denseA_vs_gram')]
    if len(d) < len(DVG_ORDER):
        return {'DVG_TEXT': pending(f'DVG_TEXT ({len(d)}/{len(DVG_ORDER)} graphs)'), 'DVG_NG': pending('DVG_NG'),
                'DVG_MINRHO': pending('DVG_MINRHO')}
    gs = sorted({g for g, _, _ in comp}, key=DVG_ORDER.index)
    mx = max(r['denseA_vs_gram']['max_rel_loss_diff'] for _, _, r in comp)
    nf = max(r['noise_floor_denseA_vs_denseB']['max_rel_loss_diff'] for _, _, r in comp)
    rho = min(r['denseA_vs_gram']['score_spearman'] for _, _, r in comp)
    top1 = min(r['denseA_vs_gram']['top1pct_overlap_test'] for _, _, r in comp)
    dap = max(abs(r['denseA_vs_gram']['delta_AP']) for _, _, r in comp)
    ftxt = ''
    for g in DVG_ORDER:
        if g not in d:
            continue
        fam = d[g]['families']
        hard = {f: r for f, r in fam.items() if not r.get('denseA_vs_gram') or spilled(r.get('dense_A') or {})}
        if not hard:
            continue
        bits = []
        for f, r in hard.items():
            gm = r.get('gram') or {}
            if not r.get('denseA_vs_gram'):
                bits.append(f'the dense {f} ran out of memory even after spilling past the 24 GB card into system memory')
            else:
                dA = r['dense_A']
                bits.append(f'the dense {f} finished only by spilling (peak {dA["peak_allocated_bytes"] / GIB:.1f} GiB, '
                            f'{dA["fit_seconds"]:.0f} s for 50 epochs, against {gm["peak_allocated_bytes"] / GIB:.2f} GiB and '
                            f'{gm["fit_seconds"]:.1f} s for the memory-saving loss)')
        ftxt += f' On {g} ({d[g]["N"]:,} nodes) ' + oxford(bits) + '.'
    worst = max(comp, key=lambda t: abs(t[2]['denseA_vs_gram']['delta_AP']))
    rest = max(abs(r['denseA_vs_gram']['delta_AP']) for g, f, r in comp if (g, f) != worst[:2])
    txt = (f'For DOMINANT and DLG-Base we trained the dense PyGOD code and our implementation from the same initial weights for 50 '
           f'epochs on {len(gs)} benchmark graphs (Table~\\ref{{tab:dense_vs_gram}}). To know how much difference GPU arithmetic alone '
           'produces, we also trained the dense model twice from the same start. The training losses of the two implementations '
           f'differed by at most {sci_tex(mx)} in relative terms (two dense runs: up to {sci_tex(nf)}), and the rank correlation of '
           f'the final anomaly scores differed from 1 by at most {sci_tex(1 - rho)}. '
           + ('The top-1\\% and top-5\\% alert lists were identical on every graph. ' if top1 == 1 and min(r['denseA_vs_gram']['top5pct_overlap_test'] for _, _, r in comp) == 1 else
              f'The top-1\\% alert lists overlapped by at least {top1 * 100:.0f}\\%. ')
           + f'AP changed by at most {sci_tex(rest)}, except on {worst[0]} ({worst[1]}), where it changed by '
           f'{sci_tex(abs(worst[2]["denseA_vs_gram"]["delta_AP"]))} because neighbouring nodes whose scores agreed to about seven '
           'significant digits swapped places; this is the rounding level of 32-bit arithmetic.' + ftxt)
    return {'DVG_TEXT': txt, 'DVG_NG': str(len(gs)), 'DVG_MINRHO': f'{rho:.4f}', 'DVG_1MRHO': sci_tex(1 - rho)}


def scaling():
    p = RESULTS / 'scaling_sweep/scaling_slopes.json'
    pts = RESULTS / 'scaling_sweep/scaling_points.csv'
    if not p.exists() or not pts.exists():
        return None
    return read_json(p), csv_rows(pts)


def scaling_numbers():
    s = scaling()
    if s is None:
        return {'SCALING_TEXT': pending('SCALING_TEXT'), 'SCALE_DENSE_MAXN': pending('SCALE_DENSE_MAXN'),
                'SCALE_GRAM_SLOPES': pending('SCALE_GRAM_SLOPES')}
    slopes, pts = s
    ok = lambda r: r['status'] == 'COMPLETED' and r.get('exceeds_physical_vram') not in ('True', True)
    by = {}
    for r in pts:
        by.setdefault(r['model'], []).append(r)
    dense = by.get('DOMINANT-dense', [])
    dmax = max((int(r['N']) for r in dense if ok(r)), default=None)
    dfail = next((r for r in dense if not ok(r)), None)
    gram = [m for m in ('DOMINANT-Gram', 'DLG-Base-Gram', 'DLG-Aug-Gram') if m in slopes]
    gmax = {m: max(int(r['N']) for r in by[m] if ok(r)) for m in gram}
    gmem = max(float(r['peak_allocated_bytes']) for m in gram for r in by[m] if ok(r) and int(r['N']) == gmax[m])
    sl = [slopes[m] for m in gram]
    rb = by.get('AnomalyDAE-rowblock', [])
    rbmax = max((int(r['N']) for r in rb if ok(r)), default=None)
    def local(m):  # growth exponent between the two largest completed sizes
        q = sorted((int(r['N']), float(r['peak_allocated_bytes'])) for r in by[m] if ok(r))[-2:]
        return np.log(q[1][1] / q[0][1]) / np.log(q[1][0] / q[0][0])
    loc = [local(m) for m in gram]
    rbt = sorted((int(r['N']), float(r['mean_epoch_seconds'])) for r in rb if ok(r) and r.get('mean_epoch_seconds'))
    tex = np.log(rbt[-1][1] / rbt[1][1]) / np.log(rbt[-1][0] / rbt[1][0]) if len(rbt) > 2 else float('nan')
    dense_txt = (f'The dense DOMINANT grew with an exponent of {slopes["DOMINANT-dense"]:.2f}, close to the $N^2$ of its '
                 f'$N \\times N$ tables, and its largest size that fitted on the card was $N = {dmax:,}$'
                 + (f' (at $N = {int(dfail["N"]):,}$ it {"ran out of GPU memory" if dfail["status"] == "GPU_OOM" else ("needed more than the 24 GB card" if dfail.get("exceeds_physical_vram") in ("True", True) else "was not attempted because its dense table alone exceeds the host-memory budget")})' if dfail else '')
                 + '. ') if 'DOMINANT-dense' in slopes and dmax else pending('dense DOMINANT sweep') + ' '
    txt = ('Figure~\\ref{fig:scaling} measures peak GPU memory on random graphs of growing size. ' + dense_txt +
           f'The three memory-saving models completed every size up to $N = {min(gmax.values()):,}$ using at most '
           f'{gmem / GIB:.2f} GiB; between the two largest sizes their memory grew by a factor of '
           f'{rng(2 ** min(loc), 2 ** max(loc), 1)} when $N$ doubled, i.e. roughly in proportion to graph size (over the whole '
           f'range the fitted exponent is {rng(min(sl), max(sl), 2)}, because fixed overheads dominate small graphs). '
           + (f'The row-block AnomalyDAE also stayed within {max(float(r["peak_allocated_bytes"]) for r in rb if ok(r)) / GIB:.1f} GiB '
              f'up to $N = {rbmax:,}$, but its time per epoch grew much faster, with an exponent of {tex:.1f} '
              f'({rbt[-1][1]:.0f} s per epoch at $N = {rbt[-1][0]:,}$), approaching the $N^2$ growth of its arithmetic.'
              if rbmax else ''))
    return {'SCALING_TEXT': txt, 'SCALE_DENSE_MAXN': f'{dmax:,}' if dmax else '---',
            'SCALE_GRAM_SLOPES': f'{min(sl):.2f}--{max(sl):.2f}', 'SCALE_GRAM_LOCAL': rng(2 ** min(loc), 2 ** max(loc), 1)}


RES_GRAPHS = ['Cora-Syn', 'Elliptic', 'DGraphFin', 'Reddit-Syn']
RES_MODELS = ['DOMINANT', 'DLG-Base', 'DLG-Aug', 'OCGNN', 'CoLA', 'AnomalyDAE', 'GADNR']
RES_STATUS = {'GPU_OOM': 'OOM', 'TIMEOUT_OBSERVED': '$>$3 h$^\\dagger$', 'HOST_OOM': 'RAM OOM', 'ERROR': 'error', 'NOT_RUN': 'n/r',
              'EXCEEDS_CAP_PROJECTED': '$>$12 h', 'OTHER_ERROR': 'error'}


def resource():
    p = RESULTS / 'resource_matrix/resource_matrix.json'
    return read_json(p) if p.exists() else None


def resource_table():
    rs = resource()
    if rs is None:
        return pending('resource table')
    by = {(r['graph'], r['model']): r for r in rs}
    rows = []
    for m in RES_MODELS:
        cells = [m]
        for g in RES_GRAPHS:
            r = by.get((g, m), {'status': 'NOT_RUN'})
            if r['status'] in ('success', 'COMPLETED'):
                fit = r.get('fit_seconds')
                spill = r.get('exceeds_physical_vram') or (r.get('peak_reserved_bytes') or 0) > 24 * GIB
                cells.append(f"{fit:,.0f} s / {gib(r.get('peak_allocated_bytes'))}" + ('$^\\ast$' if spill else '') if fit is not None else '---')
            else:
                cells.append(RES_STATUS.get(r['status'], r['status']))
        rows.append(cells)
    head = ['Detector'] + [f'{g}' for g in RES_GRAPHS]
    return table('tab:resource', 'Measured training time and peak GPU memory (fit seconds / allocated GiB; seed 42, primary settings '
                 'and full epoch budget, RTX 3090, nothing else running). These measurements replace the estimated values of the '
                 'previous version. Graph sizes (nodes / edges): Cora-Syn 2,708 / 11,276; Elliptic 46,564 / 83,188; DGraphFin 1,225,601 / 746,271; '
                 'Reddit-Syn 232,965 / 114,869,737. '
                 'GADNR values on the three larger graphs come from the retry in Table~\\ref{tab:gadnr_retry} (12 h cap); other cells '
                 'had a 3 h cap. $^\\ast$ Reserved memory exceeded the 24 GB card (spilling into system memory). $^\\dagger$ Cap of this '
                 'measurement only; the primary runs, which had no time cap, completed (Table~\\ref{tab:pr_auc_main}).',
                 'l' + '>{\\raggedleft\\arraybackslash}X' * len(RES_GRAPHS), head, rows, size='\\scriptsize')


def resource_text():
    rs = resource()
    if rs is None:
        return pending('RESOURCE_TEXT')
    by = {(r['graph'], r['model']): r for r in rs}
    if any(r['status'] == 'NOT_RUN' for r in rs) and not PREVIEW:
        return pending('RESOURCE_TEXT (GADNR retry cells not yet run)')
    ok = lambda r: r['status'] in ('success', 'COMPLETED')
    span = lambda g: [by[(g, m)] for m in RES_MODELS if ok(by[(g, m)]) and m != 'GADNR']
    rd, dg = span('Reddit-Syn'), span('DGraphFin')
    fit = lambda rr: rng(min(r['fit_seconds'] for r in rr), max(r['fit_seconds'] for r in rr), 0)
    mem = lambda rr: rng(min(r['peak_allocated_bytes'] for r in rr) / GIB, max(r['peak_allocated_bytes'] for r in rr) / GIB, 1)
    txt = ('Table~\\ref{tab:resource} lists the measured training time and peak GPU memory of all seven detectors on four graphs. '
           'Cost followed the number of edges more than the number of nodes. On Reddit-Syn (233 thousand nodes, 115 million edges) the '
           f'{NUMWORD[len(rd)]} detectors that finished took {fit(rd)} s and {mem(rd)} GiB, whereas on DGraphFin (1.2 million nodes, '
           f'0.75 million edges) they took {fit(dg)} s and at most {max(r["peak_allocated_bytes"] for r in dg) / GIB:.1f} GiB. ')
    base = by[('Reddit-Syn', 'DLG-Base')]
    if ok(base):
        txt += ('The previous version\'s first-order estimate for DLG-Base on Reddit-Syn, 4.9 GB, left out the sparse message-passing '
                f'workspace and the stored activations; the measured {base["peak_allocated_bytes"] / GIB:.1f} GiB, and the much higher cost '
                'on Reddit-Syn than on the larger DGraphFin, show that such edge-dependent buffers dominate on dense graphs. The estimated '
                'values have therefore been replaced by these measurements. ')
    ad = {g: by[(g, 'AnomalyDAE')]['status'] for g in RES_GRAPHS}
    adfail = [(g, RES_STATUS.get(st, st)) for g, st in ad.items() if st not in ('success', 'COMPLETED')]
    why = {'TIMEOUT_OBSERVED': lambda g: (f'on {g} it reached the 3 h cap, consistent with its exact row-by-row reconstruction visiting every '
                                          'pair of nodes (its primary runs, which had no time cap, completed all five seeds; '
                                          'Table~\\ref{tab:pr_auc_main})'),
           'GPU_OOM': lambda g: f'on {g} its edge attention needed a single 27.4 GiB tensor (64 values for each of the 115 million edges), '
                                'more than the 24 GiB card holds'}
    if adfail:
        txt += f'Within the caps of this measurement, AnomalyDAE did not finish on {NUMWORD[len(adfail)]} graph{"s" if len(adfail) > 1 else ""}: ' + '; '.join(why.get(ad[g], lambda g: f'on {g} ({s})')(g) for g, s in adfail) + '. '
    gd = [(g, by[(g, 'GADNR')]) for g in RES_GRAPHS[1:]]
    done = [(g, r) for g, r in gd if ok(r)]
    if done:
        parts = []
        for g, r in done:
            sp = (r.get('peak_reserved_bytes') or 0) > 24 * GIB
            parts.append(f'{g} in {r["fit_seconds"]:,.0f} s with {r["peak_allocated_bytes"] / GIB:.0f} GiB'
                         + (' (more than the card holds, so it ran only by spilling into system memory)' if sp else ''))
        txt += 'GADNR, the slowest detector, finished ' + '; '.join(parts) + '. '
    nd = [(g, RES_STATUS.get(r['status'], r['status'])) for g, r in gd if not ok(r)]
    if nd:  # the reasons are given once, with the GADNR retry (Section 5.1, Table 2)
        txt += ('It did not finish on ' + oxford([g for g, _ in nd]).replace(' and ', ' or ') + '; the reasons are given with the GADNR retry in '
                'Section~\\ref{sec:res_run} (Table~\\ref{tab:gadnr_retry}). ')
    return txt.strip()


# ---------------------------------------------------------------- A29-08 texts
def real_numbers(runs):
    best = real_best(runs)
    done = [g for g in REAL if all(len(runs[(g, m, 'all')]) == 5 for m in REAL_MODELS)]
    out = {}
    if len(done) < len(REAL) and not PREVIEW:
        miss = [g for g in REAL if g not in done]
        out['REAL_TEXT'] = pending('REAL_TEXT (incomplete: ' + ', '.join(miss) + ')')
        out['KF_REAL'] = pending('KF_REAL')
        out['GUIDE_FEATURE_GRAPHS'] = pending('GUIDE_FEATURE_GRAPHS')
        out['ABS_REAL'] = pending('ABS_REAL')
        return out
    gains = {}
    for g in REAL:
        gm = [real_mean(runs, g, m) for m in GRAPH_MODELS]
        fm = [real_mean(runs, g, m) for m in ('IForest', 'ECOD')]
        gains[g] = max(v for v in gm if v is not None) - max(v for v in fm if v is not None)
    flat = [g for g in REAL if gains[g] <= 0.01]
    out['GUIDE_FEATURE_GRAPHS'] = oxford(flat) if flat else 'none of the five graphs'
    out.update(real_story(runs, gains))
    out['_gains'] = gains
    return out


def real_story(runs, gains):
    """Narrative of the real-label study. The asserts guard the wording: if final results change a group, rewrite the text."""
    mean = lambda g, m, k='pr_auc': real_mean(runs, g, m, k)
    pi = lambda g: float(np.mean([r['test_prevalence'] for m in REAL_MODELS for r in runs[(g, m)]]))
    best = lambda g, ms: max(((m, mean(g, m)) for m in ms if mean(g, m) is not None), key=lambda t: t[1])
    FEAT, REC = ('IForest', 'ECOD'), ('DOMINANT', 'DLG-Base', 'DLG-Aug')
    helped = [g for g in REAL if gains[g] > 0.05]
    flat = [g for g in REAL if abs(gains[g]) <= 0.01 and best(g, REAL_MODELS)[1] < 1.3 * pi(g)]
    feat = [g for g in REAL if gains[g] < -0.05]
    assert helped == ['Weibo', 'Tolokers'] and flat == ['Reddit-Real'] and feat == ['Amazon-Fraud', 'YelpChi'], (helped, flat, feat, gains)
    w_rec = [mean('Weibo', m) for m in REC]
    t_best = sorted(((m, mean('Tolokers', m)) for m in GRAPH_MODELS if mean('Tolokers', m) is not None), key=lambda t: -t[1])
    t_top = [m for m, v in t_best if v >= t_best[0][1] - 0.005]
    rd = [mean('Reddit-Real', m) for m in REAL_MODELS if mean('Reddit-Real', m) is not None]
    af_if, yc_if = mean('Amazon-Fraud', 'IForest'), mean('YelpChi', 'IForest')
    af_g, yc_g = best('Amazon-Fraud', GRAPH_MODELS), best('YelpChi', GRAPH_MODELS)
    af_roc = [mean('Amazon-Fraud', m, 'roc_auc') for m in REC]
    sup = supervised()
    spread = {g: max(mean(g, m) for m in REC) - min(mean(g, m) for m in REC) for g in REAL}
    close = [g for g in REAL if spread[g] <= 0.01]
    text = (
        'The five graphs fall into three groups (Table~\\ref{tab:real_label}).\n\\begin{itemize}\n'
        f'    \\item \\textbf{{The edges helped on Weibo and Tolokers.}} On Weibo the three reconstruction detectors reached AP '
        f'{min(w_rec):.2f}--{max(w_rec):.2f}, about {min(w_rec) / pi("Weibo"):.0f} times the random baseline '
        f'($\\pi_{{\\mathrm{{test}}}} = {pi("Weibo"):.3f}$), while the feature-only detectors stayed at '
        f'{max(mean("Weibo", m) for m in FEAT):.2f}. On Tolokers the best graph detectors ({oxford(t_top)}; AP '
        f'{rng(t_best[len(t_top) - 1][1], t_best[0][1], 2)}) exceeded the feature-only ones ({max(mean("Tolokers", m) for m in FEAT):.2f}), '
        f'whose scores ranked banned workers below the others ($\\pi_{{\\mathrm{{test}}}} = {pi("Tolokers"):.2f}$).\n'
        f'    \\item \\textbf{{Nothing worked on Reddit.}} Every detector stayed between {min(rd):.3f} and {max(rd):.3f} AP, close to '
        f'$\\pi_{{\\mathrm{{test}}}} = {pi("Reddit-Real"):.3f}$, in line with earlier benchmarks on this graph \\cite{{liu2022bond}}.\n'
        f'    \\item \\textbf{{A feature-only detector won on the two review graphs.}} Isolation Forest reached {af_if:.3f} on Amazon-Fraud '
        f'and {yc_if:.3f} on YelpChi, against {af_g[1]:.3f} ({af_g[0]}) and {yc_g[1]:.3f} ({yc_g[0]}) for the best graph detectors. On '
        f'Amazon-Fraud, DOMINANT, DLG-Base and DLG-Aug even ranked fraudulent users below normal ones (ROC-AUC '
        f'{min(af_roc):.2f}--{max(af_roc):.2f}), so their reconstruction error measured something other than fraud there. The labels '
        f'themselves are predictable: the supervised reference reached {sup["Amazon-Fraud"]["HGB-X"]["AP_mean"]:.2f} and '
        f'{sup["YelpChi"]["HGB-X"]["AP_mean"]:.2f} AP on these graphs (Section~\\ref{{sec:res_reference}}).\n\\end{{itemize}}\n'
        f'DOMINANT, DLG-Base and DLG-Aug again behaved alike: their mean AP differed by at most 0.01 on {NUMWORD.get(len(close), len(close))} '
        f'of the five graphs' + (f' and by {max(spread.values()):.3f} on {max(spread, key=spread.get)}' if len(close) < 5 else '') + '. ')
    text += real_gadnr_sentence(runs) + real_rerun_sentence()
    kf = ('On five further graphs with real labels and their own edges, graph detectors clearly beat feature-only detectors on '
          'two (Weibo and Tolokers), no detector beat random ranking on Reddit, and on the two review-fraud graphs (Amazon and '
          'YelpChi) the feature-only Isolation Forest was best.')
    ab = ('on five further graphs with real labels the edges helped on two, while a detector ignoring the edges did best on '
          'two others.')
    return {'REAL_TEXT': text, 'KF_REAL': kf, 'ABS_REAL': ab}


def real_gadnr_sentence(runs):
    parts = []
    for g in ('Amazon-Fraud', 'YelpChi'):
        for m in ('GADNR', 'AnomalyDAE'):
            al = runs[(g, m, 'all')]
            ok = runs[(g, m)]
            if len(ok) == 5:
                continue
            st = sorted({r.get('status') for r in al if r.get('status') != 'success'})
            if not al or len(al) < 5:
                return pending(f'real-label {m} on {g} ({len(al)}/5 records)')
            parts.append(f'{m} on {g} ({len(ok)} of 5 seeds completed; ' + ', '.join(STATUS_TXT.get(x, x) for x in st) + ')')
    if not parts:
        return ''
    return 'Some cells did not complete: ' + '; '.join(parts) + '; they are marked in Table~\\ref{tab:real_label}.'


def real_rerun_sentence():
    """Real-label cells whose earlier attempts crashed (crash_attempt1.json: 2026-11-23, before the CUDA-context fix;
    attempt2_concurrent_spill_*.json: two spilling jobs at once). The reported value is the latest run, made alone."""
    root = LOCAL / 'real_label'
    cells = sorted({p.parent for p in root.rglob('*.json') if p.name == 'crash_attempt1.json' or p.name.startswith('attempt2_')})
    if not cells:
        return ''
    txt = dict(STATUS_TXT, success='completed')
    desc = [f'{c.parent.name} on {c.parent.parent.name} (seed {c.name}; '
            + (txt.get(read_json(c / 'result.json')['status'], read_json(c / 'result.json')['status']) if (c / 'result.json').exists() else 'pending') + ')'
            for c in cells]
    if any(d.endswith('pending)') for d in desc):
        return pending('real-label repeat runs')
    return (f' Earlier attempts at {NUMWORD.get(len(cells), len(cells))} of these runs stopped with GPU memory errors, either before '
            'a fix that records such failures or while a second memory-heavy job was running; each was repeated with one GPU job at a '
            f'time, and the repeat is reported: {"; ".join(desc)}. The earlier attempts are kept in the evidence record.')


# ---------------------------------------------------------------- A29-12 numbers
def sup_numbers():
    s = supervised()
    ap = lambda g, m: s[g][m]['AP_mean']
    X, XN = 'HGB-X', 'HGB-X+NeighbourMean'
    lanl = next(g for g in s if g.startswith('LANL'))
    others = [g for g in s if g != lanl and ap(g, XN) - ap(g, X) > 0.03]
    oth = ''
    if others:
        oth = ' and on ' + oxford(f'{g} from {ap(g, X):.3f} to {ap(g, XN):.3f}' for g in others)
    return {'SUP_ELLIPTIC': f"{ap('Elliptic', X):.3f}", 'SUP_ETHEREUM': f"{ap('Ethereum', X):.3f}",
            'SUP_LANL_X': f'{ap(lanl, X):.3f}', 'SUP_LANL_XN': f'{ap(lanl, XN):.3f}', 'SUP_EDGE_GAIN_OTHERS': oth,
            'SUP_DGRAPHFIN': f"{ap('DGraphFin', XN):.3f}", 'SUP_REDDIT': f"{ap('Reddit-Real', XN):.3f}"}


# ---------------------------------------------------------------- abstract and key findings
ABSTRACT = (
    'Graph neural networks are often proposed for detecting fraud and cyberattacks without labels. Users need to know '
    'whether a detector will run on their graph and whether its top alerts contain real anomalies. We test seven '
    'unsupervised detectors, including two configurations of our Decoupled Local-to-Global graph neural network (DLG-GNN), '
    'on 13 graphs from finance, blockchain and other domains plus an enterprise login graph. Of 91 detector--graph pairs, 84 '
    'completed; all seven failures involved two detectors whose memory or preprocessing cost grew quickly with graph size. '
    'Computing a common reconstruction loss without the node-by-node matrix scales well: <<ABS_EXACT>> Detection quality depended more '
    'on the graph than on the detector. On the Elliptic and DGraphFin financial graphs, all detectors ranked near '
    'random, although a supervised model predicted Elliptic labels well. The three reconstruction detectors led on average '
    'but were statistically indistinguishable, and <<ABS_REAL>> Detector choice should therefore test memory feasibility '
    'and top-alert precision on the target graph.')


def exact_abstract(dv, sc):
    if 'pending' in (dv['DVG_TEXT'] + sc['SCALING_TEXT']).lower():
        return pending('ABS_EXACT')
    return (f'on {NUMWORD.get(int(dv["DVG_NG"]), dv["DVG_NG"])} benchmark graphs it matched the standard loss and top alerts, '
            'with memory growing roughly in proportion to graph size.')


NUMWORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten'}


def key_findings(dv, sc, gtxt):
    kf = {}
    kf['KF_RUN'] = ('Of 91 primary detector--graph pairs, 84 completed all five seeds on a 24 GB GPU. Every failure involved '
                    'AnomalyDAE or GADNR, whose memory or preprocessing grows quickly with graph size.')
    kf['KF_EXACT'] = (pending('KF_EXACT') if 'pending' in (dv['DVG_TEXT'] + sc['SCALING_TEXT']).lower() else
                      f'The memory-saving loss reproduced the dense training loss and anomaly ranking on {dv["DVG_NG"]} benchmark graphs '
                      f'(identical top-1\\% and top-5\\% alert lists), and its memory grew roughly in proportion to graph size, whereas '
                      f'the dense code no longer fitted on a 24 GB GPU beyond {sc["SCALE_DENSE_MAXN"]} nodes.')
    kf['KF_RANK'] = ('DOMINANT, DLG-Base and DLG-Aug formed a top group across graphs but could not be separated: a Bayesian test '
                     'found DLG-Base practically equivalent to each of the other two with probability above 0.9. We make no claim '
                     'that one detector is best.')
    return kf


# ---------------------------------------------------------------- sections, figures, appendix
def sections():
    text = (ROOT / 'projects/benchmark/paper/a29/template/sections.tex').read_text()
    return {m.group(1): m.group(2).strip() for m in re.finditer(r'%%% BEGIN (\w+)\n(.*?)%%% END', text, re.S)}


def figures():
    figs = {}
    for name, path in FIG.items():
        if Path(path).exists():
            figs[name] = path
            continue
        alt = LOCAL / 'draft_figures' / name
        alt.parent.mkdir(parents=True, exist_ok=True)
        if name == 'fig_support_matrix.pdf':  # draft fallback: frozen A28 matrix until the A29 annotation exists
            import zipfile
            with zipfile.ZipFile(ROOT / 'projects/benchmark/paper/a28/applsci/source_bundle.zip') as z:
                alt.write_bytes(z.read('figures/fig_support_matrix.pdf'))
            pending('fig_support_matrix (A28 fallback)')
        else:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            f, ax = plt.subplots(figsize=(6, 2)); ax.axis('off'); ax.text(.5, .5, f'PENDING: {name}', ha='center', va='center')
            f.savefig(alt); plt.close(f)
            pending(name)
        figs[name] = alt
    return figs


APPENDIX_A28 = ['tab:alert_budget', 'tab:conditional_delta', 'tab:friedman_stats', 'tab:decision_matrix', 'tab:lanl_protocols',
                'tab:a24_measured', 'tab:app_roc_auc_full', 'tab:app_f1_full', 'tab:orientation_audit', 'tab:crypto_identity',
                'tab:app_ablation_elliptic', 'tab:a24_ap_sd', 'tab:a24_ap_sd_part2', 'tab:a24_feature_only', 'tab:a24_controls',
                'tab:a24_alert', 'tab:a24_lift', 'tab:a24_split', 'tab:a24_membership']


A28_CAPTION_PART = {'tab:a24_ap_sd': 'Part 1 of 2: DOMINANT, AnomalyDAE, CoLA and GADNR.',
                    'tab:a24_ap_sd_part2': 'Part 2 of 2: OCGNN, DLG-Base and DLG-Aug.'}


def appendix(tables):
    from a29_build_manuscript import a28_between, a28_table
    tex = A28.read_text()
    out = ['\\noindent Tables A1--A19 are carried over unchanged from the frozen primary benchmark (numbers untouched); the '
           'remaining tables belong to the extension studies of this version.\n']
    for lab in APPENDIX_A28:
        if lab == 'tab:orientation_audit':
            out.append(a28_between('\\begin{sloppypar}\\noindent\\textbf{Feature-Only Score Orientation', '\\end{sloppypar}') + '\\end{sloppypar}\n')
        t = a28_table(lab, tex).replace('[H]', '[htbp]')
        if lab in A28_CAPTION_PART:  # the two halves of one A28 table had identical captions; only the caption is touched
            t = t.replace('(ddof=1). ', '(ddof=1). ' + A28_CAPTION_PART[lab] + ' ', 1)
        out.append(t + '\n')
    for k in ('rank_sensitivity', 'tuning', 'real_detail'):
        out.append(tables[k])
    return '\n'.join(out)


def build_all():
    PENDING.clear()
    runs = real_runs()
    tu = tuning()
    dv, sc = dvg_numbers(), scaling_numbers()
    real = real_numbers(runs)
    gtxt = gadnr_text()
    nums = {'KEYWORDS': KEYWORDS, 'FEATURED': FEATURED, 'DATA_AVAILABILITY': data_availability(),
            'PREREG_SHA': read_json(RESULTS / 'preregistration/A29_PREREGISTRATION_BINDING.json')['protocol_sha256'],
            'GADNR_RETRY_TEXT': gtxt,
            'SUPPORT_CAPTION_RETRY': ('(Table~\\ref{tab:gadnr_retry}).'),
            'RESOURCE_TEXT': resource_text(), 'COLA_TEXT': cola_text(), 'OCGNN_TEXT': ocgnn_text(), 'TUNE_TEXT': tune_text(tu),
            'CONC_EXACT': (pending('CONC_EXACT') if 'pending' in dv['DVG_TEXT'].lower() else
                           f'gave the same training loss and anomaly ranking as the dense code on {NUMWORD.get(int(dv["DVG_NG"]), dv["DVG_NG"])} benchmark graphs and '
                           'kept memory growing roughly in proportion to graph size')}
    nums.update(dv); nums.update(sc); nums.update(rank_numbers()); nums.update(sup_numbers())
    nums.update({k: v for k, v in real.items() if not k.startswith('_')})
    nums['ABS_EXACT'] = exact_abstract(dv, sc)
    nums.update(key_findings(dv, sc, gtxt))
    nums['ABSTRACT'] = ABSTRACT
    rb = real_best(runs)
    tables = {'supervised': supervised_table(rb), 'cola_sanity': cola_table(), 'ocgnn_sanity': ocgnn_table(),
              'gadnr_retry': gadnr_table(), 'real_label': real_table(runs), 'real_datasets': real_dataset_table(),
              'real_detail': real_detail_table(runs), 'friedman_compact': friedman_compact_table(),
              'rank_sensitivity': rank_sensitivity_table(), 'dense_vs_gram': dense_vs_gram_table(), 'resource': resource_table(),
              'tuning': tuning_table(tu)}
    figs = figures()
    return nums, sections(), tables, appendix(tables), figs
