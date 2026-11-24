"""A29-11 figures: AP-lift dot plot (from the frozen A28 Table 2) and the updated support matrix.

  python a29_figures.py lift
  python a29_figures.py support
"""
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from a29_common import LOCAL, RESULTS, ROOT, read_json

A28 = ROOT / 'projects/benchmark/paper/a28/applsci/DLG-Benchmark_A28_AppliedSciences.tex'
OUT = RESULTS / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
MODELS = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'GADNR', 'OCGNN', 'DLG-Base', 'DLG-Aug']
STYLE = {'DOMINANT': ('#1f4e79', 'o'), 'AnomalyDAE': ('#d35400', 'D'), 'CoLA': ('#7f8c8d', 'x'), 'GADNR': ('#8e44ad', 'P'),
         'OCGNN': ('#95a5a6', 's'), 'DLG-Base': ('#16a085', '^'), 'DLG-Aug': ('#c0392b', 'v')}


def table2():
    tex = A28.read_text()
    block = tex[tex.index(r'\label{tab:pr_auc_main}'):]
    block = block[:block.index(r'\end{tabular*}')]
    rows = {}
    for line in block.splitlines():
        if '&' not in line or 'Dataset' in line:
            continue
        cells = [c.strip() for c in line.replace(r'\\', '').replace(r'\midrule', '').replace(r'\bottomrule', '').split('&')]
        name = re.sub(r'\$\^\\dagger\$', '', cells[0]).strip()
        pi = float(re.sub(r'[^0-9.]', '', cells[1]))
        vals = []
        for c in cells[2:9]:
            c = re.sub(r'\\textbf\{([^}]*)\}', r'\1', c)
            vals.append(None if 'N/A' in c else float(c))
        rows[name] = (pi, dict(zip(MODELS, vals)))
    return rows


def lift():
    rows = table2()
    order = list(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, g in enumerate(order):
        pi, vals = rows[g]
        for j, m in enumerate(MODELS):
            v = vals[m]
            if v is None:
                continue
            c, mk = STYLE[m]
            ax.scatter(v / pi, i + (j - 3) * 0.09, color=c, marker=mk, s=18, zorder=3,
                       linewidths=0.9 if mk == 'x' else 0)
    ax.axvline(1, color='k', ls='--', lw=0.8)
    ax.set_xscale('log')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{g} ($\\pi$={rows[g][0]:.3f})' for g in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('AP / test prevalence (1 = random ranking)', fontsize=8)
    ax.tick_params(axis='x', labelsize=7)
    ax.grid(axis='x', which='both', lw=0.3, alpha=0.5)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=STYLE[m][0], marker=STYLE[m][1], ls='', ms=5, label=m) for m in MODELS]
    ax.legend(handles=handles, fontsize=6.5, ncol=7, loc='lower center', bbox_to_anchor=(0.45, 1.0), frameon=False,
              handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_ap_lift.pdf'); fig.savefig(OUT / 'fig_ap_lift.png', dpi=200)
    print('lift figure written', len(order), 'graphs')


def support():
    """Primary support status (frozen) with the A29-06 GADNR retry outcome added as an annotation."""
    graphs = ['Amazon-Syn', 'BSC', 'BitcoinOTC', 'CiteSeer-Syn', 'Cora-Syn', 'DGraphFin', 'Elliptic', 'Ethereum', 'Flickr-Syn',
              'Polygon', 'PubMed-Syn', 'Reddit-Syn', 'Yelp-Syn', 'LANL-RedTeam']
    fail = {('DGraphFin', 'GADNR'): 'no run', ('Elliptic', 'GADNR'): 'no run', ('Flickr-Syn', 'GADNR'): 'OOM 8 GB',
            ('Reddit-Syn', 'GADNR'): 'no run', ('Yelp-Syn', 'GADNR'): 'no run',
            ('Reddit-Syn', 'AnomalyDAE'): 'OOM 24 GB', ('Yelp-Syn', 'AnomalyDAE'): 'OOM 24 GB'}
    short = {'COMPLETED': 'retry: ran', 'EXCEEDS_CAP_PROJECTED': 'retry: >12 h', 'TIMEOUT_OBSERVED': 'retry: >12 h',
             'HOST_OOM': 'retry: host RAM', 'GPU_OOM': 'retry: GPU OOM', 'OTHER_ERROR': 'retry: error'}
    retry = {}
    for g in graphs:
        p = LOCAL / 'gadnr_retry' / f'{g}__seed42' / 'result.json'
        if p.exists():
            retry[g] = short.get(read_json(p)['status'], read_json(p)['status'])
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    for i, g in enumerate(graphs):
        for j, m in enumerate(MODELS):
            if g == 'LANL-RedTeam':
                col, txt = '#dbe9f6', 'external'
            elif (g, m) in fail:
                col, txt = '#f9dcdc', fail[(g, m)]
                if m == 'GADNR' and g in retry:
                    txt += '\n' + retry[g]
            else:
                col, txt = '#ddf2e3', '5/5'
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=col, edgecolor='white', lw=1.5))
            ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=6.2)
    ax.set_xlim(0, len(MODELS)); ax.set_ylim(len(graphs), 0)
    ax.set_xticks(np.arange(len(MODELS)) + 0.5); ax.set_xticklabels(MODELS, fontsize=7, rotation=25, ha='right')
    ax.set_yticks(np.arange(len(graphs)) + 0.5); ax.set_yticklabels(graphs, fontsize=7)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('Primary support: 84 of 91 pairs ran for all five seeds', fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_support_matrix.pdf'); fig.savefig(OUT / 'fig_support_matrix.png', dpi=200)
    print('support figure written', retry)


if __name__ == '__main__':
    {'lift': lift, 'support': support}[sys.argv[1]]()
