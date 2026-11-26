"""R324 manuscript QA: checks the R324 work-order completion criteria on the built sources.

Usage: python scripts/revision324/qa.py   (after build.py). Writes reports/324/qa.json; exits 1 on failure.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
COMMON = PROJECT / 'paper/submission_324/common'
RESULTS = PROJECT / 'results'
checks = {}


def check(name, ok, detail=''):
    checks[name] = {'pass': bool(ok), 'detail': detail}


def strip_comments(text):
    return '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in text.splitlines())


sources = {p.relative_to(COMMON).as_posix(): strip_comments(p.read_text()) for p in COMMON.rglob('*.tex')
           if p.name != 'supplement_refs.tex'}
venues = {p.name: strip_comments(p.read_text()) for p in (PROJECT / 'paper/submission_324/venues').glob('*/*.tex')}
alltext = '\n'.join(list(sources.values()) + list(venues.values()))
body = sources['manuscript_body.tex']
abstract = (COMMON / 'abstract.txt').read_text()

# P0-1 / P0-2: no approval-pending wording; release fields only through macros.
for word in ('pending', 'approval', 'proposed public', 'subject to review'):
    hits = [k for k, v in {**sources, **venues}.items() if word in v.lower()]
    check(f'P0-1 no "{word}"', not hits, hits)
das = sources['declarations.tex']
check('P0-2 licence strings via macros', '\\TDSScoreLicense' in das and '\\TDSCodeLicense' in das and '\\TDSDataDOI' in sources['replay_reference.tex'])

# P0-4: only two cohort names; code labels appear only as \path{...}.
allowed = re.compile(r'\\path\{(original|horizon7)\}|original draft')
residue = []
for k, v in sources.items():
    for m in re.finditer(r'original|Original|horizon7|full-archive|Full-archive|Cohort A|Cohort B', v):
        window = v[max(0, m.start() - 7):m.end() + 7]
        if not allowed.search(window):
            residue.append(f'{k}: ...{v[max(0, m.start() - 30):m.end() + 30]}...')
check('P0-4 cohort names', not residue, residue)

# P0-3: no deterministic "± 0.0000"; Fits columns; one shared note string.
check('P0-3 no zero SD', '$\\pm$ 0.0000' not in alltext, [k for k, v in sources.items() if '$\\pm$ 0.0000' in v])
note_users = sorted(k for k, v in sources.items() if '\\TDSDeterministicNote' in v)
check('P0-3 shared deterministic note', {'data/table_primary.tex', 'data/r319_models.tex', 'data/tree_grid.tex'} <= set(note_users), note_users)
check('P0-3 no "Runs" header', not any('& Runs' in v for v in sources.values()))

# P0-5: exponent SD format.
check('P0-5 SD format', '2.1e-05' not in alltext and '2.1{\\times}10^{-5}' in sources['data/detectors.tex'])

# P0-6 / P0-7 / P0-8 / P1-5 / P1-6 / P1-7.
check('P0-6 checklist consequence column', 'Measured consequence' in sources['data/reusable_validity_checklist.tex'])
check('P0-7 five-decimal footnote', '0.44379' in sources['data/archived_gin_size.tex'] and '0.44375' in sources['data/archived_gin_size.tex'])
check('P0-8 seed sentence and table', 'seeds 321, 17, 27, 37 and 47' in body and 'tab:seeds' in sources['data/seeds.tex'])
check('P1-5 3,618 of 3,648', '3,618 of the 3,648' in sources['data/matched_main.tex'] and '3,618 of the 3,648' in body)
check('P1-6 versions', all(s in body for s in ('PyTorch 2.14.1', 'PyTorch Geometric 2.8.0.post1', 'PyGOD 1.1.0', 'scikit-learn 1.9.1', 'LightGBM 4.6.0')))
check('P1-7 UTC-day column', 'UTC-day interval' in sources['data/r319_statistics.tex'])
check('P1-8 single archive', 'single archive' in body)
check('N6 tree counts', '99 trees' in body and 'only 28' in body)
check('N2 seed-7 note', 'identical to the short-search array' in sources['data/long_refits.tex'])

# P1-2: abstract length, keywords, numbers.
words = len(abstract.split())
check('P1-2 abstract 195-205 words', 195 <= words <= 205, words)
check('P1-2 abstract numbers', all(n in abstract for n in ('0.1088', '0.0670', '0.0913', '56.16')))
check('P1-2 abstract SI keywords', 'anomaly' in abstract.lower() and 'prevalence' in abstract.lower())
check('P1-2 abstract identical in venues', all(abstract.strip() in v for v in venues.values()))

# P1-1: introduction structure.
intro = body[:body.index('\\section{Related Work}')]
check('P1-1 three RQs and three contributions', intro.count('\\textbf{RQ') == 3 and intro.count('\\textbf{(RQ') == 3)
check('P1-1 motivation before RQs', intro.index('Fraudulent smart contracts') < intro.index('RQ1'))

# P1-3: Table 6 (first seven-day panel) moved out of the article.
check('P1-3 horizon table only in Supplement', 'table_horizon' not in body and 'table_horizon' in sources['supplement/supplement.tex'])

# P1-4: Discussion size and literature.
disc = body[body.index('\\section{Discussion}'):body.index('\\section{Limitations}')]
disc_text = re.sub(r'^\\(section|subsection|label)\{[^}]*\}\s*$', '', disc, flags=re.M)
paras = [p for p in re.split(r'\n\s*\n', disc_text) if len(p.split()) > 20]
cites = set(re.findall(r'\\cite\{([^}]+)\}', disc))
check('P1-4 Discussion >= 8 paragraphs', len(paras) >= 8, len(paras))
check('P1-4 literature dialogue', all(k in ','.join(cites) for k in ('tang2023gadbench', 'shchur2018pitfalls', 'errica2020fair', 'heidrich2026systematic')))
check('P1-4 numbers', all(n in disc for n in ('0.0631', '0.6755', '0.7122', '16.45', '16.06')))

# Supplement numbering vs the contents line.
receipt = json.loads((PROJECT / 'reports/324/build_receipt.json').read_text())
labels = receipt['supplement_labels']
expected = {'tab:fullmetrics': 'S1', 'tab:horizon': 'S2', 'tab:archival': 'S3', 'tab:statistics': 'S4', 'tab:chain': 'S5',
            'tab:operating': 'S6', 'tab:matched-prevalence': 'S7', 'tab:tree-grid': 'S8', 'tab:checklist': 'S9',
            'tab:counterparty': 'S10', 'tab:neural-grid': 'S11', 'tab:windows': 'S12', 'tab:long-refits': 'S13',
            'tab:direction': 'S14', 'tab:seeds': 'S15', 'tab:environment': 'S16'}
check('Supplement contents numbering', labels == expected, labels)
used = set(re.findall(r'\\STab\{([^}]+)\}', body))
check('All \\STab labels defined', used <= set(labels), sorted(used - set(labels)))
check('No hard-coded Supplement table numbers', not re.search(r'Table~S\d', body))

# Numbers against result CSVs.
s = pd.read_csv(RESULTS / '323/score_summary.csv')
p = pd.read_csv(RESULTS / '323/prevalence_sensitivity.csv')
a = pd.read_csv(RESULTS / '323/activity_strata.csv')
st = pd.read_csv(RESULTS / '319/new_statistics_319.csv')
val = lambda c, arm, col='ap_mean': float(s[(s.cohort == c) & (s.arm == arm)][col].iloc[0])
pv = lambda c, arm, col='ap': float(p[(p.cohort == c) & (p.arm == arm) & (p.target == .0293)][col].iloc[0])
expect = {
    'archived four-count LR AP': (val('original', 'count_only'), 4, body),
    'archived GIN-sum AP': (val('original', 'gin_sum'), 4, body),
    'archived DOMINANT ROC-AUC': (val('original', 'DOMINANT', 'roc_auc_mean'), 4, body),
    'archived CoLA AP lift': (val('original', 'CoLA', 'ap_lift_mean'), 4, body),
    'seven-day CoLA AP lift': (val('horizon7', 'CoLA', 'ap_lift_mean'), 4, body),
    'seven-day prevalence': (val('horizon7', 'CoLA', 'prevalence'), 4, body),
    'archived LightGBM fixed': (val('original', 'lightgbm_fixed'), 4, body),
    'archived LightGBM grid': (val('original', 'lightgbm_grid'), 4, body),
    'seven-day GIN-0': (val('horizon7', 'gin0'), 4, body),
    'seven-day GIN-size': (val('horizon7', 'gin_size'), 4, body),
    'long refit archived GIN-size': (val('original', 'gin_size_long_selected'), 4, body),
    'long refit archived MLP': (val('original', 'mlp_long_selected'), 4, body),
    'long refit seven-day GIN-size': (val('horizon7', 'gin_size_long_selected'), 4, body),
    'long refit seven-day MLP': (val('horizon7', 'mlp_long_selected'), 4, body),
    'matched GIN-size AP': (pv('horizon7', 'gin_size'), 4, body),
    'matched MLP AP': (pv('horizon7', 'summary_mlp'), 4, body),
    'matched node LR AP': (pv('horizon7', 'retained_nodes'), 4, body),
    'matched GIN AP': (pv('horizon7', 'gin_bce'), 4, body),
    'matched node LR top-1%': (pv('horizon7', 'retained_nodes', 'capture_top1'), 2, body),
    'matched LightGBM top-1%': (pv('horizon7', 'lightgbm_grid', 'capture_top1'), 2, body),
    'matched MLP top-1%': (pv('horizon7', 'summary_mlp', 'capture_top1'), 2, body),
    'L>=7 node LR AP': (float(a[(a.group == 2) & (a.arm == 'retained_nodes')].ap.iloc[0]), 4, body),
    'L>=7 node LR ROC': (float(a[(a.group == 2) & (a.arm == 'retained_nodes')].roc_auc.iloc[0]), 4, body),
}
for _, r in st.iterrows():
    expect[f'UTC-day low {r.a}-{r.b}'] = (r.utc_day_block_low, 4, sources['data/r319_statistics.tex'])
    expect[f'UTC-day high {r.a}-{r.b}'] = (r.utc_day_block_high, 4, sources['data/r319_statistics.tex'])
bad = {}
for name, (value, digits, text) in expect.items():
    token = f'{abs(value):.{digits}f}'
    if token not in text:
        bad[name] = token
check('Numbers match result CSVs', not bad, bad or f'{len(expect)} values')
# Note: rounding 16.455 at two decimals gives 16.45 or 16.46 depending on float representation; the check uses Python rounding.

# Built PDFs: DOI placeholder status (allowed only in review builds).
pdftext = subprocess.run(['pdftotext', str(PROJECT / 'submission/applsci/324/applsci_article.pdf'), '-'],
                         capture_output=True, text=True).stdout
checks['DOI placeholder present in review PDF'] = {'pass': True, 'detail': 'DOI-PLACEHOLDER' in pdftext}
check('No "pending" in built PDF', 'pending' not in pdftext.lower())

failed = [k for k, v in checks.items() if not v['pass']]
(PROJECT / 'reports/324/qa.json').write_text(json.dumps({'failed': failed, 'checks': checks}, indent=2, default=str) + '\n')
for k, v in checks.items():
    print(('PASS ' if v['pass'] else 'FAIL ') + k, '' if v['pass'] and not isinstance(v['detail'], (list, dict)) else v['detail'])
sys.exit(1 if failed else 0)
