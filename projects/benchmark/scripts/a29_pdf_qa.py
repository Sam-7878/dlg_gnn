"""A29 PDF quality assurance for both manuscript variants.

Checks per variant: leftover placeholders (<<, %%, PENDING, ??), LaTeX warnings of the last pass, page map of every
numbered table/figure, table order, and that each carried-over A28 table body is byte-identical to the frozen A28 source
(only captions of the two split AP tables carry a part note). Writes publication/a29/{applsci,preprints}_qa.json.
  python a29_pdf_qa.py
"""
from __future__ import annotations

import json
import re
import subprocess

from a29_build_manuscript import A28, EPOCH, PAPER, a28_table
from a29_common import ROOT, dump
from a29_numbers import APPENDIX_A28

PUB = ROOT / 'projects/benchmark/publication/a29'
VARIANTS = {'applsci': 'DLG-Benchmark_A29_AppliedSciences', 'preprints': 'DLG-Benchmark_A29_Preprints'}


def body(t):
    """Table without its caption, for the A28 identity check."""
    return re.sub(r'\\caption\{(?:[^{}]|\{[^{}]*\})*\}', '', t).replace('[H]', '[htbp]')


def pages(pdf):
    n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout).group(1))
    out = []
    for p in range(1, n + 1):
        out.append(subprocess.run(['pdftotext', '-layout', '-f', str(p), '-l', str(p), pdf, '-'], capture_output=True, text=True).stdout)
    return out


def qa(variant, stem):
    d = PAPER / 'a29' / variant
    pdf = str(d / (stem + '.pdf'))
    txt = pages(pdf)
    page_map, order = {}, []
    for i, t in enumerate(txt, 1):
        for m in re.finditer(r'^\s*(Table A?\d+|Figure \d+)[.:]\s', t, re.M):
            if m.group(1) not in page_map:
                page_map[m.group(1)] = i
                order.append(m.group(1))
    full = '\n'.join(txt)
    leftovers = sorted(set(re.findall(r'<<[A-Z0-9_]+>>|%%[A-Z0-9]+[:A-Za-z_]*%%|PENDING[^\n]{0,40}|\?\?', full)))
    log = (d / 'clean2' / (stem + '.log')).read_text(errors='replace')
    warn = [r for r in log.splitlines() if any(k in r for k in ('Overfull', 'undefined', 'Float too large', 'multiply defined', 'Missing character'))]
    tex = (d / (stem + '.tex')).read_text()
    a28 = A28.read_text()
    identity = {lab: body(a28_table(lab, a28)) in body(tex) for lab in APPENDIX_A28}
    main_tables = [k for k in order if re.fullmatch(r'Table \d+', k)]
    app_tables = [k for k in order if k.startswith('Table A')]
    rec = {'pdf_sha256': json.loads((ROOT / 'projects/benchmark/results/a29/manuscript_build.json').read_text())[variant]['pdf_sha256'],
           'pages': len(txt), 'page_map': page_map,
           'main_tables_in_order': main_tables == sorted(main_tables, key=lambda k: int(k.split()[1])),
           'appendix_tables_in_order': app_tables == sorted(app_tables, key=lambda k: int(k.split()[1][1:])),
           'appendix_tables': len(app_tables), 'main_tables': len(main_tables), 'figures': sum(k.startswith('Figure') for k in order), 'leftovers': leftovers, 'log_warnings': warn,
           'a28_tables_identical': identity, 'a28_tables_all_identical': all(identity.values()),
           'abstract_has_no_pending': 'PENDING' not in full.split('Keywords')[0]}
    dump(PUB / f'{variant}_qa.json', rec)
    return rec


def key(k):
    kind, num = k.split()
    return (kind == 'Figure', num.startswith('A'), int(num.lstrip('A')))


def write_md(recs, epoch):
    """Human-readable QA record in the A28 format (publication/a29/A29_PDF_QA.md)."""
    a, p = recs['applsci'], recs['preprints']
    build = json.loads((ROOT / 'projects/benchmark/results/a29/manuscript_build.json').read_text())
    ok = all(build[v]['byte_identical_rebuilds'] for v in recs)
    clean = all(not r['log_warnings'] and not r['leftovers'] for r in recs.values())
    lines = ['# A29 PDF quality assurance', '',
             (f'Both source bundles produced {"byte-identical" if ok else "NON-IDENTICAL"} PDFs in two fresh builds at fixed epoch {epoch}. '
              f'The final builds showed {"zero" if clean else "SOME"} overfull boxes, undefined references, oversized floats or leftover placeholders. '
              f'Each variant has {a["main_tables"]} main tables, {a["figures"]} figures and {a["appendix_tables"]} appendix tables, '
              f'{"in order" if all(r["main_tables_in_order"] and r["appendix_tables_in_order"] for r in recs.values()) else "NOT in order"}. '
              f'The {len(a["a28_tables_identical"])} tables carried over from A28 are '
              f'{"identical to the A28 source apart from the part notes in two captions" if all(r["a28_tables_all_identical"] for r in recs.values()) else "NOT all identical"}.'),
             '', '| Artifact | Preprints page | Applied Sciences page |', '|---|---:|---:|']
    for k in sorted(set(a['page_map']) | set(p['page_map']), key=key):
        lines.append(f'| {k} | {p["page_map"].get(k, "missing")} | {a["page_map"].get(k, "missing")} |')
    lines += ['', f'Preprints PDF has {p["pages"]} pages and SHA-256 `{p["pdf_sha256"]}`.', '',
              f'Applied Sciences PDF has {a["pages"]} pages and SHA-256 `{a["pdf_sha256"]}`.', '',
              '[Submission hash manifest](A29_SUBMISSION_HASHES.sha256)', '']
    (PUB / 'A29_PDF_QA.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    recs = {}
    for v, s in VARIANTS.items():
        r = recs[v] = qa(v, s)
        print(v, {k: r[k] for k in ('pages', 'main_tables', 'figures', 'main_tables_in_order', 'appendix_tables_in_order', 'appendix_tables',
                                     'a28_tables_all_identical', 'leftovers', 'log_warnings')})
    write_md(recs, EPOCH)
