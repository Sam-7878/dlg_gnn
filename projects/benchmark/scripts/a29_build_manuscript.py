"""Build the manuscript (Applied Sciences + Preprints variants) of release REV (a29_common) from the template.

* A28 tables are copied verbatim by \\label (numbers untouched); only captions may be simplified.
* A29 numbers come from results/a29 via a29_numbers.numbers(); every <<KEY>> must resolve.
* Each PDF is compiled twice from a clean copy with a fixed SOURCE_DATE_EPOCH; hashes must match.
  python a29_build_manuscript.py [--no-compile]
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import datetime

from a29_common import EVIDENCE_TAG, FREEZE_DATE, PAPER_OUT, PUB_OUT, REV, ROOT, dump

PAPER = ROOT / 'projects/benchmark/paper'
TPL = PAPER / 'a29/template'
A28 = PAPER / 'a28/applsci/DLG-Benchmark_A28_AppliedSciences.tex'
A28P = PAPER / 'a28/preprints/DLG-Benchmark_A28_Preprints.tex'
A28ZIP = PAPER / 'a28/applsci/source_bundle.zip'
_FD = datetime.datetime.fromisoformat(FREEZE_DATE).replace(tzinfo=datetime.timezone.utc)
EPOCH = str(int(_FD.timestamp()))  # freeze date 00:00 UTC: PDF date, header and metadata (A30-02)
ZIPDATE = (_FD.year, _FD.month, _FD.day, 0, 0, 0)
STEM = 'DLG-Benchmark_' + REV.upper() + '_{}'

TITLE = 'Graph Anomaly Detection for Fraud and Cybersecurity: A Cross-Domain Benchmark of Detection Quality and GPU Memory Feasibility'


def a28_table(label, tex=None):
    tex = tex or A28.read_text()
    i = tex.index(r'\label{' + label + '}')
    start = tex.rfind(r'\begin{table}', 0, i)
    end = tex.index(r'\end{table}', i) + len(r'\end{table}')
    return tex[start:end]


def a28_between(start_marker, end_marker):
    tex = A28.read_text()
    a = tex.index(start_marker)
    return tex[a:tex.index(end_marker, a)]


def fill(text, nums):
    def rep(m):
        k = m.group(1)
        if k not in nums:
            raise KeyError(f'unresolved placeholder <<{k}>>')
        return str(nums[k])
    for _ in range(3):  # nested placeholders inside generated text
        text = re.sub(r'<<([A-Z0-9_]+)>>', rep, text)
    return text


def front_applsci(abstract, keywords, featured):
    pre = A28.read_text()
    pre = pre[:pre.index('% Title and Citations')]
    auth = a28_between(r'\newcommand{\orcidauthorA}', r'\featuredapplication')
    auth = re.sub(r'\\\\ \$\^\{1\}\$ \\quad ORCID:.*?(?=\}\n\n\\corres)', '', auth, flags=re.S)  # A30-06 (F9)
    assert 'ORCID:' not in auth, auth
    return (pre + '% Title and Citations\n\\Title{' + TITLE + '}\n\\TitleCitation{' + TITLE + '}\n\n' + auth +
            '\\featuredapplication{' + featured + '}\n\n\\abstract{' + abstract + '}\n\n\\keyword{' + keywords +
            '}\n\n\\reftitle{References}\n')


def front_preprints(abstract, keywords):
    pre = A28P.read_text()
    head = pre[:pre.index('\n\\title{') + 1]  # the line-start \title, not the one inside the \Title macro
    auth = pre[pre.index(r'\author[1]'):pre.index(r'\begin{document}')]
    auth = re.sub(r'(\\author\[1\]\{Seong-Su Park)\\thanks\{ORCID: .*?\}\}\}\n', r'\1}\n', auth, flags=re.S)  # A30-06 (F6)
    auth = auth.replace('; ORCID: \\href{https://orcid.org/0000-0001-6624-3003}{\\mbox{0000-0001-6624-3003}}}}', '}}')
    auth = auth.replace('\\texttt{kkim86@ajou.ac.kr}}\n',
                        '\\texttt{kkim86@ajou.ac.kr}\\\\ ORCID: S.-S.P. \\href{https://orcid.org/0009-0008-4056-3875}{0009-0008-4056-3875}; '
                        'K.-H.K. \\href{https://orcid.org/0000-0001-6624-3003}{0000-0001-6624-3003}}\n', 1)
    assert auth.count('\\thanks') == 1 and 'Seong-Su Park}' in auth and auth.count('ORCID') == 1, auth
    body = ('\\begin{document}\n\\maketitle\n\n\\begin{abstract}' + abstract + '\\end{abstract}\n\n'
            '\\noindent{\\small\\textbf{Keywords: }' + keywords + '}\\par\n')
    return head + '\\title{' + TITLE + '}\n\n' + auth, body


def backmatter(variant, nums):
    tex = A28.read_text()
    contrib = re.search(r'\\authorcontributions\{.*?\}\n', tex, re.S).group(0)
    funding = re.search(r'\\funding\{.*?\}\n', tex, re.S).group(0)
    da = nums['DATA_AVAILABILITY']
    ack = ('The generative AI tools named in Methods assisted with literature and manuscript review, translation and LaTeX '
           'preparation. The authors reviewed the assisted text and source attributions and take responsibility for the manuscript.')
    out = (contrib + '\n' + funding + '\n\\institutionalreview{Not applicable.}\n\n\\informedconsent{Not applicable.}\n\n'
           '\\dataavailability{' + da + '}\n\n')
    if variant == 'applsci':
        out += ('\\acknowledgments{' + ack + '}\n\n\\conflictsofinterest{The authors declare no conflict of interest.}\n\n'
                '\\begin{adjustwidth}{-\\extralength}{0cm}\n\\reftitle{References}\n\\bibliography{references}\n\\end{adjustwidth}\n')
    else:
        out += ('\\par\\smallskip\\noindent\\textbf{Acknowledgments / Use of Generative AI: }' + ack + '\n\n'
                '\\conflictsofinterest{The authors declare no conflict of interest.}\n\n'
                '\\bibliographystyle{IEEEtran}\n\\begingroup\\small\n\\bibliography{references}\n\\endgroup\n')
    return out


def assemble(variant, nums, sections, tables, appendix):
    tex = (TPL / 'manuscript_template.tex').read_text()
    for k, v in sections.items():
        tex = tex.replace('<<' + k + '>>', v)
    for lab in re.findall(r'%%A28TABLE:([^%]+)%%', tex):
        tex = tex.replace('%%A28TABLE:' + lab + '%%', tables.get(lab) or a28_table(lab))
    for k, v in tables.items():
        tex = tex.replace('%%TABLE:' + k + '%%', v)
    tex = tex.replace('%%APPENDIX%%', appendix)
    tex = tex.replace('%%BACKMATTER%%', backmatter(variant, nums))
    if variant == 'applsci':
        tex = tex.replace('%%PREAMBLE%%', front_applsci(nums['ABSTRACT'], nums['KEYWORDS'], nums['FEATURED']))
    else:
        head, body = front_preprints(nums['ABSTRACT'], nums['KEYWORDS'])
        tex = tex.replace('%%PREAMBLE%%', head).replace('\\begin{document}\n', body, 1)
    tex = fill(tex, nums)
    left = re.findall(r'<<[A-Z0-9_]+>>|%%[A-Z0-9]+[:A-Za-z_]*%%', tex)
    assert not left, left
    return tex


def compile_variant(dirpath: Path, stem: str):
    hashes = []
    for run in ('clean1', 'clean2'):
        out = dirpath / run
        if out.exists():
            shutil.rmtree(out)
        with zipfile.ZipFile(dirpath / 'source_bundle.zip') as z:
            z.extractall(out)
        log = (dirpath / f'{run}.stdout').open('wb')
        subprocess.run(['latexmk', '-g', '-pdf', '-interaction=nonstopmode', '-halt-on-error', stem + '.tex'], cwd=out,
                       env=dict(os.environ, SOURCE_DATE_EPOCH=EPOCH, FORCE_SOURCE_DATE='1'), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
        hashes.append(hashlib.sha256((out / (stem + '.pdf')).read_bytes()).hexdigest())
        final = (out / (stem + '.log')).read_text(errors='replace')
        warn = [r for r in final.splitlines() if any(t in r for t in ('Overfull', 'undefined', 'Float too large', 'multiply defined'))]
    assert hashes[0] == hashes[1], hashes
    shutil.copyfile(out / (stem + '.pdf'), dirpath / (stem + '.pdf'))
    subprocess.run(['pdftotext', '-layout', str(dirpath / (stem + '.pdf')), str(dirpath / 'native.txt')], check=True)
    pages = subprocess.run(['pdfinfo', str(dirpath / (stem + '.pdf'))], capture_output=True, text=True).stdout
    return {'pdf_sha256': hashes[0], 'byte_identical_rebuilds': True, 'warnings': warn,
            'pages': int(re.search(r'Pages:\s+(\d+)', pages).group(1))}


def bundle(dirpath: Path, stem: str, tex: str, figures: dict):
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / (stem + '.tex')).write_text(tex)
    with zipfile.ZipFile(A28ZIP) as src, zipfile.ZipFile(dirpath / 'source_bundle.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for info in src.infolist():
            if (info.filename.startswith('Definitions/') or info.filename.startswith('soul')
                    or info.filename in ('figures/fig_dlg_architecture.pdf', 'figures/fig_a24_delta_sample_sd.pdf')):
                z.writestr(zipfile.ZipInfo(info.filename, date_time=ZIPDATE), src.read(info.filename))
        z.writestr(zipfile.ZipInfo(stem + '.tex', date_time=ZIPDATE), tex)
        z.writestr(zipfile.ZipInfo('references.bib', date_time=ZIPDATE), (TPL / 'references.bib').read_text())
        for name, path in sorted(figures.items()):
            z.writestr(zipfile.ZipInfo('figures/' + name, date_time=ZIPDATE), Path(path).read_bytes())


def plain(tex):
    """LaTeX snippet -> plain text for the cover letter / inquiry (abstract contains no maths)."""
    s = tex.replace('---', '\u2014').replace('--', '\u2013').replace('~', ' ').replace('\\%', '%')
    s = re.sub(r'\\(emph|textbf|textit)\{([^}]*)\}', r'\2', s)
    s = re.sub(r'\\[a-zA-Z]+\s?', '', s).replace('{', '').replace('}', '').replace('$', '')
    return re.sub(r'\s+', ' ', s).strip()


def abstract_words(tex):
    return len(plain(tex).split())


def submission_docs(nums):
    out = {}
    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    for tpl, name in (('COVER_LETTER_template.md', f'COVER_LETTER_{REV.upper()}.md'), ('SI_EDITOR_INQUIRY_template.md', f'SI_EDITOR_INQUIRY_{REV.upper()}.md')):
        text = ((TPL / tpl).read_text().replace('<<TITLE>>', TITLE).replace('<<ABSTRACT_PLAIN>>', plain(fill(nums['ABSTRACT'], nums)))
                .replace('<<REV>>', REV.upper()).replace('<<REV_LOWER>>', REV).replace('<<EVIDENCE_TAG>>', EVIDENCE_TAG))
        assert '<<' not in text, re.findall(r'<<[A-Z_]+>>', text)
        (PAPER_OUT / name).write_text(text)
        out[name] = str((PAPER_OUT / name).relative_to(ROOT))
    return out


def main(compile_pdf=True, final=False):
    import a29_numbers
    nums, sections, tables, appendix, figures = a29_numbers.build_all()
    if a29_numbers.PENDING:
        print('PENDING items:', *sorted(set(a29_numbers.PENDING)), sep='\n  ', flush=True)
        if final:
            raise SystemExit('final build refused: pending items remain')
    report = {'release': REV, 'freeze_date': FREEZE_DATE, 'source_date_epoch': EPOCH, 'final': final, 'pending': sorted(set(a29_numbers.PENDING)),
              'abstract_words': abstract_words(fill(nums['ABSTRACT'], nums)), 'submission_docs': submission_docs(nums)}
    print('abstract words:', report['abstract_words'], flush=True)
    if final and report['abstract_words'] > 200:
        raise SystemExit('final build refused: abstract exceeds the 200-word limit of Applied Sciences')
    for variant, venue in (('applsci', 'AppliedSciences'), ('preprints', 'Preprints')):
        stem = STEM.format(venue)
        tex = assemble(variant, nums, sections, tables, appendix)
        d = PAPER_OUT / variant
        bundle(d, stem, tex, figures)
        if compile_pdf:
            report[variant] = compile_variant(d, stem)
            print(variant, report[variant], flush=True)
    PUB_OUT.mkdir(parents=True, exist_ok=True)
    dump(PUB_OUT / 'manuscript_build.json', report)


if __name__ == '__main__':
    main('--no-compile' not in sys.argv, final='--final' in sys.argv)
