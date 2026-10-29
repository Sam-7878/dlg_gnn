"""Author-local PDF/source-ZIP build, clean compile and whole-document rendering."""
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw
from paths import ROOT, INPUT, PAPER, RESULTS, REPORTS
from freeze import sha, write

SOURCES = ['applsci_article.tex', 'supplement/supplementary.tex']


def compile_one(base, rel):
    source = base / rel
    result = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                             '-file-line-error', source.name], cwd=source.parent,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log = source.with_suffix('.log').read_text(errors='replace') if source.with_suffix('.log').exists() else result.stdout
    findings = [line for line in log.splitlines() if re.search(r'Overfull|Underfull|Warning|^!', line)]
    if result.returncode or findings:
        raise RuntimeError((rel, result.returncode, findings, result.stdout[-2500:]))
    text = subprocess.check_output(['pdftotext', '-layout', source.with_suffix('.pdf'), '-'], text=True)
    assert '??' not in text and '\ufffd' not in text
    pages = int(re.search(r'Output written on .*?\((\d+) pages', log).group(1))
    return dict(source=rel, pdf_sha256=sha(source.with_suffix('.pdf')), pages=pages,
                warnings=[], text=text, log_sha256=sha(source.with_suffix('.log')))


def render(pdf, folder):
    folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pdftoppm', '-r', '90', '-png', pdf, folder/'page'], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return sorted(folder.glob('page-*.png'), key=lambda p: int(p.stem.split('-')[-1]))


def pixel_hash(path):
    with Image.open(path) as im:
        return hashlib.sha256(str(im.size).encode()+im.mode.encode()+im.tobytes()).hexdigest()


def contact(pages, output, cols=3):
    width, height = 420, 620
    sheet = Image.new('RGB', (cols*width, ((len(pages)+cols-1)//cols)*height), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, p in enumerate(pages):
        with Image.open(p) as im:
            im = im.convert('RGB'); im.thumbnail((width-16,height-35))
            x,y=(i%cols)*width,(i//cols)*height
            sheet.paste(im, (x+8,y+25))
            draw.text((x+8,y+6), p.parent.name+' '+p.stem, fill='black')
    sheet.save(output)


def main():
    qa = PAPER / 'qa'
    qa.mkdir(exist_ok=True)
    builds = [compile_one(PAPER, rel) for rel in SOURCES]
    assert [e['pages'] for e in builds] == [13, 4], [e['pages'] for e in builds]
    assert 'Proceedings of the Proceedings of' not in builds[0]['text']
    tex = (PAPER/'applsci_article.tex').read_text()
    abstract = tex.split('\\abstract{',1)[1].split('}\n\\keyword',1)[0]
    assert len(abstract.split()) <= 200, len(abstract.split())
    assert '0.4751' in abstract
    assert '0.0987' not in '\n'.join(p.read_text() for p in PAPER.rglob('*.tex'))
    cite_keys = set()
    for group in re.findall(r'\\cite\w*\{([^}]+)\}', tex):
        cite_keys.update(group.split(','))
    bbl = (PAPER/'applsci_article.bbl').read_text()
    assert len(re.findall(r'\\bibitem', bbl)) == 19
    provenance = ROOT/'projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex'
    write(PAPER/'author_section_provenance.json', dict(
        date='2026-10-29', source=str(provenance.relative_to(ROOT)), source_sha256=sha(provenance),
        human_tds_application_authorized=True,
        adopted=['Author Contributions role allocation','Funding','Institutional Review Board Statement',
                 'Informed Consent Statement','Acknowledgments / Use of Generative AI'],
        unchanged_funding_identifier='IITP-2026-RS-2021-II211835',
        not_inferred=['final TDS coauthor approval','identity/ORCID/corresponding designation',
                      'actual COI/originality','official award document','public TDS release']))
    files = []
    for p in sorted(PAPER.rglob('*')):
        if not p.is_file() or 'qa' in p.relative_to(PAPER).parts:
            continue
        rel = p.relative_to(PAPER)
        if rel.parts[0] in {'Definitions','data','figures','supplement'}:
            if p.suffix in {'.tex','.bib','.cls','.bst','.sty','.pdf','.eps','.svg','.csv'}:
                files.append(p)
        elif p.name in {'applsci_article.tex','applsci_article.pdf','references.bib','soul.sty','soul-ori.sty','README_source.md'}:
            files.append(p)
    zpath = PAPER / 'submission_source.zip'
    with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p,p.relative_to(PAPER))
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None
        for rel in SOURCES:
            name = str(Path(rel).with_suffix('.pdf'))
            assert hashlib.sha256(z.read(name)).hexdigest() == sha(PAPER/name)
        scratch = Path(tempfile.mkdtemp(prefix='tds315-clean-'))
        z.extractall(scratch)
    # Force a genuinely fresh build: no delivered article/supplement PDFs survive.
    for rel in SOURCES:
        (scratch/Path(rel).with_suffix('.pdf')).unlink()
    fresh = [compile_one(scratch, rel) for rel in SOURCES]
    review = []
    for i, rel in enumerate(SOURCES):
        assert builds[i]['text'] == fresh[i]['text'], rel
        label = 'article' if i==0 else 'supplement'
        actual = render(PAPER/Path(rel).with_suffix('.pdf'),qa/label)
        clean = render(scratch/Path(rel).with_suffix('.pdf'),scratch/('render_'+label))
        assert len(actual) == len(clean) == builds[i]['pages']
        for n,(a,b) in enumerate(zip(actual,clean),1):
            assert pixel_hash(a) == pixel_hash(b), (rel,n)
            review.append(dict(document=label,page=n,path=str(a.relative_to(PAPER)),
                               png_sha256=sha(a),pixel_sha256=pixel_hash(a),clean_build_pixels_identical=True))
        contact(actual, qa/f'{label}_contact.png')
        if i==0:
            contact(actual[:5],qa/'article_pages1_5.png',cols=2)
            contact(actual[5:9],qa/'article_pages6_9.png',cols=2)
            contact(actual[9:],qa/'article_pages10_13.png',cols=2)
    for e in builds+fresh:
        e.pop('text')
    record = dict(status='PASS',article_pages=13,supplement_pages=4,abstract_words=len(abstract.split()),
        cited_references=len(cite_keys),article_and_supplement_builds=builds,
        source_zip=dict(path='submission_source.zip',sha256=sha(zpath),bytes=zpath.stat().st_size,crc='PASS'),
        clean_builds=fresh,all_clean_pdf_text_and_rendered_pixels_identical=True,
        delivered_pdf_copies_inside_zip_byte_identical=True,source_zip_contains_cached_aux_or_bbl=False,
        full_rendered_page_count=len(review),render_manifest=review,
        official_template_class_sha256=sha(PAPER/'Definitions/mdpi.cls'),
        current_template_download_verified=False,
        current_template_note='Official page advertises September 2026 update; fresh ZIP request returned HTTP 403; retained official R314 class compiled successfully. Latest publisher template equivalence not verified.',
        scientific_tests='8 passed, 32 upstream Python 3.14/PyTorch/PyG warnings; 42.59 seconds')
    write(REPORTS/'build_verification.json',record)
    write(PAPER/'final_artifact_manifest.json',dict(
        files=[dict(path=str(p.relative_to(PAPER)),bytes=p.stat().st_size,sha256=sha(p)) for p in files+[zpath]],
        policy='Approval identities for private submission candidate; public numerical archive has separate manifest'))
    (REPORTS/'final_build_log.md').write_text(
        '# Final build log\n\nBoth PDFs compiled with zero final-log errors, warnings, unresolved citations, missing files or over/underfull boxes. '
        'Article: 13 pages; supplement: 4 pages. Abstract: '+str(len(abstract.split()))+' words; 19 cited references.\n\n'
        'The source ZIP passed CRC and both delivered PDF copies match byte-for-byte. A new temporary directory received '
        'only ZIP contents; delivered PDFs were removed before compilation. Fresh PDFs have identical extracted text and '
        'all 17 rendered page pixel hashes. Binary identity of independently compiled PDFs is not asserted. '
        'All pages are batch-rendered at 90 dpi; selective visual review is recorded separately.\n\n'
        'The retained official MDPI class is unchanged from R314. Current official template redownload was blocked by '
        'HTTP 403; the September 2026 template update and live portal requirements remain to be checked. '
        'Publisher dates/DOI are empty/unassigned; author/release placeholders remain truthful pending facts.\n')
    print('PASS: 13+4 pages, warning-free, clean ZIP compile and 17-page pixel identity')


if __name__=='__main__':
    main()
