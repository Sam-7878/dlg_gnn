"""Reconcile immutable R317 generations and cold-build separate private R318 candidates."""
import collections
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
REPORTS = PROJECT / 'reports/318'
OUT = PROJECT / 'results/318'
PRIVATE = PROJECT / 'private/318'
PAPER = PROJECT / 'paper/submission_318'
CURRENT = 'FINAL_CANDIDATE_318_RECONCILED'
BODY = 'a37995262f2d4ff96ad1da3016cc8bab6eb1b07347d541e418546a2d4b51baac'
BIB = '152dbcf958a12b2be8b2c96e0fe10979539d879ded963c9bac2f75033e9f63af'
VENUES = [('preprints', 'preprints', 'preprint_main.tex', 'preprint_article.pdf', 'preprint_source.zip'),
          ('applsci', 'mdpi_applsci_anomaly', 'applsci_article.tex', 'applsci_article.pdf', 'mdpi_source.zip')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def rel(path):
    return str(path.relative_to(ROOT))


def text_of(path):
    return subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)


def compile_pdf(folder, entry, logname):
    env = dict(os.environ, SOURCE_DATE_EPOCH=str(int(datetime.datetime(2026,10,30,tzinfo=datetime.timezone.utc).timestamp())), FORCE_SOURCE_DATE='1')
    run = subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', entry],
                         cwd=folder, env=env, text=True, capture_output=True)
    (REPORTS / (logname + '.txt')).write_text(run.stdout[-4500:] + run.stderr[-1500:])
    assert run.returncode == 0, run.stdout[-2500:]
    log = (folder / Path(entry).with_suffix('.log')).read_text(errors='replace')
    assert not re.search(r'undefined citations|Citation .* undefined|Reference .* undefined|LaTeX Error|Emergency stop|undefined references|multiply defined', log, re.I)
    pdf = folder / Path(entry).with_suffix('.pdf')
    info = subprocess.check_output(['pdfinfo', str(pdf)], text=True)
    return dict(pdf_sha256=sha(pdf), pages=int(re.search(r'Pages:\s+(\d+)', info).group(1)),
                overfull_boxes=len(re.findall(r'Overfull \\hbox', log)), undefined_references=0,
                bibliography_sha256=sha(folder / Path(entry).with_suffix('.bbl')) if entry != 'supplement317.tex' else None)


def metrics_in_pdf(text):
    # Decimal quantities (including all displayed AP/calibration/p-values) are independent
    # of page numbers and MDPI line counters. Integer counts are bound by identical source.
    return collections.Counter(re.findall(r'(?<![\d.])\d+\.\d{2,}(?![\d.])', text))


def main():
    for p in (REPORTS, OUT, PRIVATE, PAPER):
        p.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((PROJECT / 'reports/317/build_verification.json').read_text())
    post = json.loads((PROJECT / 'reports/317/author_confirmation_20261030/build_verification.json').read_text())
    assert baseline['shared_body_sha256'] == post['shared_body_sha256'] == BODY
    assert baseline['shared_bib_sha256'] == post['shared_bib_sha256'] == BIB
    history = PRIVATE / 'r317_history'
    history.mkdir(exist_ok=True)
    history_paths = [PROJECT / 'reports/317/build_verification.json', PROJECT / 'results/317/submission_readiness_317.json',
                     PROJECT / 'reports/317/author_confirmation_20261030/build_verification.json',
                     PROJECT / 'submission/author_records/317/approval_identity.json']
    for d, _, _, _, _ in VENUES:
        history_paths.append(PROJECT / f'submission/{d}/317/approval_identity.json')
    for p in history_paths:
        dest = history / p.relative_to(PROJECT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            assert dest.read_bytes() == p.read_bytes(), 'Historical evidence changed: ' + str(p)
        else:
            shutil.copy2(p, dest)
    history_manifest = [dict(path=rel(p), sha256=sha(p)) for p in history_paths]
    rows, builds, shared_manifests = [], [], []
    for delivery, paper_name, entry, article_name, zip_name in VENUES:
        source = PROJECT / f'submission/{delivery}/317'
        post_report = next(v for v in post['venues'] if v['venue'] == paper_name)
        early_report = next(v for v in baseline['venues'] if v['venue'] == paper_name)
        approval = json.loads((source / 'approval_identity.json').read_text())['files'][paper_name]
        supplement_name = 'preprint_supplement.pdf' if delivery == 'preprints' else 'applsci_supplement.pdf'
        for name, expected in [(article_name, post_report['local']['pdf_sha256']),
                               (supplement_name, post_report['supplement']['pdf_sha256']),
                               (zip_name, post_report['source_zip_sha256'])]:
            assert sha(source / name) == expected
        assert approval['article_sha256'] == sha(source / article_name)
        assert approval['supplement_sha256'] == sha(source / supplement_name)
        assert approval['source_zip_sha256'] == sha(source / zip_name)
        with zipfile.ZipFile(PROJECT / 'private/317/before_author_confirmation_20261030/snapshot.zip') as old:
            oldname = f'paper/submission_317/{paper_name}/{zip_name}'
            old_bytes = old.read(oldname)
            assert hashlib.sha256(old_bytes).hexdigest() == early_report['source_zip_sha256']
        with tempfile.TemporaryDirectory(prefix='tds318_early_') as tmp:
            arc = Path(tmp) / 'early.zip'
            arc.write_bytes(old_bytes)
            with zipfile.ZipFile(arc) as z, zipfile.ZipFile(source / zip_name) as later:
                assert set(z.namelist()) == set(later.namelist())
                changed = [n for n in z.namelist() if z.read(n) != later.read(n)]
                assert set(changed) <= {entry, 'declarations.tex', 'supplement/supplement317.tex'}
                classification = {n: ('author/affiliation/ORCID front matter' if n == entry else
                                       'author/funding/COI/AI declarations' if n == 'declarations.tex' else
                                       'supplement author line') for n in changed}
        folder = PAPER / paper_name
        if not folder.exists():
            with zipfile.ZipFile(source / zip_name) as z:
                z.extractall(folder)
            # Keep every word and row; extend into MDPI's existing wide left margin.
            table = folder / 'data/reusable_validity_checklist.tex'
            before = table.read_text()
            header = (r'\begingroup\footnotesize\setlength{\tabcolsep}{3pt}' + '\n'
                      r'\newlength{\TDSChecklistWidth}\setlength{\TDSChecklistWidth}{\textwidth}' + '\n'
                      r'\ifdefined\extralength' + '\n'
                      r'\addtolength{\TDSChecklistWidth}{\extralength}' + '\n'
                      r'\setlength{\LTleft}{-\extralength}\setlength{\LTright}{\fill}' + '\n'
                      r'\fi' + '\n')
            first_end = before.index('\n') + 1
            # Replacement excludes the newly introduced width definition above.
            after = header + before[first_end:].replace(r'p{.', r'>{\raggedright\arraybackslash}p{.').replace(r'\textwidth', r'\TDSChecklistWidth')
            assert after[after.index(r'\caption'): ] == before[before.index(r'\caption'): ]
            table.write_text(after)
            wrapper = folder / entry
            w = wrapper.read_text()
            w = w.replace('\n', '\n\\usepackage{array}\n', 1)
            wrapper.write_text(w)
        declaration = folder / 'declarations.tex'
        old_declaration = declaration.read_text()
        prior_clarification = (' The package retains pseudonymous sample IDs and timing fields; identifier reduction is not anonymization.'
                               ' Saved-score numerical replay differs from source-based reconstruction.'
                               ' Full training reconstruction additionally requires separately obtained upstream data,'
                               ' graph-building inputs, and the documented training environment.')
        clarification = (' Retained sample IDs and times are pseudonymous and linkable.'
                         ' Saved-score replay differs from source-based reconstruction and full training,'
                         ' which require upstream licensed inputs and the documented runtime.')
        marker = 'Raw third-party records, identifying contract mappings, manuscript sources, and checkpoints are excluded from the numerical package.'
        if prior_clarification in old_declaration:
            declaration.write_text(old_declaration.replace(prior_clarification, clarification))
        elif clarification.strip() not in old_declaration:
            assert marker in old_declaration
            declaration.write_text(old_declaration.replace(marker, marker + clarification))
        with zipfile.ZipFile(source / zip_name) as old:
            for n in old.namelist():
                if n not in [entry, 'data/reusable_validity_checklist.tex', 'declarations.tex']:
                    assert (folder / n).read_bytes() == old.read(n), n
            assert declaration.read_text() == old.read('declarations.tex').decode().replace(marker, marker + clarification)
            before = old.read('data/reusable_validity_checklist.tex').decode()
            after = (folder / 'data/reusable_validity_checklist.tex').read_text()
            assert before[before.index(r'\caption'):] == after[after.index(r'\caption'):]
        assert sha(folder / 'manuscript_body.tex') == BODY
        assert sha(folder / 'references.bib') == BIB
        shared_manifest = {p.relative_to(folder).as_posix(): sha(p) for p in folder.rglob('*')
                           if p.is_file() and (p.name in ['manuscript_body.tex','declarations.tex','references.bib'] or
                           p.relative_to(folder).parts[0] in ['data','figures'])}
        shared_manifests.append(shared_manifest)
        local = compile_pdf(folder, entry, delivery + '_article_local')
        supp = compile_pdf(folder / 'supplement', 'supplement317.tex', delivery + '_supplement_local')
        pdf = folder / Path(entry).with_suffix('.pdf')
        old_numbers, new_numbers = metrics_in_pdf(text_of(source / article_name)), metrics_in_pdf(text_of(pdf))
        assert old_numbers == new_numbers, dict(removed=old_numbers-new_numbers, added=new_numbers-old_numbers)
        dest = PROJECT / f'submission/{delivery}/318'
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(pdf, dest / article_name)
        shutil.copy2(folder / 'supplement/supplement317.pdf', dest / 'supplement.pdf')
        with zipfile.ZipFile(source / zip_name) as old, zipfile.ZipFile(dest / zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
            for n in sorted(old.namelist()):
                info = zipfile.ZipInfo(n, (2026,10,30,0,0,0))
                info.external_attr = 0o100644 << 16
                info.compress_type = zipfile.ZIP_DEFLATED
                z.writestr(info, (folder / n).read_bytes())
        # Independently build BOTH the supplied post-declaration ZIP and current ZIP.
        for phase, zip_path, article_path, supplement_path in [
            ('post_declaration', source / zip_name, source / article_name, source / supplement_name),
            ('current', dest / zip_name, dest / article_name, dest / 'supplement.pdf')]:
            with tempfile.TemporaryDirectory(prefix='tds318_source_') as tmp:
                cold = Path(tmp)
                with zipfile.ZipFile(zip_path) as z:
                    z.extractall(cold)
                a = compile_pdf(cold, entry, delivery + '_' + phase + '_article_cold')
                s = compile_pdf(cold / 'supplement', 'supplement317.tex', delivery + '_' + phase + '_supplement_cold')
                assert text_of(article_path) == text_of(cold / Path(entry).with_suffix('.pdf'))
                assert text_of(supplement_path) == text_of(cold / 'supplement/supplement317.pdf')
                builds.append(dict(venue=delivery, phase=phase, source_zip_sha256=sha(zip_path), article=a,
                                   supplement=s, extracted_text_identical=True,
                                   article_pdf_byte_identical=sha(article_path) == a['pdf_sha256'],
                                   supplement_pdf_byte_identical=sha(supplement_path) == s['pdf_sha256']))
        artifacts = {}
        for name in [article_name, 'supplement.pdf', zip_name]:
            artifacts[name] = dict(path=rel(dest / name), sha256=sha(dest / name))
        row = dict(venue=delivery, earlier=early_report, post_declaration=post_report,
                   source_changes_early_to_post=classification, current=artifacts,
                   current_article_pages=local['pages'], current_supplement_pages=supp['pages'],
                   layout_only_changes=[entry, 'data/reusable_validity_checklist.tex'],
                   disclosure_changes=['declarations.tex: pseudonymous linkage and replay/full-training scope clarified; author declarations unchanged'],
                   scientific_decimal_quantity_multiset_identical=True)
        rows.append(row)
        metadata = dict(candidate_id=CURRENT, title='Auditing Sampling-Associated Prevalence Shift in Supervised Blockchain Graph Anomaly Screening',
                        venue='Preprints.org' if delivery == 'preprints' else 'Applied Sciences',
                        special_issue=None if delivery == 'preprints' else 'Machine Learning and Its Application for Anomaly Detection',
                        special_issue_url=None if delivery == 'preprints' else 'https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt',
                        submission_status='NOT_SUBMITTED', posting_status='NOT_POSTED', preprint_doi=None,
                        special_issue_portal_selection_verified=False, permission_to_submit=False,
                        artifact_identities=artifacts, body_sha256=BODY, bibliography_sha256=BIB)
        write(dest / 'metadata.json', metadata)
        (dest / 'README.md').write_text('# R318 ' + metadata['venue'] + ' candidate\n\nPrivate manuscript files; current hashes are in `metadata.json` and the canonical\n`projects/tds/reports/318/final_candidate_ledger.md`. R317 remains unchanged.\n\nStatus: NOT_SUBMITTED / NOT_POSTED. Await exact-file author approval and rights/release closure.\n')
    assert shared_manifests[0] == shared_manifests[1]
    common = PAPER / 'shared'
    common.mkdir(exist_ok=True)
    for name in shared_manifests[0]:
        dest = common / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PAPER / 'preprints' / name, dest)
    shared_files = [dict(path=rel(common / n), sha256=s) for n,s in shared_manifests[0].items()]
    write(OUT / 'candidate_identity.json', dict(current_final_candidate=CURRENT,
          earlier_id='EARLIER_317_SCIENTIFIC_BUILD', post_declaration_id='FINAL_CANDIDATE_317_POST_DECLARATION',
          current_final_candidate_count=1, original_scientific_body_sha256=BODY, original_bibliography_sha256=BIB,
          original_317_science_changed=False, shared_content_identical=True, shared_files=shared_files,
          histories=history_manifest, venues=rows))
    write(OUT / 'build_verification.json', dict(status='PASS', fresh_builds=builds,
          all_current_and_post_declaration_zip_text_matches=True, detector_fits=0))
    checks = [a for row in rows for a in row['current'].values()] + shared_files
    sums = ''.join(e['sha256'] + '  ' + e['path'] + '\n' for e in sorted(checks,key=lambda e:e['path']))
    (PROJECT / 'publication/318/SHA256SUMS_FINAL.txt').write_text(sums)
    (REPORTS / 'SHA256SUMS_FINAL.txt').write_text(sums)
    # QA renders stay private, away from numerical evidence and paper source trees.
    qa = PRIVATE / 'qa'
    qa.mkdir(exist_ok=True)
    for delivery, _, _, article, _ in VENUES:
        pdf = PROJECT / f'submission/{delivery}/318/{article}'
        pages = text_of(pdf).split('\f')
        selected = {1} | {i+1 for i,t in enumerate(pages) if 'Reusable validity-first checklist.' in t or 'Figure 2.' in t}
        for page in sorted(selected):
            subprocess.run(['pdftoppm','-f',str(page),'-l',str(page),'-r','110','-singlefile','-png',str(pdf),str(qa / f'{delivery}_{page}')], check=True, capture_output=True)
    print('R318 candidates reconciled and four article/supplement pairs cold-built PASS')


if __name__ == '__main__':
    main()
