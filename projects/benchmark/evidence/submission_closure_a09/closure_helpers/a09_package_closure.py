#!/usr/bin/env python3
"""Package reviewed A09 candidates locally; never publish or alter A08."""
import csv, hashlib, importlib.util, json, os, re, shutil, subprocess, tempfile, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
L = ROOT / 'local_storage/benchmark/a09_submission_closure'
E = ROOT / 'projects/benchmark/evidence/submission_closure_a09'
R = ROOT / 'projects/benchmark/reports/submission_closure_a09'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def inventory(paths, base):
    return [{'path': str(p.relative_to(base)), 'bytes': p.stat().st_size,
             'sha256': sha(p)} for p in sorted(paths)]

def csvwrite(p, rows):
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def zipwrite(p, base, paths):
    with zipfile.ZipFile(p, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for path in sorted(paths):
            info = zipfile.ZipInfo(str(path.relative_to(base)), (2026, 10, 10, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            z.writestr(info, path.read_bytes())
    with zipfile.ZipFile(p) as z:
        assert len(z.namelist()) == len(paths)
        assert len(set(z.namelist())) == len(paths), 'duplicate ZIP entry'
        for path in paths:
            assert z.read(str(path.relative_to(base))) == path.read_bytes()

def private_sources(review):
    outputs = []
    folder = L / 'submission_packages'; folder.mkdir(exist_ok=True)
    env = dict(os.environ, SOURCE_DATE_EPOCH='1791590400', FORCE_SOURCE_DATE='1')
    for a in review['artifacts']:
        pdf = ROOT / a['path']; source = pdf.parent; name = pdf.stem
        stage = folder / (name+'_sources')
        stage.mkdir(exist_ok=True)
        source_paths = [pdf.with_suffix('.tex'), source/'references.bib']
        if (source/'Definitions').is_dir():
            source_paths += [p for p in (source/'Definitions').rglob('*') if p.is_file()]
            source_paths += [source/'soul.sty', source/'soul-ori.sty']
        for p in source_paths:
            dest = stage/p.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
        (stage/'README_SUBMISSION.txt').write_text(
            'Private author-review submission source candidate; not for GitHub.\n'
            'Author approval and corrected public evidence identity remain pending.\n'
            'Build with a TeX Live installation including latexmk, pdfLaTeX and BibTeX.\n'
            'SOURCE_DATE_EPOCH=1791590400 FORCE_SOURCE_DATE=1 latexmk -g -pdf '
            '-interaction=nonstopmode -halt-on-error '+name+'.tex\n'
            'Official MDPI class is unmodified; its assets and missing soul dependencies '
            'are included when applicable. Converted logos come from the official EPS assets.\n')
        payload = [p for p in stage.rglob('*') if p.is_file() and p.name != 'SOURCE_INVENTORY.json']
        inv = stage/'SOURCE_INVENTORY.json'
        dump(inv, {'scope': 'all source payloads except this inventory itself',
                   'expected_pdf_sha256': a['sha256'], 'files': inventory(payload, stage)})
        payload.append(inv); package = folder/(name+'_LaTeX_submission.zip')
        zipwrite(package, stage, payload)
        clean_root = L/'source_recompile'
        clean_root.mkdir(parents=True, exist_ok=True)
        clean = Path(tempfile.mkdtemp(prefix=name+'-', dir=clean_root))
        # Rebuild from the exact ZIP in a new directory, without source-side auxiliaries.
        with zipfile.ZipFile(package) as z: z.extractall(clean)
        with (folder/(name+'_clean_compile.log')).open('w') as f:
            run = subprocess.run(['latexmk', '-g', '-pdf', '-interaction=nonstopmode',
                                  '-halt-on-error', name+'.tex'], cwd=clean, env=env,
                                 stdout=f, stderr=f)
        assert run.returncode == 0, name
        built = clean/(name+'.pdf')
        assert built.read_bytes() == pdf.read_bytes(), (name, sha(built), a['sha256'])
        outputs.append({'path': str(package.relative_to(ROOT)), 'bytes': package.stat().st_size,
                        'sha256': sha(package), 'source_payloads': len(payload),
                        'recompiled_pdf_sha256': sha(built), 'expected_pdf_sha256': a['sha256'],
                        'actual_clean_compilation': 'PASS_BYTE_IDENTICAL', 'publication': 'PRIVATE'})
    dump(L/'submission_source_packages.json', {'status': 'PASS', 'packages': outputs})
    return outputs

def private_check(value):
    if isinstance(value, dict):
        for k, v in value.items():
            assert k != 'selected_node_ids', k
            if k in {'node_ids', 'contract_ids', 'raw_scores'}:
                assert not isinstance(v, list), k
            private_check(v)
    elif isinstance(value, list):
        for v in value: private_check(v)

def main():
    review = json.loads((E/'pdf_review.json').read_text())
    assert review['status'] == 'PASS_LOCAL_PDF_REVIEW'
    assert review['source_resolved_crosswalk_rows'] == 1394
    before = json.loads((L/'a08_preservation_before.json').read_text())
    assert before == {p: sha(ROOT/p) for p in before}
    execution_path = ROOT/'projects/benchmark/evidence/a08_data_repair/execution_manifest.json'
    execution = json.loads(execution_path.read_text())
    installed = Path(importlib.util.find_spec('pygod').origin).parent
    for name, digest in execution['source_hashes'].items():
        path = installed/name.removeprefix('installed_pygod/') if name.startswith('installed_pygod/') else ROOT/name
        assert sha(path) == digest, name
    dump(E/'current_execution_source_recheck.json', {
        'status': 'PASS', 'execution_manifest_sha256': sha(execution_path),
        'source_files_checked': len(execution['source_hashes']),
        'installed_pygod_resolver': 'installed_pygod/ is a manifest namespace resolved at pygod.__file__.parent',
        'source_hashes': execution['source_hashes'], 'scientific_source_mutations': []})
    original = ROOT/'projects/benchmark/evidence/a08_public_numeric_evidence.zip'
    old = ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip'
    inv = json.loads((E/'existing_226_payload_inventory.json').read_text())
    assert sha(original) == inv['zip_sha256']
    with zipfile.ZipFile(original) as z:
        names = [n for n in z.namelist() if not n.endswith('/')]
        assert len(names) == inv['payload_count'] == 227
        assert len([n for n in names if n != 'release_manifest.json']) == 226
        for row in inv['payload']:
            data = z.read(row['path'])
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
    history = json.loads((E/'historical_identity_recheck.json').read_text())
    assert history['count'] == 351 and not history['tier_upgrade']
    with zipfile.ZipFile(old) as z:
        for row in history['records']:
            assert hashlib.sha256(z.read(row['member'])).hexdigest() == row['sha256']
    recovery = json.loads((E/'qualification_recovery.json').read_text())
    for path, digest in recovery['source_hashes'].items(): assert sha(ROOT/path) == digest
    facade = json.loads((E/'actual_reproduction/facade_checks.json').read_text())
    assert len(facade) == 2 and all(r['returncode'] == 0 for r in facade)
    packages = private_sources(review)
    for name in ['Final_PDF_Review_A09.md', 'HISTORICAL_STATUS_NOTE.md']:
        shutil.copy2(R/name, E/name)
    # Exact scientific source/config/locks remain in the existing226-payload package.
    # These helpers are editorial/evidence revisions, separately bound by the new inventory.
    helper_dir = E/'closure_helpers'; helper_dir.mkdir(exist_ok=True)
    for name in ['a09_collect_evidence.py', 'a09_summarize_evidence.py',
                 'a09_recover_qualification.py', 'a09_pdf_review.py', 'a09_package_closure.py']:
        shutil.copy2(ROOT/'projects/benchmark/scripts'/name, helper_dir/name)
    licensefile = L/'upstream/GoG_LICENSE.txt'
    shutil.copy2(licensefile, E/'GoG_upstream_LICENSE_observation.txt')
    # Index only the frozen companion payload, not later outer manifests/reports recursively.
    excluded = {'evidence_inventory.csv', 'companion_payload_manifest.json',
                'submission_closure_manifest.json', 'REVIEW_RESPONSE_A09.md'}
    files = [p for p in E.rglob('*') if p.is_file() and p.name not in excluded]
    for p in files:
        assert p.suffix.lower() in {'.json','.csv','.py','.txt','.md','.yaml','.log','.sh'}, p
        assert 'build_manuscript' not in p.name, p
        data = p.read_text()
        assert not re.search(r'(?i)\b0x[a-f0-9]{40}\b', data), p
        if p.suffix == '.json': private_check(json.loads(data))
    payload_inv = inventory(files, E)
    csvwrite(E/'evidence_inventory.csv', payload_inv)
    dump(E/'companion_payload_manifest.json', {
        'status': 'LOCAL_CANDIDATE_NOT_PUBLISHED', 'files': payload_inv,
        'inventory_self_scope': 'payload list excludes this manifest and evidence_inventory.csv; both are included in ZIP',
        'outer_documents_excluded': sorted(excluded-{'evidence_inventory.csv','companion_payload_manifest.json'}),
        'privacy_check': 'no new production IDs/scores/arrays/checkpoints/private writer/TeX/Bib/PDF',
        'scientific_run_set': 'A08 execution revision2 unchanged'})
    files += [E/'evidence_inventory.csv', E/'companion_payload_manifest.json']
    companion = ROOT/'projects/benchmark/evidence/a09_submission_closure_companion.zip'
    zipwrite(companion, E, files)
    # Existing publication guards must still classify each manuscript and private writer as ignored.
    private = [a['path'] for a in review['artifacts']] + [
        'projects/benchmark/scripts/a09_build_manuscript.py'] + [r['path'] for r in packages]
    for p in private:
        assert subprocess.run(['git','check-ignore','-q',p],cwd=ROOT).returncode == 0, p
    assert subprocess.run(['git','check-ignore','-q',str(companion.relative_to(ROOT))],cwd=ROOT).returncode == 1
    assert before == {p: sha(ROOT/p) for p in before}
    manifest = {
        'status': 'LOCAL_EDITORIAL_AND_EVIDENCE_CHECKS_PASSED_C5_PENDING',
        'scientific_run_set': 'A08 execution revision2 unchanged', 'new_production_training_runs': 0,
        'protected_A08_files': len(before), 'protected_A08_identity': 'PASS',
        'original_acceptance_sha256': sha(ROOT/'projects/benchmark/reports/a08_data_repair/FINAL_ACCEPTANCE_A08.json'),
        'packages': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                     for p in [old, original, companion]],
        'companion_payload_count': len(files), 'payload_inventory_sha256': sha(E/'evidence_inventory.csv'),
        'outer_manifest_scope': 'references the companion and pre-existing ZIPs; not included recursively inside them',
        'pdf_review_sha256': sha(E/'pdf_review.json'), 'pdf_artifacts': review['artifacts'],
        'private_submission_packages': packages,
        'C1': 'PASS_LOCAL_EVIDENCE_HANDOFF_AND_CPU_REPLAY',
        'C2': 'PASS_TECHNICAL_PROVENANCE_FINAL_MAP_AUTHOR_CONFIRMATION_IN_C5',
        'C3': 'PASS_MEASURED_QUALIFICATION_AND_HONEST_UNAVAILABLE_CAUSES',
        'C4': 'PASS_UNCHANGED_RESULTS_AND_CONDITIONAL_INTERPRETATION',
        'C5': 'PENDING_COAUTHOR_APPROVAL_AUTHORIZED_PUBLICATION_AND_RETRIEVAL',
        'C6': 'PASS_LOCAL_REVIEW_FINAL_UPDATE_REQUIRED_AFTER_C5',
        'coauthor_approval': 'NOT_RECORDED', 'public_identity': 'LOCAL_CANDIDATE_NOT_PUBLISHED',
        'submission_ready': False, 'commit_push_release_submission': 'NOT_EXECUTED',
        'journal_acceptance': 'NOT_DETERMINED', 'script_sha256': sha(Path(__file__))}
    dump(E/'submission_closure_manifest.json', manifest)
    shutil.copy2(E/'evidence_inventory.csv', R/'evidence_inventory.csv')
    shutil.copy2(E/'submission_closure_manifest.json', R/'submission_closure_manifest.json')
    print(manifest['status'],len(files),'companion files;',len(before),'A08 protected files unchanged')
    for p in manifest['packages'][-2:]+packages: print(p['path'],p['sha256'],p['bytes'])

if __name__ == '__main__': main()
