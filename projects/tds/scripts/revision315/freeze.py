"""Seal existing evidence before changes; never train or overwrite R314."""
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from paths import ROOT, INPUT, PRIMARY, REPORTS, RESULTS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def main():
    target = RESULTS / 'input_hash_manifest.json'
    if target.exists():
        old = json.loads(target.read_text())
        for entry in old['files']:
            # The only mutable input is the explicitly audited path repair.
            if entry['group'] != 'source_before_path_repair':
                assert sha(ROOT / entry['path']) == entry['sha256'], entry['path']
        print('Existing input seal verified; no input manifest overwritten')
        return
    expected = {'applsci_article.pdf': '1e0d06a5f34ff0fbfd2414e0f565bdce56549cc8ce8f17173cf2d154bb88bdfa',
                'supplement/supplementary.pdf': '0970ada7fbd946817ccf225dd015d60e6de6c282c34dbc800945f71f75d8f641'}
    for name, value in expected.items():
        assert sha(INPUT / name) == value, name
    files = []
    def add(path, group):
        files.append(dict(path=str(path.relative_to(ROOT)), group=group,
                          bytes=path.stat().st_size, sha256=sha(path)))
    for p in sorted(INPUT.rglob('*')):
        if not p.is_file() or 'submission_315' in p.parts:
            continue
        if p.suffix in {'.aux', '.log', '.bbl', '.blg', '.fls', '.out', '.fdb_latexmk', '.pyc'}:
            continue
        add(p, 'supplemental314' if 'runs' in p.parts else 'revision314')
    manifest = json.loads((PRIMARY / 'reproducibility_manifest.json').read_text())
    for e in manifest['entries']:
        assert sha(PRIMARY / e['path']) == e['sha256'], e['path']
    for p in sorted(PRIMARY.rglob('*')):
        if p.is_file():
            add(p, 'primary_frozen')
    for p in sorted((ROOT / 'projects/tds/scripts/revision314').rglob('*.py')):
        add(p, 'source_before_path_repair')
    graph = ROOT / 'archive/tds/data/gog_scimain_v1/graph.pt'
    assert sha(graph) == '067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd'
    add(graph, 'primary_packed_graph')
    checks = []
    for p in sorted((INPUT / 'packages').glob('*.zip')):
        with zipfile.ZipFile(p) as z:
            assert z.testzip() is None, p
            matches = {}
            for rel, value in expected.items():
                names = [n for n in z.namelist() if n.endswith('/' + rel) or n == rel]
                if not names and 'LaTeX_sources' in p.name:
                    matches[rel] = 'not embedded in source-only ZIP'
                    continue
                assert len(names) == 1, (p.name, rel, names)
                digest = hashlib.sha256(z.read(names[0])).hexdigest()
                assert digest == value, (p.name, rel)
                matches[rel] = digest
            checks.append(dict(path=str(p.relative_to(ROOT)), crc='PASS', pdf_copies=matches))
    write(target, dict(work_order='TDS-WO-315-SUBMISSION-CLOSURE', sealed_date='2026-10-29',
                       git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                       files=files, zip_checks=checks, primary_manifest_entries_verified=len(manifest['entries'])))
    legacy = ROOT / 'projects/tds/paper_applsci'
    tracked = subprocess.check_output(['git', 'ls-files', 'projects/tds/paper_applsci'], cwd=ROOT, text=True).splitlines()
    write(REPORTS / 'canonical_source_tree.json', dict(
        input_tree=str(INPUT.relative_to(ROOT)), input_tree_exists=INPUT.is_dir(),
        legacy_tree=str(legacy.relative_to(ROOT)), legacy_exists=legacy.exists(),
        legacy_tracked_files=len(tracked), legacy_role='historical Git/ZIP prefix; user moved local files before 315',
        preserve_user_move=True, input_pdf_sha256=expected,
        working_manuscript='projects/tds/paper/applsci/submission_315',
        task_results='projects/results/tds/315_applsci',
        legacy_alias='registry mapping only; no second manuscript tree created',
        input_tree_mtime_ns=INPUT.stat().st_mtime_ns,
        legacy_last_commit=subprocess.check_output(['git', 'log', '-1', '--format=%H %cI', '--', 'projects/tds/paper_applsci'], cwd=ROOT, text=True).strip()))
    print(f'Sealed {len(files)} inputs; {len(checks)} ZIPs; {len(manifest["entries"])} primary artifacts')


if __name__ == '__main__':
    main()
