#!/usr/bin/env python3
"""Execute an actual prospective public export, without commit or push.

Run after G7/G8/G9. Exported files reflect the current index/eligible working
tree, not a claim that a corresponding immutable Git revision was published.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
E = ROOT / 'projects/benchmark/evidence/a08_data_repair'
L = ROOT / 'local_storage/benchmark/a08_data_repair'
R = ROOT / 'projects/benchmark/reports/a08_data_repair'
PRIVATE_SUFFIXES = {'.tex', '.bib', '.cls', '.sty', '.pdf', '.pt', '.pkl', '.pickle', '.npz', '.npy', '.whl'}
HISTORICAL_ZIP_SHA256 = 'b1caab1124404eb6ac4dcd909007c69ceab2410e02f84363d53b0f8465b0c7f5'
HISTORICAL_PLOT = 'evaluation/benchmark/v2/paper_ready_a05/figure_memory_measured_a05.pdf'
HISTORICAL_PLOT_SHA256 = 'b41636c732327390561c969a080877db6f220b75a882b6d8dae854ffcfbd7548'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def private_path(name):
    path = PurePosixPath(name)
    if name in {f'projects/{project}/paper/README.md' for project in ('dlg_gnn','benchmark','stream_mc','tds')}:
        return False  # Public boundary explanations, not manuscript bytes.
    if name.startswith('projects/dlg_gnn/paper/published_v1/') and (path.suffix.lower()=='.png' or path.name=='SOURCE_COPY_MANIFEST.json'):
        return False  # Preserved already-public foundation-paper figures/hash metadata.
    return (path.suffix.lower() in PRIVATE_SUFFIXES
            or any(p in {'local_storage', 'paper', 'manuscript_base_a06'} for p in path.parts)
            or 'build_manuscript' in name or 'tracked-working-tree.patch' in name)


def inspect_archive(data, prefix, depth=0):
    if depth > 4:
        raise ValueError('nested archive depth exceeds reviewed boundary: ' + prefix)
    issues = []
    preserved = depth == 0 and prefix == 'projects/benchmark/evidence/public_numeric_evidence.zip' and digest(data) == HISTORICAL_ZIP_SHA256
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('duplicate archive entry: ' + prefix)
        for item in archive.infolist():
            if item.is_dir():
                continue
            path = PurePosixPath(item.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in item.filename:
                raise ValueError('unsafe nested archive entry: ' + prefix + '::' + item.filename)
            full = prefix + '::' + item.filename
            # Already public, immutable historical numeric vectors/one scientific
            # plot are preserved. This exception cannot admit new raw scores or
            # a renamed manuscript: the whole historical ZIP identity is fixed.
            allowed = preserved and (
                (item.filename == HISTORICAL_PLOT and digest(archive.read(item)) == HISTORICAL_PLOT_SHA256)
                or (path.suffix.lower() == '.npy' and any(part in {'anomalydae_scores', 'bitcoinotc_scores'} for part in path.parts)))
            if private_path(item.filename) and not allowed:
                issues.append(full)
            if path.suffix.lower() == '.zip':
                issues += inspect_archive(archive.read(item), full, depth + 1)
    return issues


def candidates():
    command = ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard']
    names = subprocess.check_output(command, cwd=ROOT).decode().split('\0')
    return sorted({n for n in names if n and (ROOT / n).is_file()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--export-only', action='store_true', help='Snapshot and scan; does not close G10')
    args = ap.parse_args()
    for number in (7, 8, 9):
        path = E / 'audit' / f'G{number}.json'
        if not path.is_file() or json.loads(path.read_text()).get('status') != 'PASS':
            raise ValueError(f'actual G{number} required before final public export')
    subprocess.run([sys.executable, str(ROOT / 'projects/benchmark/scripts/a08_public_evidence.py'), '--build'],
                   cwd=ROOT, check=True)
    payload = candidates()
    private = []
    archives = []
    for name in payload:
        if private_path(name):
            private.append(name)
        if PurePosixPath(name).suffix.lower() == '.zip':
            matches = inspect_archive((ROOT / name).read_bytes(), name)
            private += matches
            archives.append({'path': name, 'sha256': digest((ROOT / name).read_bytes()), 'private_matches': matches})
    boundary = {'status': 'PASS' if not private else 'FAIL', 'scope': 'actual index/eligible working-tree and nested ZIP payloads',
                'index_only_private_matches': [n for n in subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0') if n and private_path(n)],
                'private_payload_matches': private, 'archives': archives,
                'raw_identifiers_scores_checkpoints': 'local-only under declared source-access constraints',
                'preserved_historical_public_numeric_exception': {'archive_sha256': HISTORICAL_ZIP_SHA256, 'scope': 'previously public numeric score vectors and one measured-memory supporting plot, unchanged bytes; no A08 private scores'},
                'historical_git_content': 'not removed from history; no history rewrite or push',
                'generic_scientific_templates': 'existing shared latex_exporter API is not an A08 manuscript writer; current unsubmitted writer excluded',
                'scanner_sha256': digest(Path(__file__).read_bytes())}
    dump(E / 'audit/public_private_boundary.json', boundary)
    if private or boundary['index_only_private_matches']:
        raise ValueError('private payload exists in public candidates/index: ' + '; '.join(private))
    # Include the just-created boundary report in this snapshot. Later clean-export
    # audit/G10 reports describe the snapshot; no recursive self-hash is claimed.
    payload = candidates()
    target = Path(tempfile.mkdtemp(prefix='dlg-a08-public-', dir='/tmp'))
    entries = []
    for name in payload:
        source = ROOT / name
        if source.is_symlink():
            raise ValueError('public symlink needs an explicit audit: ' + name)
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        entries.append({'path': name, 'bytes': dest.stat().st_size, 'sha256': digest(dest.read_bytes())})
    manifest = {'schema_version': 1, 'scope': 'prospective public snapshot, not a published commit',
                'base_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'file_count': len(entries), 'files': entries}
    dump(E / 'public_payload_manifest.json', manifest)
    if args.export_only:
        print(json.dumps({'status': 'EXPORTED_NOT_VERIFIED', 'checkout': str(target), 'file_count': len(entries)}))
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(target / 'src'))
    runs = []
    for project in ('dlg_gnn', 'benchmark', 'stream_mc', 'tds'):
        command = [sys.executable, 'scripts/reproduce_project.py', '--project', project, '--mode', 'verify']
        start = time.monotonic()
        result = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True)
        runs.append({'command': command, 'exit': result.returncode, 'seconds': time.monotonic()-start,
                     'stdout': result.stdout, 'stderr': result.stderr})
        if result.returncode or (project != 'benchmark' and 'SKIPPED' in result.stdout):
            dump(E / 'audit/clean_export_verify_A08.json', {'status': 'FAIL', 'checkout': str(target), 'runs': runs})
            raise ValueError('required clean-export verification failed/skipped: ' + project)
    numeric = ['number_registry.json', 'statistics/statistics_s1_s4.json']
    numeric += [p.relative_to(E).as_posix() for directory in ('tables', 'statistics') for p in sorted((E / directory).glob('*.csv'))]
    baseline = {name: digest((target / 'projects/benchmark/evidence/a08_data_repair' / name).read_bytes()) for name in numeric}
    command = [sys.executable, 'scripts/reproduce_project.py', '--project', 'benchmark', '--mode', 'tables']
    start = time.monotonic()
    result = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True)
    runs.append({'command': command, 'exit': result.returncode, 'seconds': time.monotonic()-start,
                 'stdout': result.stdout, 'stderr': result.stderr})
    unchanged = all(digest((target / 'projects/benchmark/evidence/a08_data_repair' / name).read_bytes()) == sha for name, sha in baseline.items())
    if result.returncode or not unchanged:
        dump(E / 'audit/clean_export_verify_A08.json', {'status': 'FAIL', 'checkout': str(target), 'runs': runs, 'numeric_replay_identical': unchanged})
        raise ValueError('public numeric replay failed or changed approved table bytes')
    command = [sys.executable, 'scripts/reproduce_project.py', '--project', 'benchmark', '--mode', 'paper']
    result = subprocess.run(command, cwd=target, env=env, capture_output=True, text=True)
    guarded = result.returncode != 0 and 'intentionally withheld' in result.stderr
    runs.append({'command': command, 'exit': result.returncode, 'expected_private_guard': guarded,
                 'stdout': result.stdout, 'stderr': result.stderr})
    if not guarded:
        raise ValueError('public paper private-input guard did not reject unavailable manuscript')
    # Author-local compilation is a separate actual invocation on the same registry.
    before = json.loads((L / 'paper_build_manifest.json').read_text())
    command = [sys.executable, 'scripts/reproduce_project.py', '--project', 'benchmark', '--mode', 'paper']
    with (L / 'author_local_reproduction.log').open('w') as log:
        local = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log, text=True)
    after = json.loads((L / 'paper_build_manifest.json').read_text())
    if local.returncode or before != after:
        raise ValueError('author-local paper replay failed or changed reviewed PDF/build identity; redo G8/G9')
    result = {'status': 'PASS', 'checkout': str(target), 'method': 'empty prospective public export with exact-byte copies',
              'base_head': manifest['base_head'], 'public_payload_manifest_sha256': digest((E / 'public_payload_manifest.json').read_bytes()),
              'runs': runs, 'numeric_replay_identical': True, 'numeric_replay_hashes': baseline,
              'private_paper_guard': 'PASS', 'author_local_paper_replay': 'PASS',
              'author_local_log_sha256': digest((L / 'author_local_reproduction.log').read_bytes()),
              'reviewed_build_identity_unchanged': True, 'source_newline_policy': '.gitattributes preserves exact source/evidence bytes; no normalization performed'}
    dump(E / 'audit/clean_export_verify_A08.json', result)
    (R / 'PUBLIC_PRIVATE_BOUNDARY_A08.md').write_text('# A08 public/private boundary and actual reproduction\n\n'
        'Actual current public numeric ZIP, index/working-tree candidates and nested archives contain no unsubmitted TeX/Bib/class/style/manuscript-PDF or private A08 writers, scores, checkpoints or raw node identifiers. The unchanged historical public ZIP retains its previously public numeric score vectors and measured-memory supporting plot under a whole-archive hash exception. '
        'Four project integrity/scientific facades were executed in an empty public export with dependencies available; no required scientific smoke was skipped. '
        'Current A08 tables/statistics reproduce identical bytes. Public paper mode rejects missing private input; the author-local paper invocation recompiles the same reviewed artifact identities. '
        'This is a prospective local package, not a public Git commit, push, immutable release, author approval or submission. Historical Git content was not erased. '
        'Shared family regression scope remains narrower than complete Stream/TDS empirical qualification.\n')
    dump(E / 'audit/G10.json', {'status': 'PASS', 'clean_export_verify_sha256': digest((E / 'audit/clean_export_verify_A08.json').read_bytes()),
         'public_private_boundary_sha256': digest((E / 'audit/public_private_boundary.json').read_bytes()),
         'final_checker_tests_sha256': digest((E / 'audit/final_checker_tests.json').read_bytes()),
         'public_payload_manifest_sha256': digest((E / 'public_payload_manifest.json').read_bytes()),
         'public_release_identity_sha256': digest((ROOT / 'projects/benchmark/evidence/a08_public_release.json').read_bytes()),
         'scope': 'actual prospective public/private reproduction and shared fixture regression; no publication authorization'})
    print('G10 public export and author-local reproduction PASS; final acceptance still requires the full checker')


if __name__ == '__main__':
    main()
