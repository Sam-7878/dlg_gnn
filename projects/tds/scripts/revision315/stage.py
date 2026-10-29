"""Build an unpublished numeric candidate with explicit source transformations."""
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from paths import ROOT, INPUT, PRIMARY, RESULTS, REPORTS, PUBLICATION
from freeze import sha, write


def main():
    candidate = PUBLICATION / 'candidate'
    if candidate.exists():
        shutil.rmtree(candidate)
    candidate.mkdir()
    def copy(src, rel):
        dst = candidate / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    identity = pd.read_csv(PRIMARY / 'raw_predictions/proposed/seed7_test.csv')[['event_id', 'timestamp', 'label']]
    lineage = []
    numeric = RESULTS / 'predictions'
    for panel, paths in [('primary', sorted((PRIMARY / 'raw_predictions').glob('*/seed*_test.csv'))),
                         ('supplemental', sorted((INPUT / 'runs').glob('*/seed*/test.csv')))]:
        for src in paths:
            f = pd.read_csv(src)
            assert f[['event_id', 'timestamp', 'label']].equals(identity), src
            arm = src.parent.name if panel == 'primary' else src.parent.parent.name
            seed = int(src.stem.split('_')[0][4:] if panel == 'primary' else src.parent.name[4:])
            rel = f'predictions/{panel}/{arm}/seed{seed}.csv'
            out = pd.DataFrame(dict(row_id=np.arange(len(f)), label=f.label, p_mean=f.p_mean))
            dest = RESULTS / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(dest, index=False, float_format='%.17g')
            # Verify the serialized derivative, not just the input DataFrame.
            check = pd.read_csv(dest, float_precision='round_trip')
            assert np.array_equal(check.p_mean, f.p_mean), src
            copy(dest, rel)
            lineage.append(dict(panel=panel, arm=arm, seed=seed, path=rel,
                source_path=str(src.relative_to(ROOT)), source_sha256=sha(src), sha256=sha(dest),
                ap=average_precision_score(f.label, f.p_mean), n=3648, positive=107))
    assert len(lineage) == 65
    write(candidate / 'prediction_lineage.json', dict(
        transformation='Replace contract/time/event identifiers by frozen zero-based row_id; preserve labels, score floats and order',
        aligned_original_identity_sha256=hashlib.sha256(identity.to_csv(index=False).encode()).hexdigest(), files=lineage))
    for p in sorted((INPUT / 'data').glob('*.csv')):
        if p.name == 'strict_time_source_audit.csv':
            continue  # Per-contract paths/identifiers stay private.
        copy(p, 'data/' + p.name)
    for name in ['statistical_comparisons_corrected.csv', 'holm_adjusted_statistics.csv',
                 'control_statistical_comparisons.csv', 'convergence_audit.csv',
                 'reliability_bin_support.csv', 'per_seed_results.csv', 'ap_tie_audit.csv',
                 'protocol_freeze.json', 'summary_scaler.json']:
        copy(INPUT / 'supplement' / name, 'statistics/' + name)
    transformations = []
    source = ROOT / 'projects/tds/scripts/revision314'
    for name in ['common.py', 'train.py', 'historical_gin.py', 'historical_models.py',
                 'historical_metrics.py', 'historical_statistics.py', 'statistical_audit.py',
                 'control_statistics.py', 'strict_time.py', 'audit.py']:
        copy(source / name, 'scientific/' + name)
    # Portable closure bundle has no repository structure or private manuscript inputs.
    p = candidate / 'scientific/common.py'
    before = sha(p)
    code = p.read_text().replace("ROOT = Path(__file__).resolve().parents[4]", "ROOT = Path(os.environ.get('TDS_REPO_ROOT', '.')).resolve()")
    code = code.replace("OUT = PROJECT / 'results/314_applsci'", "OUT = Path(os.environ.get('TDS_OUTPUT_ROOT', 'tds_generated')).resolve()")
    p.write_text(code)
    transformations.append(dict(path='scientific/common.py', source_sha256=before, staged_sha256=sha(p),
                                change='Explicit TDS_REPO_ROOT and separate output directory; never write frozen manuscript inputs'))
    # Include metric-only operating helper, excluding the manuscript/table writer.
    original = (source / 'outputs.py').read_text()
    fn = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'operating')
    lines = original.splitlines(keepends=True)
    p = candidate / 'scientific/outputs.py'
    p.write_text('"""Numeric operating-budget helper extracted from R314; manuscript writer excluded."""\nimport math\nimport numpy as np\n\n' + ''.join(lines[fn.lineno-1:fn.end_lineno]))
    transformations.append(dict(path='scientific/outputs.py', source_sha256=sha(source/'outputs.py'), staged_sha256=sha(p),
                                change='Only operating() retained, verbatim function body; no TeX or narrative writers'))
    for src in sorted((source / 'original_source').glob('*.py')):
        copy(src, 'original_source/' + src.name)
        p = candidate / 'original_source' / src.name
        if src.name == 'build_gog_chronological_dataset.py':
            before = sha(p)
            p.write_text(p.read_text().replace('default=Path("/mnt/d/_Work/_data/GoG_sci_v2")', 'required=True'))
            transformations.append(dict(path=str(p.relative_to(candidate)), source_sha256=before, staged_sha256=sha(p),
                change='Require --source-root instead of an author workstation default; scientific algorithm unchanged'))
    copy(ROOT / 'projects/tds/tests/test_revision314.py', 'tests/test_revision314.py')
    p = candidate / 'tests/test_revision314.py'
    p.write_text(p.read_text().replace("'scripts/revision314'", "'scientific'"))
    transformations.append(dict(path='tests/test_revision314.py', source_sha256=sha(ROOT/'projects/tds/tests/test_revision314.py'),
                                staged_sha256=sha(p), change='Import path only; all eight invariant test bodies unchanged'))
    copy(Path(__file__).with_name('replay.py'), 'replay.py')
    copy(INPUT / 'licenses/GoG_LICENSE.txt', 'licenses/GoG_CC_BY_NC_SA_4.txt')
    copy(ROOT / 'LICENSE', 'licenses/repository_LICENSE.txt')
    for name in ['README.md', 'dataset_protocol.md', 'reproduction_levels.md', 'license_and_redistribution_review.md', 'figure_data_map.csv']:
        copy(PUBLICATION / name, name)
    packages = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'torch-geometric', 'pytest']
    env = dict(python=platform.python_version(), platform=platform.platform(),
               packages={x: importlib.metadata.version(x) for x in packages},
               policy='Executed environment snapshot, not a wheel-hashed cross-platform installer; CPU replay needs numpy/pandas/sklearn only')
    write(candidate / 'environment.json', env)
    (candidate / 'requirements-replay.txt').write_text('\n'.join(f'{x}=={env["packages"][x]}' for x in ['numpy', 'pandas', 'scikit-learn'])+'\n')
    copy(INPUT / 'supplement/runtime_lock.json', 'environment_r314.json')
    data_contract = json.loads((INPUT / 'reports/data_contract.json').read_text())
    data_contract['label_file'] = 'upstream Token Data/labels.csv'
    write(candidate / 'dataset_contract.json', data_contract)
    # Full original manifests can contain host paths or identifiers. Keep hash-only identities here.
    input_seal = json.loads((RESULTS/'input_hash_manifest.json').read_text())
    write(candidate / 'frozen_artifact_hashes.json', dict(files=[e for e in input_seal['files'] if e['group'] in ['primary_frozen','primary_packed_graph']]))
    write(candidate / 'source_transformations.json', dict(entries=transformations, original_scientific_outputs_unchanged=True))
    prohibited = []
    for p in sorted(candidate.rglob('*')):
        if not p.is_file(): continue
        if p.suffix in {'.tex', '.bib', '.pdf', '.pt', '.pkl', '.npz', '.png', '.svg', '.zip'}:
            prohibited.append(str(p.relative_to(candidate)))
        text = p.read_text(errors='replace')
        patterns = [r'/mnt/[a-z]/', r'/home/[^/\s]+/', r'[A-Z]:\\', r'0x[0-9a-fA-F]{40}',
                    r'gh[pousr]_[A-Za-z0-9]{20,}', r'sk-[A-Za-z0-9]{20,}', r'AKIA[A-Z0-9]{16}',
                    r'-----BEGIN .*PRIVATE KEY-----', r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}']
        for pattern in patterns:
            if re.search(pattern, text):
                prohibited.append(dict(path=str(p.relative_to(candidate)), pattern=pattern))
    assert not prohibited, prohibited
    entries = [dict(path=str(p.relative_to(candidate)), bytes=p.stat().st_size, sha256=sha(p))
               for p in sorted(candidate.rglob('*')) if p.is_file()]
    write(candidate / 'manifest.json', dict(status='LOCAL_UNPUBLISHED_CANDIDATE', files=entries))
    write(RESULTS / 'public_prediction_lineage.json', json.loads((candidate/'prediction_lineage.json').read_text()))
    write(PUBLICATION / 'candidate_manifest.json', dict(files=entries, manifest_sha256=sha(candidate/'manifest.json'),
        candidate_release_tag='tds-315-submission-2026-10-29', release_url=None, release_commit=None,
        status='LOCAL_STAGING_ONLY', manuscript_sources_included=False, raw_transactions_included=False,
        checkpoints_included=False, upstream_license='CC BY-NC-SA 4.0; no CC BY relicensing inferred',
        privacy_scan=dict(status='PASS', scanned_files=len(entries), findings=[])))
    archive = PUBLICATION / 'TDS_315_numeric_candidate.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(candidate.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(candidate))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    write(PUBLICATION / 'candidate_archive.json', dict(path=archive.name, bytes=archive.stat().st_size, sha256=sha(archive), crc='PASS', published=False))
    print(f'Local candidate: {len(entries)} files, 65 prediction arrays, CRC/privacy PASS')


if __name__ == '__main__':
    main()
