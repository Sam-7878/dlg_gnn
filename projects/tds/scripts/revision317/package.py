"""Build an identifier-reduced local numeric candidate, never publish it."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from common import ROOT,PROJECT,OUT,sha,write


def main():
    target=PROJECT/'publication/317';target.mkdir(exist_ok=True)
    stage=target/'candidate';shutil.rmtree(stage,ignore_errors=True);stage.mkdir()
    # Explicit public numerical output allowlist, no run directories or private sources.
    for p in sorted(OUT.glob('*')):
        if p.is_file() and p.suffix in {'.csv','.json','.tex'} and p.name not in ['submission_readiness_317.json']:
            shutil.copy2(p,stage/p.name)
    for folder in ['predictions','figures','tables']:shutil.copytree(OUT/folder,stage/folder)
    shutil.copy2(Path(__file__).with_name('replay.py'),stage/'replay.py')
    # Include auditable reconstruction/training code as a repository-shaped snapshot.
    # Raw data/weights remain external; these sources do not turn L2 into an L3 claim.
    for revision in ['revision314','revision317']:
        source=PROJECT/'scripts'/revision
        destcode=stage/f'source/projects/tds/scripts/{revision}';destcode.mkdir(parents=True)
        for p in source.glob('*.py'):shutil.copy2(p,destcode/p.name)
    for revision in ['revision315','revision316']:
        destcode=stage/f'source/projects/tds/scripts/{revision}';destcode.mkdir(parents=True)
        shutil.copy2(PROJECT/f'scripts/{revision}/gates.py',destcode/'gates.py')
    testdir=stage/'source/projects/tds/tests';testdir.mkdir(parents=True)
    for name in ['test_revision314.py','test_submission315.py','test_submission316.py','test_revision317.py']:
        shutil.copy2(PROJECT/'tests'/name,testdir/name)
    (stage/'README.md').write_text('''# TDS317 numerical evidence — local release candidate

Run `python replay.py` with NumPy, pandas, SciPy and scikit-learn. This checks all
110 score arrays and 95 validation-only calibrators without manuscripts, weights
or upstream raw data. The arrays consist of 65 preserved original results and
45 new fits; distinct original and first+7d cohorts must not be paired.

This is not an approved public release. No TDS immutable tag or anonymous download
has been verified. Identifier reduction is not a formal anonymization guarantee.
Upstream GoG terms, including CC BY-NC-SA 4.0, continue to apply to relevant
derivatives. No claim of unrestricted MIT licensing for these data is made.
Raw transactions, contract mappings, model weights, author records, and private
manuscript/source ZIPs are excluded. Codebase: https://github.com/Sam-7878/dlg_gnn

`source/` supplies reconstruction/training helpers and tests in repository layout.
Full reconstruction still needs raw data and the documented training environment.
The old18 unit regressions run cold with PyTorch/PyG installed; the new317 integration
tests require the author-local full artifacts and are run in the original checkout.
''')
    (stage/'LICENSE_NOTICE.md').write_text('''# Rights separation — not publication authorization

Repository software follows the applicable repository/third-party code licenses.
Numerical predictions, source-derived lineage and graph/data derivatives are not
automatically MIT-licensed by inclusion beside code. Upstream GoG data retain their
original access and CC BY-NC-SA4.0 terms. Derivative redistribution has not received
the required author/rights approval. This local candidate must not be advertised
as an unrestricted approved public release. Raw data, protected contract mappings,
model weights, paper sources and private author records are excluded.
''')
    if (ROOT/'LICENSE').is_file():shutil.copy2(ROOT/'LICENSE',stage/'REPOSITORY_CODE_LICENSE.txt')
    for e in json.loads((stage/'source_manifest.json').read_text())['files']:
        assert sha(stage/'source'/e['path'])==e['sha256'],'Source changed while packaging: '+e['path']
    entries=[dict(path=p.relative_to(stage).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file()]
    write(stage/'manifest.json',dict(files=entries,n_test_vectors=110,anonymous_public_release=False,rights_verified=False))
    dest=target/'TDS_317_numeric_release_candidate.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(stage.rglob('*')):
            if not p.is_file():continue
            info=zipfile.ZipInfo(p.relative_to(stage).as_posix(),date_time=(2026,10,30,0,0,0));info.external_attr=0o100644<<16;info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,p.read_bytes())
    with tempfile.TemporaryDirectory(prefix='tds317_cold_') as temp:
        cold=Path(temp)
        with zipfile.ZipFile(dest) as z:z.extractall(cold)
        result=subprocess.run([sys.executable,str(cold/'replay.py')],cwd=cold,text=True,capture_output=True,check=True)
        replay=json.loads(result.stdout)
        tests=[str(cold/'source/projects/tds/tests'/name) for name in ['test_revision314.py','test_submission315.py','test_submission316.py']]
        env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='',PYTORCH_NVML_BASED_CUDA_CHECK='1',MPLCONFIGDIR=str(cold/'mpl-cache'))
        regression=subprocess.run([sys.executable,'-m','pytest',*tests,'-q'],cwd=cold/'source',env=env,text=True,capture_output=True,check=True)
        replay['cold_regression18_passed']=True
        replay['cold_regression_output']=regression.stdout.strip()
    write(PROJECT/'reports/317/numeric_cold_replay.json',dict(**replay,archive_sha256=sha(dest),archive_bytes=dest.stat().st_size,
        extracted_to_fresh_directory=True,anonymous_public_download_verified=False))
    write(target/'release_manifest.json',dict(path=str(dest.relative_to(ROOT)),sha256=sha(dest),bytes=dest.stat().st_size,
        publicly_released=False,approved_for_public_release=False,tag=None,commit=None,asset_url=None))
    print(json.dumps(replay,indent=2))


if __name__=='__main__':main()
