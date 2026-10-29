"""Derivative publication package: preserve all65 frozen score vectors."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from paths import ROOT,PROJECT,RESULTS,REPORTS,PUBLICATION
from common import sha,write


def main():
    source=PROJECT/'publication/315/TDS_315_numeric_candidate.zip'
    reviewed=PROJECT/'private/315_before_venue_split/submission_315/evidence/TDS_315_numeric_candidate.zip'
    candidate=PUBLICATION/'candidate'
    if candidate.exists():shutil.rmtree(candidate)
    candidate.mkdir()
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None;z.extractall(candidate)
        old={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('predictions/') and n.endswith('.csv')}
    with zipfile.ZipFile(reviewed) as z:
        baseline={n:hashlib.sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith('predictions/') and n.endswith('.csv')}
    assert len(old)==65 and old==baseline
    for rel,h in old.items():assert sha(candidate/rel)==h
    (candidate/'README_R315.md').write_bytes((candidate/'README.md').read_bytes())
    (candidate/'README.md').write_text('''# TDS 316 numeric candidate — unpublished

This local candidate accompanies the publication-only R316 revision. All65 test
score/label arrays are byte-identical to both the originally reviewed R315 numeric
candidate and the current relocated R315 candidate. No new training or scoring
was performed. The R316 manuscript describes supervised retrospective financial
graph risk screening, not unsupervised discovery or prospective deployment.

Extract this ZIP into an empty folder and run:

```bash
python replay.py . --statistics
python -m pytest tests -q
```

Replay verifies the manifest, aligned3648/107 panels, seven supplemental AP mean/SD
pairs and all three10000-replicate protocols for one original comparison. Scientific
and publication-gate tests require PyTorch/PyG in addition to the CPU replay stack.
Exact environment versions and the R315 detailed reproduction guide are included.

Reproduction levels: L0 private author manuscript builds; L1 author-local processed
artifact hashes; L2 locally verified numeric/statistical replay, with anonymous
public release still pending; L3 full independent retraining not guaranteed.
RTX3090 capacity does not retroactively reproduce frozen experiments.

Manuscripts, publisher class/styles/logos, raw transaction data, contract addresses,
checkpoints and private author records are excluded. Raw third-party data remain
under GoG CC BY-NC-SA4.0. Software, numerical derivatives and the prospective
manuscript's publication license are separate rights categories. No blanket CC BY
relicensing or rights approval is inferred. Read the included upstream license and
redistribution review; author approval is required before public deposition.

The shared repository is https://github.com/Sam-7878/dlg_gnn. Current TDS results
are under projects/tds/results/<task>. This package is local, with no verified
TDS immutable tag/release URL or DOI. A Git commit is not public-release proof.
''')
    for revision in ['315','316']:
        dest=candidate/'scripts'/('revision'+revision);dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PROJECT/f'scripts/revision{revision}/gates.py',dest/'gates.py')
        shutil.copy2(PROJECT/f'tests/test_submission{revision}.py',candidate/'tests'/f'test_submission{revision}.py')
    (candidate/'manifest.json').unlink()
    files=[dict(path=str(p.relative_to(candidate)),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(candidate.rglob('*')) if p.is_file()]
    forbidden={'.tex','.bib','.cls','.sty','.bst','.pdf','.pt','.pkl'}
    assert not [e['path'] for e in files if Path(e['path']).suffix in forbidden]
    patterns=[r'/mnt/[a-z]/',r'/home/[^/\s]+/',r'0x[0-9a-fA-F]{40}',
              r'gh[pousr]_[A-Za-z0-9]{20,}',r'sk-[A-Za-z0-9]{20,}',r'AKIA[A-Z0-9]{16}',
              r'-----BEGIN .*PRIVATE KEY-----']
    for e in files:
        content=(candidate/e['path']).read_text(errors='replace')
        assert not any(re.search(pattern,content) for pattern in patterns),e['path']
    write(candidate/'manifest.json',dict(status='LOCAL_UNPUBLISHED_CANDIDATE',files=files))
    archive=PUBLICATION/'TDS_316_numeric_release_candidate.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(candidate.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(candidate))
    scratch=Path(tempfile.mkdtemp(prefix='tds316-numeric-'))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None;z.extractall(scratch)
    replay=subprocess.run([sys.executable,'replay.py','.', '--statistics'],cwd=scratch,capture_output=True,text=True)
    assert replay.returncode==0,(replay.stdout,replay.stderr)
    tests=subprocess.run([sys.executable,'-m','pytest','tests','-q'],cwd=scratch,capture_output=True,text=True)
    assert tests.returncode==0,(tests.stdout,tests.stderr)
    assert '18 passed' in tests.stdout,tests.stdout
    manifest=dict(path=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,crc='PASS',
        source_current_315_sha256=sha(source),source_reviewed_315_sha256=sha(reviewed),
        unchanged_score_vectors=old,score_vectors_exactly_preserved=65,files=files,
        source_changes='R316 README and fail-closed publication gate code/tests only; all scientific score vectors preserved',
        status='LOCAL_COLD_REPLAY_PASS_NOT_PUBLIC',privacy_inventory_pass=True,
        release_url=None,commit=None,tag=None,published=False)
    write(PUBLICATION/'release_manifest.json',manifest)
    write(REPORTS/'numeric_cold_replay.json',dict(status='PASS_LOCAL_ONLY',archive_sha256=sha(archive),
         replay=json.loads(replay.stdout),tests=tests.stdout.replace(str(ROOT.parent),'<workspace>').replace(str(scratch),'<fresh-extraction>'),
         anonymous_release_download_verified=False,training_performed=False))
    print('PASS:65 original score vectors preserved; cold ZIP replay and18 tests; no public release claimed')


if __name__=='__main__':main()
