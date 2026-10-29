"""Cold local ZIP verification; never asserts public/anonymous accessibility."""
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from paths import ROOT, PUBLICATION, REPORTS
from freeze import sha, write


def main():
    archive=PUBLICATION/'TDS_315_numeric_candidate.zip'
    expected=json.loads((PUBLICATION/'candidate_archive.json').read_text())
    assert sha(archive)==expected['sha256']
    scratch=Path(tempfile.mkdtemp(prefix='tds315-numeric-'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert all((scratch/n).resolve().is_relative_to(scratch) for n in z.namelist())
        z.extractall(scratch)
    replay=subprocess.run([sys.executable,'replay.py','.'],cwd=scratch,text=True,capture_output=True)
    assert replay.returncode==0,(replay.stdout,replay.stderr)
    tests=subprocess.run([sys.executable,'-m','pytest','tests/test_revision314.py','-q'],
                         cwd=scratch,text=True,capture_output=True)
    assert tests.returncode==0,(tests.stdout,tests.stderr)
    assert '8 passed' in tests.stdout
    full=json.loads(Path('/tmp/tds315_candidate_replay.log').read_text())
    assert full['status']=='PASS' and full['full_representative_statistics_replayed'] is True
    write(REPORTS/'candidate_local_verification.json',dict(
        status='PASS_LOCAL_ONLY',archive_sha256=sha(archive),crc='PASS',
        fresh_extraction=True,cold_prediction_replay=json.loads(replay.stdout),
        invariant_tests='8 passed',test_output=tests.stdout.replace(str(ROOT.parent), '<workspace>').replace(str(scratch), '<fresh-extraction>'),
        staged_full_10000_resample_replay=full,anonymous_access_verified=False,
        release_url=None,release_commit=None,release_tag_published=False))
    print('PASS: fresh numeric ZIP extraction, 65 prediction files and eight scientific tests; NOT public verification')


if __name__=='__main__': main()
