"""R03 candidate tooling; R01/R02 inputs and hosted assets remain immutable."""
from __future__ import annotations
import datetime
import json
import subprocess
from pathlib import Path
from r01_common import ROOT,PROJECT,read,write,sha256

R01=PROJECT/'results/selectivestream_r01_20261011'
R02=PROJECT/'results/selectivestream_r02_20261012'
OLD=PROJECT/'paper/current/r02'
PAPER=PROJECT/'paper/current/r03'
OUT=PROJECT/'handoff/r03'
RUN='selectivestream_r03_20261020_submission_candidate'
SOURCE_SUFFIXES={'.tex','.bib','.png'}


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def artifact(path):
    p=Path(path)
    return {'path':str(p.relative_to(ROOT)),'sha256':sha256(p),'bytes':p.stat().st_size}


def sources(base):
    return sorted(p for p in Path(base).rglob('*') if p.is_file() and p.suffix in SOURCE_SUFFIXES and not any(x.startswith(('build','visual','packages')) for x in p.relative_to(base).parts))


def unchanged():
    baseline=read(OUT/'r02_final_identity_lock.json')
    for name,value in baseline['preserved_file_hashes'].items():
        if not (ROOT/name).is_file() or sha256(ROOT/name)!=value:raise ValueError('Prior evidence changed: '+name)
    return len(baseline['preserved_file_hashes'])


def command_record(command,cwd,log):
    begin=now();result=subprocess.run([str(v) for v in command],cwd=cwd,capture_output=True,text=True)
    Path(log).parent.mkdir(parents=True,exist_ok=True);Path(log).write_text(result.stdout+result.stderr)
    return {'command':[str(v) for v in command],'started_utc':begin,'finished_utc':now(),'actual_exit_code':result.returncode,'log':artifact(log)}
