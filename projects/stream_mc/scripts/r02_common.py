"""R02 sidecars: original R01 files are read-only inputs, never outputs."""
from __future__ import annotations
import json
import platform
import subprocess
from pathlib import Path
from r01_common import sha256,read,write

ROOT=Path(__file__).resolve().parents[3]
PROJECT=ROOT/'projects/stream_mc'
R01=PROJECT/'results/selectivestream_r01_20261011'
R02=PROJECT/'results/selectivestream_r02_20261012'
VERIFY=R02/'verification'
PAPER=PROJECT/'paper/current/r02'
RUN='selectivestream_r02_20261012_verification'


def environment():
    import importlib.metadata as metadata
    names=['torch','torch-geometric','pygod','numpy','pandas','scipy','scikit-learn','pyarrow']
    versions={}
    for name in names:
        try:versions[name]=metadata.version(name)
        except metadata.PackageNotFoundError:versions[name]='not installed in primary site-packages; project-local optional dependency may apply'
    return {'verification_id':RUN,'python':platform.python_version(),'platform':platform.platform(),
        'os_release':Path('/etc/os-release').read_text(),'versions':versions,
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'scope':'same operator/host implementation-side verification, not external reviewer approval'}


def csv_write(path,rows):
    import csv
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def artifact(path):
    path=Path(path)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size}


def original_unchanged():
    manifest=read(R01/'release_manifest.json');mismatches=[]
    for name,expected in manifest['files'].items():
        path=R01/name
        if not path.is_file() or sha256(path)!=expected:mismatches.append(name)
    if mismatches:raise ValueError('original R01 drift: '+repr(mismatches))
    return len(manifest['files'])
