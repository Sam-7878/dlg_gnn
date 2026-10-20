"""Seal every local revision artifact and verify the handoff ZIP membership.

Uses only the Python standard library, so independent archive verification does
not require importing the scientific environment.
"""
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'projects/tds/paper_applsci'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    manifest_path=OUT/'reports/artifact_manifest.json'
    entries=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or 'packages' in p.parts or p==manifest_path:continue
        if p.suffix in ['.aux','.log','.out','.fls','.fdb_latexmk','.blg','.bbl','.pyc']:continue
        entries.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=digest(p)))
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=digest(p)) for p in sorted((ROOT/'projects/tds/scripts/revision314').rglob('*.py'))]
    manifest_path.write_text(json.dumps(dict(entries=entries,artifact_count=len(entries),source_entries=sources,
        policy='covers completed local revision, not a public release attestation'),indent=2,sort_keys=True)+'\n')
    archive=OUT/'packages/TDS_R314_local_evidence.zip'
    with zipfile.ZipFile(archive) as z:names=set(z.namelist())
    names.update(str((OUT/e['path']).relative_to(ROOT)) for e in entries)
    names.update(e['path'] for e in sources)
    names.add(str(manifest_path.relative_to(ROOT)))
    handle,tmp=tempfile.mkstemp(prefix='tds-evidence-seal-',suffix='.zip',dir=archive.parent)
    os.close(handle)
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(names):
            p=ROOT/name
            assert p.is_file(),p
            z.write(p,name)
    os.replace(tmp,archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for e in entries:
            name=str((OUT/e['path']).relative_to(ROOT))
            assert hashlib.sha256(z.read(name)).hexdigest()==e['sha256'],name
        for e in sources:
            assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256'],e['path']
        assert z.read(str(manifest_path.relative_to(ROOT)))==manifest_path.read_bytes()
    mpath=OUT/'packages/package_manifest.json';m=json.loads(mpath.read_text())
    m['evidence'].update(bytes=archive.stat().st_size,sha256=digest(archive))
    m['evidence_manifest_membership_verified']=True
    m['revision_artifacts_verified']=len(entries);m['revision_sources_verified']=len(sources)
    source=OUT/m['source']['path']
    assert digest(source)==m['source']['sha256']
    with zipfile.ZipFile(source) as z:assert z.testzip() is None
    mpath.write_text(json.dumps(m,indent=2,sort_keys=True)+'\n')
    print(json.dumps(m,indent=2))


if __name__=='__main__':main()
