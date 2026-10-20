"""Create local source/evidence ZIPs and qualify a clean LaTeX-source build."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from common import FROZEN, GRAPH, OUT, PROJECT, ROOT, sha, write_json


def archive(path, files, base):
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(files)):
            z.write(p,str(p.relative_to(base)))
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    return dict(path=str(path.relative_to(OUT)),bytes=path.stat().st_size,sha256=sha(path))


def main():
    packages=OUT/'packages';packages.mkdir(exist_ok=True)
    license_path=ROOT/'local_storage/benchmark/a09_submission_closure/upstream/GoG_LICENSE.txt'
    if license_path.is_file():
        shutil.copy2(license_path,OUT/'vendor/GoG_LICENSE.txt')
    source=[]
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        rel=p.relative_to(OUT)
        if rel.parts[0] in ['runs','reports','packages']:continue
        if p.suffix in ['.aux','.log','.fls','.out','.fdb_latexmk','.blg','.bbl','.pt']:continue
        if p.suffix=='.pdf' and rel.parts[0]!='Definitions' and rel.parts[0]!='figures' and rel.parts[0]!='vendor':continue
        source.append(p)
    source_info=archive(packages/'TDS_R314_LaTeX_sources.zip',source,OUT)
    # Compile the actual ZIP contents in a fresh directory, without a cached BBL.
    scratch=Path(tempfile.mkdtemp(prefix='tds-r314-source-build-'))
    with zipfile.ZipFile(packages/'TDS_R314_LaTeX_sources.zip') as z:z.extractall(scratch)
    compile_checks=[]
    for rel in [Path('applsci_article.tex'),Path('supplement/supplementary.tex')]:
        cmd=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',rel.name]
        done=subprocess.run(cmd,cwd=scratch/rel.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if done.returncode:raise RuntimeError(done.stdout[-5000:])
        log=(scratch/rel.with_suffix('.log')).read_text(errors='replace')
        assert all(t not in log for t in ['Overfull','Underfull','Warning','! LaTeX Error'])
        local=subprocess.check_output(['pdftotext','-layout',str(OUT/rel.with_suffix('.pdf')),'-'])
        fresh=subprocess.check_output(['pdftotext','-layout',str(scratch/rel.with_suffix('.pdf')),'-'])
        assert local==fresh,rel
        compile_checks.append(dict(source=str(rel),clean_compile=True,extracted_text_identical=True,
                                   clean_pdf_sha256=sha(scratch/rel.with_suffix('.pdf'))))
    write_json(OUT/'reports/clean_source_compile.json',dict(source_zip=source_info,checks=compile_checks,
        pdf_binary_identity_claimed=False,reason='PDF dates and trailer IDs may differ; document text and layout-affecting warnings checked',
        scratch_directory=str(scratch)))
    evidence=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or 'packages' in p.parts:continue
        if p.suffix in ['.aux','.log','.fls','.out','.fdb_latexmk','.blg','.bbl','.pyc']:continue
        evidence.append(p)
    evidence.extend((PROJECT/'scripts/revision314').rglob('*.py'))
    evidence.extend((PROJECT/'tests').glob('test_revision314.py'))
    evidence.extend(p for p in FROZEN.rglob('*') if p.is_file())
    evidence.append(GRAPH)
    evidence.extend((ROOT/'archive/tds/data/gog_scimain_v1_preserved_panel/checkpoints').glob('*.pt'))
    if license_path.is_file():
        # Already included by the general output traversal.
        assert (OUT/'vendor/GoG_LICENSE.txt') in evidence
    evidence.extend((ROOT/'docs/work_reports/tds/314_grok_review'/n) for n in ['implementation_plan.md','task.md','walkthrough.md'])
    # Seal after clean-build evidence exists and before the ZIP reads the
    # manifest bytes, avoiding an archive containing an obsolete manifest.
    entries=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or 'packages' in p.parts or p.name=='artifact_manifest.json':continue
        if p.suffix in ['.aux','.log','.out','.fls','.fdb_latexmk','.blg','.bbl','.pyc']:continue
        entries.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in sorted((PROJECT/'scripts/revision314').rglob('*.py'))]
    write_json(OUT/'reports/artifact_manifest.json',dict(entries=entries,artifact_count=len(entries),source_entries=sources,
        policy='covers completed local revision, not a public release attestation'))
    evidence_info=archive(packages/'TDS_R314_local_evidence.zip',evidence,ROOT)
    assert evidence_info['bytes']<120*1024*1024
    write_json(packages/'package_manifest.json',dict(source=source_info,evidence=evidence_info,
        clean_latex_compilation=True,publication_status='local packages only; no public upload or journal submission',
        upstream_data_license='CC BY-NC-SA 4.0; upstream source ZIPs are not included',
        reproduction='repository-relative source/evidence paths preserved in the evidence ZIP'))
    print(json.dumps(dict(source=source_info,evidence=evidence_info,clean_compile=compile_checks),indent=2),flush=True)


if __name__=='__main__':main()
