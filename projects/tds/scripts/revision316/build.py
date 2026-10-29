"""Build independent private source ZIPs and verify all rendered pages."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from paths import ROOT,PROJECT,SHARED,PREPRINT,MDPI,REPORTS,PRIVATE,RESULTS,delivery
from common import sha,write
from latex_tools import compile_one,render,pixel_hash,contact


def main():
    subprocess.run([sys.executable,str(SHARED/'prepare.py')],check=True)
    metadata=json.loads((SHARED/'metadata.json').read_text())
    assert len(metadata['abstract'].split())<=200
    builds=[]
    for venue,folder,entry in [('preprints',PREPRINT,'preprint_main.tex'),('mdpi',MDPI,'applsci_article.tex')]:
        out=delivery(venue);out.mkdir(parents=True,exist_ok=True)
        docs=[entry,'supplement/supplementary.tex']
        original=[compile_one(folder,rel) for rel in docs]
        if venue=='preprints':
            assert not re.search(r'Applied Sciences|Appl\. Sci\.|Prepared for|Received:|Accepted:|Published:|MDPI',
                                 '\n'.join(e['text'] for e in original))
            assert 'not peer reviewed' in original[0]['text']
        first_page=original[0]['text'].split('\f')[0]
        assert all(s in first_page for s in ['SeongSu Park','Ki-Hyung Kim','Ajou','parky@ajou.ac.kr','kkim86@ajou.ac.kr','Abstract','Keywords'])
        assert sha(folder/'manuscript_body.tex')==sha(SHARED/'sections/manuscript_body.tex')
        assert sha(folder/'references.bib')==sha(SHARED/'references.bib')
        alias={'preprint_article.pdf','preprint_supplement.pdf'}
        files=[p for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in alias
               and p.suffix in {'.tex','.bib','.cls','.bst','.sty','.eps','.svg','.pdf'}]
        assert not list(folder.rglob('*.csv')) and not list(folder.rglob('*.pt'))
        if venue=='preprints':
            assert not any('Definitions' in p.parts or 'mdpi' in p.name.lower() or 'logo-' in p.name for p in files)
        archive=out/('preprint_source.zip' if venue=='preprints' else 'mdpi_source.zip')
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(folder))
        scratch=Path(tempfile.mkdtemp(prefix=f'tds316-{venue}-'))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None;z.extractall(scratch)
            for rel in docs:
                pdf=Path(rel).with_suffix('.pdf')
                assert hashlib.sha256(z.read(str(pdf))).hexdigest()==sha(folder/pdf)
                (scratch/pdf).unlink()
        fresh=[compile_one(scratch,rel) for rel in docs]
        rendered=[]
        for i,rel in enumerate(docs):
            assert original[i]['text']==fresh[i]['text']
            kind='article' if i==0 else 'supplement'
            a=render(folder/Path(rel).with_suffix('.pdf'),PRIVATE/'qa'/venue/kind)
            b=render(scratch/Path(rel).with_suffix('.pdf'),scratch/('render_'+kind))
            assert len(a)==len(b)==original[i]['pages']
            for n,(p,q) in enumerate(zip(a,b),1):
                assert pixel_hash(p)==pixel_hash(q),(venue,kind,n)
                rendered.append(dict(document=kind,page=n,path=str(p.relative_to(ROOT)),sha256=sha(p),pixel_sha256=pixel_hash(p)))
            for start in range(0,len(a),4):contact(a[start:start+4],PRIVATE/'qa'/venue/f'{kind}_{start+1}_{min(start+4,len(a))}.png',cols=2)
        for e in original+fresh:e.pop('text')
        record=dict(venue=venue,status='PASS',builds=original,clean_builds=fresh,rendered_pages=rendered,
                    all_text_and_pixels_identical=True,source_zip=dict(path=str(archive.relative_to(ROOT)),sha256=sha(archive),crc='PASS'),
                    first_page_metadata_displayed=True,corresponding_author_confirmation=False,
                    neutral_template_verified=venue=='preprints',publisher_assets_in_preprint=False,
                    source_entries=[dict(path=str(p.relative_to(folder)),sha256=sha(p)) for p in files])
        builds.append(record);write(REPORTS/f'{venue}_build.json',record)
        for rel in docs:
            dest=out/Path(rel).with_suffix('.pdf');dest.parent.mkdir(exist_ok=True)
            shutil.copy2(folder/Path(rel).with_suffix('.pdf'),dest)
        if venue=='preprints':
            shutil.copy2(folder/'preprint_main.pdf',folder/'preprint_article.pdf')
            shutil.copy2(folder/'supplement/supplementary.pdf',folder/'preprint_supplement.pdf')
    write(REPORTS/'build_verification.json',dict(status='PASS',venues=builds,shared_body_identity=True,
        title=metadata['Title'],abstract_words=len(metadata['abstract'].split()),
        source_build_is_not_author_or_release_approval=True))
    write(RESULTS/'final_artifact_manifest.json',dict(files=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
        for folder in [PREPRINT,MDPI,delivery('preprints'),delivery('mdpi')] for p in sorted(folder.rglob('*.pdf'))]
        +[dict(path=v['source_zip']['path'],sha256=v['source_zip']['sha256']) for v in builds]))
    print('PASS: two independent source ZIPs, all text/rendered pages identical to fresh builds')


if __name__=='__main__':main()
