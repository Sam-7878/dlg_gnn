"""Build venue-specific private deliveries with shared scientific-source identity."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from paths import ROOT, PROJECT, PAPER, PREPRINT, COMMON, PRIVATE, AUTHORS, REPORTS, PUBLICATION, RESULTS
from freeze import sha, write
from latex_tools import compile_one, render, pixel_hash, contact

TARGET=dict(journal='Applied Sciences',special_issue='Machine Learning and Its Application for Anomaly Detection',
            special_issue_url='https://www.mdpi.com/journal/applsci/special_issues/c67h3u9qgt',
            direct_page_verified=False,deadline=None,authority='User-specified target; direct official page returned HTTP429')


def main():
    subprocess.run([sys.executable,str(COMMON/'prepare.py')],check=True)
    builds=[]
    for venue,folder,entry in [('applsci',PAPER,'applsci_article.tex'),('preprints',PREPRINT,'TDS_315_Preprints.tex')]:
        delivery=PROJECT/'submission'/venue/'315';delivery.mkdir(parents=True,exist_ok=True)
        checks=[compile_one(folder,entry),compile_one(folder,'supplement/supplementary.tex')]
        if venue=='preprints':
            assert not re.search(r'Applied Sciences|MDPI|Received:|Accepted:', '\n'.join(e['text'] for e in checks))
            assert 'Preprint draft; not peer reviewed' in checks[0]['text']
        assert sha(folder/'manuscript_body.tex')==sha(COMMON/'manuscript_body.tex')
        files=[]
        for p in sorted(folder.rglob('*')):
            if not p.is_file():continue
            if p.suffix in {'.tex','.bib','.cls','.bst','.sty','.eps','.svg','.pdf'}:files.append(p)
        assert not list(folder.rglob('*.csv')) and not list(folder.rglob('*.pt'))
        archive=delivery/f'TDS_315_{venue}_sources.zip'
        with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(folder))
        scratch=Path(tempfile.mkdtemp(prefix=f'tds315-{venue}-'))
        with zipfile.ZipFile(archive) as z:
            assert z.testzip() is None
            z.extractall(scratch)
            for rel in [entry,'supplement/supplementary.tex']:
                pdf=Path(rel).with_suffix('.pdf')
                assert hashlib.sha256(z.read(str(pdf))).hexdigest()==sha(folder/pdf)
                (scratch/pdf).unlink()
        fresh=[compile_one(scratch,entry),compile_one(scratch,'supplement/supplementary.tex')]
        renders=[]
        for i,rel in enumerate([entry,'supplement/supplementary.tex']):
            assert checks[i]['text']==fresh[i]['text']
            label='article' if i==0 else 'supplement'
            pages=render(folder/Path(rel).with_suffix('.pdf'),PRIVATE/'qa'/venue/label)
            other=render(scratch/Path(rel).with_suffix('.pdf'),scratch/('render_'+label))
            assert len(pages)==len(other)==checks[i]['pages']
            for n,(p,q) in enumerate(zip(pages,other),1):
                assert pixel_hash(p)==pixel_hash(q),(venue,label,n)
                renders.append(dict(document=label,page=n,path=str(p.relative_to(ROOT)),sha256=sha(p),pixel_sha256=pixel_hash(p)))
            for start in range(0,len(pages),4):
                contact(pages[start:start+4],PRIVATE/'qa'/venue/f'{label}_{start+1}_{min(start+4,len(pages))}.png',cols=2)
        for rec in checks+fresh:rec.pop('text')
        record=dict(venue=venue,status='PASS',builds=checks,clean_builds=fresh,
                    rendered_pages=renders,all_clean_text_and_pixels_identical=True,
                    manuscript_body_sha256=sha(folder/'manuscript_body.tex'),
                    source_zip=dict(path=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,crc='PASS'),
                    numeric_results_in_manuscript_folder=False,submitted=False)
        builds.append(record)
        write(REPORTS/'venue_split'/f'{venue}_build.json',record)
        # Delivery PDFs are convenience copies, manuscript sources remain canonical in paper/.
        for rel in [Path(entry).with_suffix('.pdf'),Path('supplement/supplementary.pdf')]:
            dest=delivery/rel;dest.parent.mkdir(exist_ok=True);shutil.copy2(folder/rel,dest)
    write(REPORTS/'venue_split/build_verification.json',dict(status='PASS',venues=builds,
          shared_body_identity=True,common_bibliography_sha256=sha(COMMON/'references.bib'),
          intended_workflow=['Preprints.org first','Applied Sciences special issue later'],target=TARGET))
    write(PROJECT/'submission/venue_plan.json',dict(date='2026-10-29',target=TARGET,
          preprint=dict(platform='Preprints.org',url=None,doi=None,posted=False),
          journal=dict(submitted=False,preprint_disclosure_required_after_actual_posting=True),
          manuscript_sources_private=True,results_root='projects/tds/results'))
    print('PASS: both venue source ZIPs clean-build; scientific body identity and all rendered pages verified')


def finalize():
    builds=json.loads((REPORTS/'venue_split/build_verification.json').read_text())
    audit=json.loads((REPORTS/'evidence_audit.json').read_text())
    assert builds['status']==audit['status']=='PASS'
    status=dict(date='2026-10-29',decision='HOLD_AUTHOR_ACTION',SUBMISSION_READY=False,SUBMITTED=False,
        target=TARGET,scientific_evidence_complete=True,venue_builds_pass=True,
        shared_body_sha256=sha(COMMON/'manuscript_body.tex'),
        results_root='projects/tds/results',numeric_release_published=False,
        pending=['Final venue-specific author approval and corresponding-author identity','Official award evidence and remaining declarations',
                 'Immutable numeric TDS release and anonymous verification','Live platform checks; current official MDPI template equivalence'],
        preprint=dict(posted=False,doi=None,url=None),journal=dict(submitted=False),
        prior_315_approval_hashes_superseded=True)
    write(REPORTS/'venue_split/submission_readiness.json',status)
    for v in builds['venues']:
        delivery=PROJECT/'submission'/v['venue']/'315'
        shutil.copy2(REPORTS/'venue_split/submission_readiness.json',delivery/'submission_readiness.json')
        write(delivery/'approval_identity.json',dict(venue=v['venue'],files=v['builds'],source_zip=v['source_zip'],
              final_author_approval_received=False,previous_version_approval_not_reused=True))
    print('Venue readiness recorded: HOLD_AUTHOR_ACTION; no posting/submission performed')


if __name__=='__main__':main()
