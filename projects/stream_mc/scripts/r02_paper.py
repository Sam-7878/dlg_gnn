"""Actual private R02 builds, negative fixtures, all-page renders and packages.

This contains tooling, not manuscript prose; PDFs remain author-local.
Rendering is NOT automatically recorded as human visual approval.
"""
from __future__ import annotations
import re
import subprocess
import zipfile
from r02_common import ROOT,PROJECT,R01,R02,VERIFY,PAPER,RUN,read,write,sha256,artifact,original_unchanged
from r01_validate import clean_build,fixtures


def main():
    original_unchanged();build=PAPER/'build_r02';visual=PAPER/'visual_r02';visual.mkdir(exist_ok=True)
    fixtures(VERIFY/'negative_fixtures')
    records=[]
    from PIL import Image,ImageOps,ImageDraw
    for entry in ('preprint.tex','journal.tex','supplement.tex'):
        record=clean_build(PAPER,entry,build);stem=entry[:-4];pdf=build/(stem+'.pdf')
        text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(re.search(r'Pages:\s+(\d+)',info).group(1))
        # A fresh render directory avoids treating copied R01 images as R02.
        folder=visual/stem;folder.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','90','-png',str(pdf),str(folder/'page')],check=True,capture_output=True)
        images=sorted(folder.glob('page-*.png'));assert len(images)==pages
        thumbs=[]
        for i,path in enumerate(images,1):
            with Image.open(path) as source:thumb=ImageOps.contain(source.convert('RGB'),(360,515))
            tile=Image.new('RGB',(380,550),'#dddddd');tile.paste(thumb,((380-thumb.width)//2,24));ImageDraw.Draw(tile).text((12,4),f'{stem} page {i}',fill='black');thumbs.append(tile)
        sheets=[]
        for start in range(0,len(thumbs),9):
            subset=thumbs[start:start+9];sheet=Image.new('RGB',(1140,550*((len(subset)+2)//3)),'white')
            for i,tile in enumerate(subset):sheet.paste(tile,((i%3)*380,(i//3)*550))
            dest=visual/f'{stem}_contact_{start//9+1}.png';sheet.save(dest);sheets.append(artifact(dest))
        record.update({'pages':pages,'bytes':pdf.stat().st_size,'rendered_pages':[artifact(p) for p in images],'contact_sheets':sheets,'visual_approval':'OPEN until exact rendered pages inspected'})
        if stem!='supplement':
            abstract=text.split('Abstract',1)[1].split('Keywords',1)[0];words=len(abstract.split());assert words<=200,(stem,words)
            record['abstract_pdf_word_count']=words
            assert '10.20944/preprints202609.0848.v1' in text.replace('\n','').replace(' ','')
            assert '1810.00826' in text and '2105.14491' in text
            bbl=(build/(stem+'.bbl')).read_text();record['cited_references']=bbl.count('\\bibitem')
        records.append(record)
    write(VERIFY/'paper_build_audit.json',{'verification_id':RUN,'builds':records,'scope':'actual builds and renders; not author or publication approval'})
    packages=[];dest=PAPER/'packages_r02';dest.mkdir(exist_ok=True)
    for label in ('preprint','journal'):
        path=dest/(label+'_r02_package.zip')
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
            for file in sorted(PAPER.rglob('*')):
                relative=file.relative_to(PAPER)
                if not file.is_file() or any(x.startswith(('build','visual','packages')) for x in relative.parts):continue
                z.write(file,str(relative))
            z.write(build/(label+'.pdf'),label+'.pdf');z.write(build/'supplement.pdf','supplement.pdf')
        packages.append(artifact(path))
    total=sum((build/(name+'.pdf')).stat().st_size for name in ('journal','supplement'))+(dest/'journal_r02_package.zip').stat().st_size
    assert total<120_000_000
    write(VERIFY/'private_package_identities_r02.json',{'packages':packages,'journal_candidate_total_bytes':total,'limit_plan_bytes':120_000_000,'large_public_evidence_separate':True,'scope':'local candidate size; portal-specific rules and exact final author approval still require confirmation'})
    baseline=read(VERIFY/'handoff_baseline.json');assert all(sha256(ROOT/name)==value for name,value in baseline['private_paper_files'].items())
    original_unchanged();print('R02 ACTUAL BUILDS, FIXTURES, RENDERS AND PRIVATE PACKAGES PASS',flush=True)


if __name__=='__main__':main()
