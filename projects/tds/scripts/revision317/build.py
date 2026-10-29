"""Package private venue sources and clean-build each in an empty directory."""
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from common import PROJECT,PAPER,REPORTS,OUT,sha,write


def compile_one(root,entry):
    run=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',entry.name],cwd=root,text=True,capture_output=True)
    log=(root/entry.with_suffix('.log').name).read_text(errors='replace')
    (REPORTS/f'{root.parent.name}_{entry.stem}_build_tail.txt').write_text(run.stdout[-3000:]+run.stderr[-1000:])
    assert run.returncode==0,run.stdout[-2000:]
    assert not re.search(r'undefined citations|Citation .* undefined|Reference .* undefined|LaTeX Error|Emergency stop',log)
    pdf=root/entry.with_suffix('.pdf').name
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    return dict(entry=str(entry),pdf_sha256=sha(pdf),pages=int(re.search(r'Pages:\s+(\d+)',info).group(1)),
        errors=0,undefined_citations=0,missing_figures=0,overfull_boxes=len(re.findall(r'Overfull \\hbox',log)))


def text_of(path):return subprocess.check_output(['pdftotext','-layout',str(path),'-'],text=True)


def main():
    records=[];bodies=[];declarations=[];refs=[]
    for venue,entry,delivery_name in [('preprints','preprint_main.tex','preprints'),('mdpi_applsci_anomaly','applsci_article.tex','applsci')]:
        folder=PAPER/venue
        bodies.append(sha(folder/'manuscript_body.tex'));declarations.append(sha(folder/'declarations.tex'));refs.append(sha(folder/'references.bib'))
        local=compile_one(folder,Path(entry));supp=compile_one(folder/'supplement',Path('supplement317.tex'))
        shutil.copy2(folder/'supplement/supplement317.pdf',folder/('preprint_supplement.pdf' if venue=='preprints' else 'applsci_supplement.pdf'))
        if venue=='preprints':shutil.copy2(folder/'preprint_main.pdf',folder/'preprint_article.pdf')
        delivery=PROJECT/f'submission/{delivery_name}/317';delivery.mkdir(parents=True,exist_ok=True)
        zipname='preprint_source.zip' if venue=='preprints' else 'mdpi_source.zip'
        dest=delivery/zipname
        include=[folder/entry,folder/'manuscript_body.tex',folder/'declarations.tex',folder/'references.bib']
        include+=list((folder/'data').glob('*.tex'))
        include += [folder/'figures'/n for n in ['selection_audit.pdf','validity_decision_tree.pdf']]
        include += [folder/'supplement/supplement317.tex']
        if venue!='preprints':include+=list((folder/'Definitions').glob('*'))+[folder/'soul.sty',folder/'soul-ori.sty']
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(include):
                if p.is_file():z.write(p,p.relative_to(folder))
        shutil.copy2(dest,folder/zipname)
        with tempfile.TemporaryDirectory(prefix='tds317_source_') as temp:
            root=Path(temp)
            with zipfile.ZipFile(dest) as z:z.extractall(root)
            clean=compile_one(root,Path(entry));clean_supp=compile_one(root/'supplement',Path('supplement317.tex'))
            assert text_of(folder/Path(entry).with_suffix('.pdf'))==text_of(root/Path(entry).with_suffix('.pdf'))
            assert text_of(folder/'supplement/supplement317.pdf')==text_of(root/'supplement/supplement317.pdf')
        for p in [folder/Path(entry).with_suffix('.pdf'),folder/'supplement/supplement317.pdf']:
            name=('preprint_article.pdf' if delivery_name=='preprints' else 'applsci_article.pdf') if p.parent==folder else ('preprint_supplement.pdf' if delivery_name=='preprints' else 'applsci_supplement.pdf')
            shutil.copy2(p,delivery/name)
        records.append(dict(venue=venue,local=local,supplement=supp,clean_article=clean,clean_supplement=clean_supp,
            source_zip_sha256=sha(dest),source_zip_path=str(dest.relative_to(PROJECT)),clean_text_identical=True))
    assert len(set(bodies))==len(set(declarations))==len(set(refs))==1
    neutral=(PAPER/'preprints/preprint_main.tex').read_text()
    assert 'Definitions/mdpi' not in neutral and 'mdpi' not in neutral.lower()
    mdpi=(PAPER/'mdpi_applsci_anomaly/applsci_article.tex').read_text()
    assert not any(x in mdpi for x in ['TDS revision prepared','Prepared for','DOI not assigned'])
    write(REPORTS/'build_verification.json',dict(status='PASS',venues=records,source_zip_fresh_directory_build=True,
        preprint_neutral=True,shared_body_sha256=bodies[0],shared_declarations_sha256=declarations[0],shared_bib_sha256=refs[0],
        platform_facts_and_numbers_identical=True,notes='Different venue layouts/page counts are allowed; source ZIP text matches cold build.'))
    print('Dual source ZIP clean build PASS')


if __name__=='__main__':main()
