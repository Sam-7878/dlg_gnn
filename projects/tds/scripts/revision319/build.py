"""Create independent R319 Preprints and MDPI source ZIPs and cold build receipts."""
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from common import PROJECT,PAPER,REPORTS,OUT,sha,write

def run(root,entry):
    cmd=['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',entry]
    p=subprocess.run(cmd,cwd=root,text=True,capture_output=True)
    if not root.name.startswith('tds319_coldbuild_'):
        (REPORTS/f'{root.name}_{Path(entry).stem}_build_tail.txt').write_text(p.stdout[-5000:]+p.stderr[-1000:])
    if p.returncode:raise RuntimeError(f'{entry}: {p.stdout[-3000:]} {p.stderr[-600:]}')
    log=(root/Path(entry).with_suffix('.log')).read_text(errors='replace')
    assert not re.search(r'Citation .* undefined|Reference .* undefined|LaTeX Error|Emergency stop|undefined citations',log)
    pdf=root/Path(entry).with_suffix('.pdf')
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    return dict(sha256=sha(pdf),pages=int(re.search(r'Pages:\s+(\d+)',info).group(1)),overfull=len(re.findall(r'Overfull \\hbox',log)))

def extract(p):return subprocess.check_output(['pdftotext','-layout',str(p),'-'],text=True)

def main():
    receipts=[]
    a=[];b=[];c=[]
    for name,venue,entry,zipname,article in [('preprints','preprints','preprint_main.tex','preprint_source.zip','preprint_article.pdf'),
                                             ('mdpi_applsci_anomaly','applsci','applsci_article.tex','mdpi_source.zip','applsci_article.pdf')]:
        root=PAPER/name;delivery=PROJECT/'submission'/venue/'319';delivery.mkdir(parents=True,exist_ok=True)
        a.append(sha(root/'manuscript_body.tex'));b.append(sha(root/'declarations.tex'));c.append(sha(root/'references.bib'))
        article_local=run(root,entry);supp_local=run(root/'supplement','supplement317.tex')
        include=[root/p for p in [entry,'manuscript_body.tex','declarations.tex','references.bib','supplement/supplement317.tex']]
        include+=list((root/'data').glob('*.tex'))+list((root/'figures').glob('*.pdf'))
        if venue=='applsci':include+=list((root/'Definitions').glob('*'))+[root/'soul.sty',root/'soul-ori.sty']
        zpath=delivery/zipname
        with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(include):
                if p.is_file():z.write(p,p.relative_to(root))
        with tempfile.TemporaryDirectory(prefix='tds319_coldbuild_') as tmp:
            cold=Path(tmp)
            with zipfile.ZipFile(zpath) as z:z.extractall(cold)
            article_cold=run(cold,entry);supp_cold=run(cold/'supplement','supplement317.tex')
            same_article=extract(root/Path(entry).with_suffix('.pdf'))==extract(cold/Path(entry).with_suffix('.pdf'))
            same_supp=extract(root/'supplement/supplement317.pdf')==extract(cold/'supplement/supplement317.pdf')
            assert same_article and same_supp
        shutil.copyfile(root/Path(entry).with_suffix('.pdf'),delivery/article)
        shutil.copyfile(root/'supplement/supplement317.pdf',delivery/'supplement.pdf')
        receipts.append(dict(venue=venue,article_local=article_local,article_cold=article_cold,supplement_local=supp_local,
            supplement_cold=supp_cold,source_sha256=sha(zpath),article_text_equal=same_article,supplement_text_equal=same_supp,
            files={p.name:sha(p) for p in [zpath,delivery/article,delivery/'supplement.pdf']}))
    assert len(set(a))==len(set(b))==len(set(c))==1
    pre=(PAPER/'preprints/preprint_main.tex').read_text();assert 'Definitions/mdpi' not in pre and 'logo-mdpi' not in pre
    write(OUT/'source_build_receipt_319.json',dict(status='PASS',sources_byte_identical=dict(manuscript_body=a[0],declarations=b[0],references=c[0]),
        neutral_preprint=True,independent_cold_builds=True,venues=receipts))
    print('R319 dual cold builds PASS',flush=True)

if __name__=='__main__':main()
