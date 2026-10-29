"""Source compilation and render verification without manuscript generation."""
import hashlib
import re
import subprocess
from PIL import Image, ImageDraw
from freeze import sha


def compile_one(base, rel):
    source=base/rel
    run=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-file-line-error',source.name],
                       cwd=source.parent,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    log=source.with_suffix('.log').read_text(errors='replace') if source.with_suffix('.log').exists() else run.stdout
    warnings=[s for s in log.splitlines() if re.search(r'Overfull|Underfull|Warning|^!',s)]
    if run.returncode or warnings:raise RuntimeError((rel,run.returncode,warnings,run.stdout[-2000:]))
    text=subprocess.check_output(['pdftotext','-layout',source.with_suffix('.pdf'),'-'],text=True)
    assert '??' not in text and '\ufffd' not in text
    return dict(source=rel,pages=int(re.search(r'Output written on .*?\((\d+) pages',log).group(1)),
                pdf_sha256=sha(source.with_suffix('.pdf')),log_sha256=sha(source.with_suffix('.log')),
                warnings=[],text=text)


def render(pdf,folder):
    folder.mkdir(parents=True,exist_ok=True)
    subprocess.run(['pdftoppm','-r','90','-png',pdf,folder/'page'],check=True,stderr=subprocess.PIPE)
    return sorted(folder.glob('page-*.png'),key=lambda p:int(p.stem.split('-')[-1]))


def pixel_hash(path):
    with Image.open(path) as im:
        return hashlib.sha256(str(im.size).encode()+im.mode.encode()+im.tobytes()).hexdigest()


def contact(pages,path,cols=3):
    w,h=420,620
    sheet=Image.new('RGB',(w*cols,h*((len(pages)+cols-1)//cols)),'white');d=ImageDraw.Draw(sheet)
    for i,p in enumerate(pages):
        with Image.open(p) as im:
            im=im.convert('RGB');im.thumbnail((w-16,h-35));x,y=(i%cols)*w,(i//cols)*h
            sheet.paste(im,(x+8,y+25));d.text((x+8,y+6),p.stem,fill='black')
    sheet.save(path)
