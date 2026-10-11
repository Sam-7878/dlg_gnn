"""Render every private PDF page for explicit human/agent visual inspection."""
from pathlib import Path
import subprocess
from r01_common import PROJECT,output,write,sha256
from PIL import Image,ImageDraw


def main():
    paper=PROJECT/'paper/current/r01';dest=paper/'visual';dest.mkdir(exist_ok=True);records=[]
    for name in ('preprint','journal','supplement'):
        pdf=paper/f'build/{name}.pdf';pages=dest/name;pages.mkdir(exist_ok=True)
        subprocess.run(['pdftoppm','-r','110','-png',str(pdf),str(pages/'page')],check=True)
        files=sorted(pages.glob('page-*.png'))
        for start in range(0,len(files),4):
            sheet=Image.new('RGB',(1200,1700),'#dddddd');draw=ImageDraw.Draw(sheet)
            for i,path in enumerate(files[start:start+4]):
                img=Image.open(path);img.thumbnail((580,805));x=(i%2)*600+10;y=(i//2)*850+30
                sheet.paste(img,(x,y));draw.text((x,y-20),name+' '+path.stem,fill='black')
            sheet.save(dest/f'{name}_sheet_{start//4+1}.png')
        records.append({'document':name,'pdf_sha256':sha256(pdf),'N_pages':len(files),'page_images':[{'path':str(p.relative_to(PROJECT)),'sha256':sha256(p)} for p in files],
                        'render_dpi':110,'rendering_is_not_visual_approval':True})
    write(output()/'audits/page_render_inventory.json',records);print('ALL PDF PAGES RENDERED',[(r['document'],r['N_pages']) for r in records],flush=True)


if __name__=='__main__':main()
