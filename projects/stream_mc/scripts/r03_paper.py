"""Author-local R03 clean builds; never infer author approval from compilation."""
from __future__ import annotations
import argparse
import hashlib
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from r03_common import ROOT,PROJECT,PAPER,OLD,OUT,RUN,artifact,sources,unchanged,now,read,write,sha256,command_record


def validate_generated():
    records=[]
    for original in sorted((OLD/'generated').glob('*.tex')):
        candidate=PAPER/'generated'/original.name
        # Captions are single lines in these generated inputs. Only three
        # explicitly named captions may change; numerical bodies stay exact.
        old=original.read_bytes();new=candidate.read_bytes()
        caption_only=original.name in {'table_primary.tex','table_control_cost.tex','table_matched.tex'}
        if caption_only:
            strip=lambda b:b'\n'.join(line for line in b.split(b'\n') if line.startswith(b'\\caption{') is False)
            assert strip(old)==strip(new),original.name
        else:assert old==new,original.name
        records.append({'name':original.name,'R02_sha256':sha256(original),'R03_sha256':sha256(candidate),'numeric_body_byte_identical':True,'caption_changed':old!=new})
    assert len(records)==19 and sum(r['caption_changed'] for r in records)==3
    write(OUT/'generated_numeric_identity_r03.json',{'inputs':records,'count':19,'table_count':18,'numeric_macro_files':1,'numeric_values_changed':False})
    return records


def clean(source,entry,dest,tag):
    """Fresh stage, actual subprocess exits, strict final-log rejection."""
    dest.mkdir(parents=True,exist_ok=True);commands=[];stem=Path(entry).stem
    with tempfile.TemporaryDirectory(prefix='streammc_r03_tex_') as directory:
        stage=Path(directory)
        for p in sources(source):
            target=stage/p.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert (stage/entry).exists()
        for i,cmd in enumerate((['pdflatex','-halt-on-error','-file-line-error','-interaction=nonstopmode',entry],['bibtex',stem],
                               ['pdflatex','-halt-on-error','-file-line-error','-interaction=nonstopmode',entry],['pdflatex','-halt-on-error','-file-line-error','-interaction=nonstopmode',entry])):
            record=command_record(cmd,stage,OUT/'build_logs'/f'{tag}_{stem}_{i}.log');commands.append(record)
            assert record['actual_exit_code']==0,record
        log=(stage/(stem+'.log')).read_text(errors='replace')
        bad=re.findall(r'(?:Overfull \\[hv]box[^\n]*|! (?:LaTeX|Package).*Error[^\n]*|[^\n]*(?:undefined references|undefined citations|Citation[^\n]*undefined|Reference[^\n]*undefined|Rerun to get cross-references)[^\n]*)',log,re.I)
        assert not bad,bad
        for suffix in ('.pdf','.bbl','.log'):
            shutil.copy2(stage/(stem+suffix),dest/(stem+suffix))
    pdf=dest/(stem+'.pdf');text=subprocess.check_output(['pdftotext','-layout',str(pdf),'-'],text=True)
    (dest/(stem+'.txt')).write_text(text)
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    return {'entry':entry,'PDF':artifact(pdf),'pages':int(re.search(r'Pages:\s+(\d+)',info).group(1)),
            'commands':commands,'unresolved_count':0,'overflow_count':0,'cited_references':(dest/(stem+'.bbl')).read_text().count('\\bibitem'),
            'extracted_text':artifact(dest/(stem+'.txt'))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-label',default='candidate1');args=parser.parse_args()
    assert re.fullmatch(r'[a-z0-9_-]+',args.run_label)
    begin=now();unchanged();validate_generated()
    # No deletion/reset of an earlier candidate or its rendering namespace.
    build=PAPER/('build_r03_'+args.run_label);visual=PAPER/('visual_r03_'+args.run_label)
    assert not build.exists() and not visual.exists(),'Use a new run label; preserve old build identities'
    from PIL import Image,ImageDraw,ImageOps
    records=[]
    for entry in ('preprint.tex','journal.tex','supplement.tex'):
        record=clean(PAPER,entry,build,args.run_label);stem=Path(entry).stem;pdf=ROOT/record['PDF']['path']
        folder=visual/(stem+'_'+record['PDF']['sha256'][:12]);folder.mkdir(parents=True)
        render=command_record(['pdftoppm','-r','100','-png',pdf,folder/'page'],ROOT,OUT/'build_logs'/f'{args.run_label}_{stem}_render.log')
        assert render['actual_exit_code']==0
        images=sorted(folder.glob('page-*.png'));assert len(images)==record['pages']
        sheets=[]
        for start in range(0,len(images),8):
            subset=images[start:start+8];sheet=Image.new('RGB',(1120,800*((len(subset)+1)//2)),'white')
            for j,p in enumerate(subset):
                with Image.open(p) as im:thumb=ImageOps.contain(im.convert('RGB'),(540,765))
                x=(j%2)*560;y=(j//2)*800;sheet.paste(thumb,(x+(560-thumb.width)//2,y+25));ImageDraw.Draw(sheet).text((x+8,y+5),f'{stem} page {start+j+1}',fill='black')
            output=folder/f'contact_{start//8+1}.png';sheet.save(output);sheets.append(artifact(output))
        text=(build/(stem+'.txt')).read_text()
        if stem!='supplement':
            record['abstract_words']=len(text.split('Abstract',1)[1].split('Keywords',1)[0].split());assert record['abstract_words']<=200
            assert record['cited_references']==20
            for v in ('0.505','0.507','27','29','pending','not yet','0.05','97.9','92.9'):assert v in text,(stem,v)
            flat=re.sub(r'\s+','',text);assert '10.20944/preprints202609.0848.v1' in flat
        if stem=='preprint':assert 'Applied Sciences author-review' not in text and 'Electronics author-review' not in text
        record.update({'render_command':render,'rendered_pages':[artifact(p) for p in images],'contact_sheets':sheets,
                       'visual_status':'OPEN until agent inspects all exact rendered pages; not author approval'})
        records.append(record);print('R03 CLEAN BUILD/RENDER',stem,record['pages'],flush=True)
    write(OUT/'paper_build_audit.json',{'run_id':RUN,'started_utc':begin,'finished_utc':now(),'build_label':args.run_label,
        'source_inputs':[artifact(p) for p in sources(PAPER)],'builds':records,'actual_exit_code':0,'script':artifact(__file__)})
    unchanged();print('R03 CLEAN BUILDS PASS; visual review and author approval are separate',flush=True)


if __name__=='__main__':main()
