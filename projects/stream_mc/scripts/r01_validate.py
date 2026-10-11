"""Fail-closed identity/hash/TeX checks. Real builds, never string-only PASS."""
from __future__ import annotations
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from r01_common import ROOT,config,output,read,write,sha256


def hashes(base,manifest):
    base=Path(base)
    for name,expected in manifest.items():
        path=base/name
        if not path.is_file():raise ValueError('missing artifact: '+name)
        if sha256(path)!=expected:raise ValueError('SHA-256 mismatch: '+name)


def identity(records,expected_run,expected_experiment):
    if not records:raise ValueError('empty aggregate')
    if any(r['run_id']!=expected_run or r['experiment_id']!=expected_experiment for r in records):raise ValueError('mixed experiment/run identity')


def clean_build(source,entry,destination):
    source=Path(source);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if not (source/entry).is_file():raise ValueError('missing manuscript entry point')
    if shutil.which('pdflatex') is None or shutil.which('bibtex') is None:raise RuntimeError('complete TeX compiler and BibTeX required')
    with tempfile.TemporaryDirectory(prefix='streammc_clean_build_') as directory:
        stage=Path(directory)/'source';shutil.copytree(source,stage)
        for suffix in ('.aux','.bbl','.blg','.log','.out','.toc','.pdf'):
            for path in stage.rglob('*'+suffix):
                # Supplied figure PDFs are inputs, not stale build outputs.
                if path.parent==stage:path.unlink()
        stem=Path(entry).stem;commands=[]
        output_text=[]
        for attempt in range(4):
            if attempt==1 and '\\bibdata' in (stage/(stem+'.aux')).read_text():command=['bibtex',stem]
            else:command=['pdflatex','-halt-on-error','-file-line-error','-interaction=nonstopmode',entry]
            result=subprocess.run(command,cwd=stage,capture_output=True,text=True);commands.append(command);output_text.append(result.stdout+result.stderr)
            if result.returncode:
                (destination/(stem+'.failed.log')).write_text('\n'.join(output_text));raise ValueError(f'TeX build failed ({result.returncode}): {entry}')
        log=(stage/(stem+'.log')).read_text(errors='replace');bad=re.findall(r'(?:LaTeX Warning: (?:Reference|Citation).*undefined|There were undefined (?:references|citations)|Overfull \\[hv]box[^\n]*|! LaTeX Error[^\n]*)',log)
        (destination/(stem+'.build.log')).write_text('\n'.join(output_text))
        if bad:raise ValueError('unresolved reference/citation or overflow: '+repr(bad))
        shutil.copy2(stage/(stem+'.pdf'),destination/(stem+'.pdf'))
        if (stage/(stem+'.bbl')).exists():shutil.copy2(stage/(stem+'.bbl'),destination/(stem+'.bbl'))
        return {'entry':entry,'pdf_sha256':sha256(destination/(stem+'.pdf')),'unresolved_count':0,'overflow_count':0,'commands':commands,'actual_clean_build':True}


def fixtures(dest):
    result=[]
    with tempfile.TemporaryDirectory(prefix='streammc_negative_') as directory:
        base=Path(directory);(base/'payload.csv').write_text('a,b\n1,2\n')
        for name,action in [('hash_mismatch',lambda:hashes(base,{'payload.csv':'0'*64})),
            ('missing_input',lambda:hashes(base,{'absent.csv':'0'*64})),
            ('mixed_experiment_id',lambda:identity([{'run_id':'r','experiment_id':'pooled'},{'run_id':'r','experiment_id':'loco_eth'}],'r','pooled'))]:
            try:action()
            except ValueError as exc:result.append({'fixture':name,'detected':True,'error':str(exc),'would_exit_nonzero':True})
            else:raise AssertionError('negative fixture accepted '+name)
        for name,body in [('missing_tikz',r'\begin{tikzpicture}\node {test};\end{tikzpicture}'),('missing_table',r'\input{absent_table.tex}')]:
            source=base/name;source.mkdir();(source/'main.tex').write_text(r'\documentclass{article}\begin{document}'+body+r'\end{document}')
            try:clean_build(source,'main.tex',dest/name)
            except ValueError as exc:result.append({'fixture':name,'detected':True,'error':str(exc),'would_exit_nonzero':True})
            else:raise AssertionError('broken TeX accepted '+name)
    for record in result:
        command=[sys.executable,str(Path(__file__).resolve()),'--negative-case',record['fixture']]
        child=subprocess.run(command,capture_output=True,text=True)
        if child.returncode==0:raise AssertionError('negative CLI returned success '+record['fixture'])
        record.update({'actual_exit_code':child.returncode,'command':command,'stderr_excerpt':child.stderr[-1000:]})
        record.pop('would_exit_nonzero',None)
    write(dest/'negative_fixtures.json',result);return result


def negative_case(name):
    with tempfile.TemporaryDirectory(prefix='streammc_negative_cli_') as directory:
        base=Path(directory);(base/'payload.csv').write_text('a,b\n1,2\n')
        if name=='hash_mismatch':hashes(base,{'payload.csv':'0'*64})
        elif name=='missing_input':hashes(base,{'absent.csv':'0'*64})
        elif name=='mixed_experiment_id':identity([{'run_id':'r','experiment_id':'pooled'},{'run_id':'r','experiment_id':'loco_eth'}],'r','pooled')
        else:
            body=r'\begin{tikzpicture}\node {test};\end{tikzpicture}' if name=='missing_tikz' else r'\input{absent_table.tex}'
            (base/'main.tex').write_text(r'\documentclass{article}\begin{document}'+body+r'\end{document}')
            clean_build(base,'main.tex',base/'logs')
    raise AssertionError('negative fixture was not rejected')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['fixtures','paper','release'],default='release')
    parser.add_argument('--negative-case',choices=['hash_mismatch','missing_input','mixed_experiment_id','missing_tikz','missing_table'])
    args=parser.parse_args()
    if args.negative_case:negative_case(args.negative_case);return
    out=output();dest=out/'audits';dest.mkdir(exist_ok=True)
    if args.mode=='fixtures':print(fixtures(dest));return
    if args.mode=='paper':
        paper=ROOT/'projects/stream_mc/paper/current/r01';records=[clean_build(paper,entry,paper/'build') for entry in ['preprint.tex','journal.tex','supplement.tex']];write(dest/'build_audit.json',records);return
    manifest=read(out/'release_manifest.json');hashes(out,manifest['files'])
    if manifest['run_id']!=config()['run_id']:raise ValueError('release run identity mismatch')
    print('RELEASE HASH VALIDATION PASS',len(manifest['files']))


if __name__=='__main__':main()
