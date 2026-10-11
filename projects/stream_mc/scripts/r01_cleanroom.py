"""Fresh stdlib-only venv/workspace check; not an external reviewer/download."""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path
from r01_common import PROJECT,ROOT,write,sha256
from r01_public import verified_archive
from r01_validate import clean_build


def safe_extract(archive,destination):
    for name in archive.namelist():
        if Path(name).is_absolute() or '..' in Path(name).parts or '\\' in name:raise ValueError('unsafe private source ZIP path')
    archive.extractall(destination)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--paper',action='store_true');args=parser.parse_args()
    source=PROJECT/'evidence/r01_numeric_evidence.zip';expected=PROJECT/'evidence/r01_release.json';report={'archive_sha256':sha256(source),'scope':'same agent, fresh standard-library-only venv and clean workspace on same host; local copy, not public re-download or external independent reviewer','reviewer':'Codex implementation-side verification','external_independent_approval':False}
    with tempfile.TemporaryDirectory(prefix='streammc_cleanroom_') as directory:
        base=Path(directory);repo=base/'dlg_gnn';project=repo/'projects/stream_mc';project.mkdir(parents=True)
        with verified_archive()[0] as archive:
            for name in archive.namelist():
                if name.startswith('code/'):
                    path=repo/name[5:];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.read(name))
            (project/'configs').mkdir();(project/'configs/r01.json').write_bytes(archive.read('configs/r01.json'))
        (project/'evidence').mkdir();shutil.copy2(source,project/'evidence/r01_numeric_evidence.zip');shutil.copy2(expected,project/'evidence/r01_release.json')
        venv.EnvBuilder(with_pip=False).create(base/'venv');python=base/'venv/bin/python'
        logs=[]
        for mode in ('verify','tables'):
            command=[str(python),str(project/'scripts/r01_public.py'),'--mode',mode]
            result=subprocess.run(command,cwd=repo,text=True,capture_output=True)
            logs.append({'mode':mode,'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
            if result.returncode:raise ValueError('cleanroom numeric failure: '+result.stderr)
        report['commands']=logs;report['stdlib_only_numeric_verification']='PASS';report['no_original_training_inputs_or_scientific_packages_needed']=True
        report['recomputed_tables']={p.name:sha256(p) for p in (project/'results/public_recomputed_r01').iterdir()}
        if args.paper:
            builds=[]
            for label in ('preprint','journal'):
                package=PROJECT/f'paper/current/r01/packages/{label}_package.zip';target=base/label;target.mkdir()
                with zipfile.ZipFile(package) as archive:safe_extract(archive,target)
                for entry in (label+'.tex','supplement.tex'):builds.append(clean_build(target,entry,target/'clean_build'))
            report['author_local_private_source_builds']=builds
            report['paper_scope']='fresh local source extraction and real compiler; not independently received author release'
    write(PROJECT/'evidence/cleanroom_r01.json',report);print('CLEANROOM PASS',report['archive_sha256'],flush=True)


if __name__=='__main__':main()
