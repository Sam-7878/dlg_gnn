"""Serial orchestration; separate OS processes for prefix and long repeats."""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
from r01_common import ROOT,config,output,write,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['verification','analysis','runtime','generate','final','repair','all'],default='all');args=parser.parse_args();cfg=config();out=output();logs=out/'execution_logs';logs.mkdir(exist_ok=True)
    steps=[]
    if args.stage in ('verification','all'):
        steps+=[['projects/stream_mc/scripts/r01_source_freeze.py'],['-m','pytest','-q','tests/stream_mc','--junitxml='+str(out/'audits/pytest.xml')],
                ['projects/stream_mc/scripts/r01_validate.py','--mode','fixtures']]
    if args.stage in ('analysis','all'):steps+=[['projects/stream_mc/scripts/r01_analysis.py']]
    if args.stage in ('runtime','all'):
        steps+=[['projects/stream_mc/scripts/r01_stress.py'],['projects/stream_mc/scripts/r01_replay.py','--mode','offline']]
        for population in ('prefix','long'):
            for policy in cfg['runtime']['policies']:
                for repeat in range(cfg['runtime']['repeats'] if population=='prefix' else 1):
                    steps.append(['projects/stream_mc/scripts/r01_replay.py','--mode',population,'--policy',policy,'--repeat',str(repeat)])
    if args.stage in ('generate','all'):steps+=[['projects/stream_mc/scripts/r01_generate.py','--paper'],['projects/stream_mc/scripts/r01_validate.py','--mode','paper']]
    if args.stage=='generate':steps.append(['projects/stream_mc/scripts/r01_render.py'])
    if args.stage=='final':
        steps=[['projects/stream_mc/scripts/r01_identity_audit.py'],
               ['-m','pytest','-q','tests/stream_mc','--junitxml='+str(out/'audits/pytest_final.xml')],
               ['projects/stream_mc/scripts/r01_validate.py','--mode','fixtures'],
               ['projects/stream_mc/scripts/r01_stress.py','--mode','restart'],
               ['projects/stream_mc/scripts/r01_controls_timing.py'],
               ['projects/stream_mc/scripts/r01_final_audits.py'],
               ['projects/stream_mc/scripts/r01_generate.py','--paper'],
               ['projects/stream_mc/scripts/r01_validate.py','--mode','paper'],
               ['projects/stream_mc/scripts/r01_render.py']]
    if args.stage=='repair':
        steps=[['projects/stream_mc/scripts/r01_offline_v2.py'],
               ['projects/stream_mc/scripts/r01_final_audits.py'],
               ['projects/stream_mc/scripts/r01_generate.py','--paper'],
               ['projects/stream_mc/scripts/r01_validate.py','--mode','paper'],
               ['projects/stream_mc/scripts/r01_render.py']]
    records=[]
    (out/'audits').mkdir(exist_ok=True)
    for i,step in enumerate(steps):
        command=[sys.executable,'-u',*step];log=logs/f'{args.stage}_{i:02}.log';print('START',command,flush=True);start=time.time()
        with log.open('w') as handle:
            process=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT/'src')},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:handle.write(line);handle.flush();print(line.rstrip(),flush=True)
            code=process.wait()
        records.append({'command':command,'exit_code':code,'seconds':time.time()-start,'log':str(log.relative_to(out)),'log_sha256':sha256(log)})
        write(out/f'execution_{args.stage}.json',records)
        if code:raise SystemExit(code)
    print('CAMPAIGN PHASE COMPLETE',args.stage,flush=True)


if __name__=='__main__':main()
