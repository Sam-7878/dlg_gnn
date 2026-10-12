"""Record actual subprocess exits/logs/source identities, including failed attempts."""
from __future__ import annotations
import argparse
import datetime
import os
import subprocess
import sys
import time
from r02_common import ROOT,PROJECT,VERIFY,RUN,write,sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['numeric','systems','tests','tables','paper','public']);args=parser.parse_args()
    VERIFY.mkdir(parents=True,exist_ok=True)
    if args.stage=='tests':command=[sys.executable,'-m','pytest','-q','tests/stream_mc','--junitxml='+str(VERIFY/'test_results.xml')]
    else:
        command=[sys.executable,str(PROJECT/'scripts'/f'r02_{args.stage}.py')]
        if args.stage=='public':command+=['--full-policy','--output',str(VERIFY/'public_local_numeric')]
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log=VERIFY/f'{args.stage}_{stamp}.log';begin=time.perf_counter()
    with log.open('w') as handle:
        child=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT/'src')},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in child.stdout:handle.write(line);handle.flush();print(line,end='',flush=True)
        code=child.wait()
    record={'verification_id':RUN,'stage':args.stage,'command':command,'actual_exit_code':code,'wall_seconds':time.perf_counter()-begin,
        'log_sha256':sha256(log),'log_path':str(log.relative_to(ROOT)),'utc_execution_stamp':stamp,
        'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (PROJECT/'scripts').glob('r02_*.py')},
        'scope':'actual same-host subprocess verification; not new training or external reviewer approval'}
    write(VERIFY/f'execution_{args.stage}_{stamp}.json',record)
    with (VERIFY/'verification_commands.log').open('a') as handle:handle.write(f'{stamp} exit={code} '+repr(command)+'\n')
    raise SystemExit(code)


if __name__=='__main__':main()
