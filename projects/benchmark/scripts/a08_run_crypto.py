#!/usr/bin/env python3
"""One fresh-process A08 run; frozen data/config required."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.pipelines.benchmark_crypto import train_run
p=argparse.ArgumentParser();p.add_argument('--chain',required=True,choices=['polygon','bsc','ethereum']);p.add_argument('--model',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-n',type=int,default=0);a=p.parse_args()
cfg=json.loads((ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text())
status=train_run(ROOT,cfg,a.chain,a.model,a.seed,smoke=a.smoke,smoke_n=a.smoke_n)
raise SystemExit(0 if status=='SUPPORTED_EXACT' else 2)
