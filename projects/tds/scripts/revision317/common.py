"""Task317 paths and immutable scientific helpers."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
PROJECT=ROOT/'projects/tds'
OUT=PROJECT/'results/317'
REPORTS=PROJECT/'reports/317'
PRIVATE=PROJECT/'private/317'
PAPER=PROJECT/'paper/submission_317'
RAW=Path(os.environ.get('TDS_RAW_ROOT',str(ROOT.parent.parent/'_data/gog_round7_upstream/Token Data')))
GRAPH=ROOT/'archive/tds/data/gog_scimain_v1/graph.pt'
GRAPH_HASH='067cbdd7d7c055da91dbed9c492ad5a099c35e178f718e53f9dfdabab908b1cd'
FROZEN=PROJECT/'results/main_final_v2'
SUPP=PROJECT/'results/314_applsci'
SEEDS=(7,17,27,37,47)
LIB=PROJECT/'scripts/revision314'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')


def original(name):
    spec=importlib.util.spec_from_file_location('_r314_'+name,LIB/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def packed():
    import torch
    if sha(GRAPH)!=GRAPH_HASH:raise RuntimeError('Frozen graph identity mismatch')
    return torch.load(GRAPH,map_location='cpu',weights_only=False)['graphs']


def frame_from(graphs):
    import pandas as pd
    rows=[]
    for i,g in enumerate(graphs):
        chain,contract,_=g['event_id'].split(':')
        rows.append(dict(row_id=i,event_id=g['event_id'],chain=chain,contract=contract,
            label=int(g['label']),timestamp=int(g['timestamp']),chain_index=int(g['chain_index']),
            original_split='train' if i<17021 else 'validation' if i<20668 else 'test'))
    return pd.DataFrame(rows)
