"""Cluster-bootstrap sensitivity by shared dominant retained-edge endpoint."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

SCRIPTS=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SCRIPTS/'revision319'))
from common import PROJECT, RAW, sha

PRIVATE=PROJECT/'private/322'
PRIOR=PROJECT/'results/319'
RESULTS=PROJECT/'results/322'
INVALID={'','nan','none','null'}
ZERO='0x'+'0'*40


def chain_groups(chain):
    lineage=pd.read_csv(PROJECT/'private/317/sample_time_lineage.csv')
    pop=pd.read_csv(PROJECT/'results/317/horizon7_population.csv').query("split=='test' and chain==@chain")
    items=pop.merge(lineage[['row_id','contract']],left_on='sample_id',right_on='row_id',validate='one_to_one')
    assert len(items)==len(pop)
    rows=[]
    path=RAW/'transactions'/f'{chain}.zip'
    import json
    assert sha(path)==json.loads((PROJECT/f'results/317/{chain}_source_identity.json').read_text())['archive_sha256']
    with zipfile.ZipFile(path) as archive:
        members={Path(name).stem.lower():name for name in archive.namelist() if name.endswith('.csv')}
        for item in items.itertuples(index=False):
            offset=0
            tail=None
            with archive.open(members[item.contract]) as file:
                chunks=pd.read_csv(file,usecols=['timestamp','block_number','transaction_hash','from','to'],chunksize=100000)
                for chunk in chunks:
                    chunk['row']=np.arange(offset,offset+len(chunk))
                    offset+=len(chunk)
                    chunk=chunk[chunk.timestamp<=item.timestamp].copy()
                    for col in ('from','to'):
                        chunk[col]=chunk[col].astype(str).str.strip().str.lower()
                    chunk=chunk[~chunk['from'].isin(INVALID)&~chunk['to'].isin(INVALID)]
                    tail=chunk if tail is None else pd.concat([tail,chunk],ignore_index=True)
                    tail=tail.sort_values(['timestamp','block_number','transaction_hash','row']).tail(128)
            assert len(tail)>0
            endpoint=Counter(pd.concat([tail['from'],tail['to']]))
            for key in (item.contract,ZERO):
                endpoint.pop(key,None)
            if endpoint:
                top=min(endpoint,key=lambda key:(-endpoint[key],key))
                group=hashlib.sha256(f'{chain}:{top}'.encode()).hexdigest()
            else:
                group=f'singleton-{item.sample_id}'
            rows.append((item.sample_id,group,len(tail)))
    return rows


def main():
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows=[row for chunk in pool.map(chain_groups,['ethereum','bsc','polygon']) for row in chunk]
    groups=pd.DataFrame(rows,columns=['sample_id','group','edge_count'])
    counts=pd.read_csv(PRIOR/'cutoff_admissible_counts.csv').query("cohort=='horizon7'")
    check=groups.merge(counts[['sample_id','retained_edges']],validate='one_to_one')
    assert len(check)==len(groups) and (check.edge_count==check.retained_edges).all()
    pop=pd.read_csv(PROJECT/'results/317/horizon7_population.csv').query("split=='test'").reset_index(drop=True)
    pop=pop.merge(groups,on='sample_id',validate='one_to_one')
    assert len(pop)==3627
    pop[['sample_id','group']].to_csv(PRIVATE/'counterparty_groups.csv',index=False)
    paths={
        'node_lr':[PRIOR/'predictions/horizon7_single_retained_nodes_test.csv'],
        'gin_size':[PRIOR/f'runs/horizon7/gin_size/seed{seed}/test.csv' for seed in (7,17,27,37,47)],
        'mlp':[PRIOR/f'runs/horizon7/mlp_full/seed{seed}/test.csv' for seed in (7,17,27,37,47)],
    }
    scores={}
    for arm,files in paths.items():
        scores[arm]=[]
        for f in files:
            frame=pd.read_csv(f)
            assert np.array_equal(frame[['sample_id','timestamp','label']],pop[['sample_id','timestamp','label']])
            scores[arm].append(frame['score' if 'score' in frame else 'p_mean'].to_numpy())
    y=pop.label.to_numpy()
    clusters=list(pop.groupby('group').indices.values())
    rng=np.random.default_rng(322)
    draws={arm:[] for arm in scores}
    for _ in range(2000):
        indices=np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))])
        if len(np.unique(y[indices]))<2:
            continue
        for arm,values in scores.items():
            draws[arm].append(np.mean([average_precision_score(y[indices],v[indices]) for v in values]))
    summary=[]
    for arm,values in scores.items():
        summary.append(dict(arm=arm,n=len(y),positive=int(y.sum()),clusters=len(clusters),largest=max(map(len,clusters)),
                            observed=np.mean([average_precision_score(y,v) for v in values]),
                            draws=len(draws[arm]),low=np.quantile(draws[arm],.025),high=np.quantile(draws[arm],.975)))
    pd.DataFrame(summary).to_csv(RESULTS/'counterparty_cluster_bootstrap.csv',index=False)


if __name__=='__main__':
    main()
