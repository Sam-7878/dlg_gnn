"""New first+7d cohort, independently rebuilt from raw transfers."""
import collections
import concurrent.futures
import json
import time
import zipfile
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from common import OUT,PRIVATE,RAW,sha,write
from protocol import register

INVALID={'','nan','none','null'}
KEYS=['timestamp','block_number','transaction_hash','original_row_index']


def build(chain,rows):
    cache=PRIVATE/f'horizon7_{chain}.pt'
    if cache.exists():return torch.load(cache,weights_only=False)
    results=[];start=time.monotonic()
    with zipfile.ZipFile(RAW/'transactions'/f'{chain}.zip') as z:
        index={n.rsplit('/',1)[-1].removesuffix('.csv').lower():n for n in z.namelist() if n.endswith('.csv')}
        for j,row in enumerate(rows.itertuples(),1):
            inc=collections.Counter();out=collections.Counter();tail=None;nraw=0;count=0;max_t=-1
            with z.open(index[row.contract]) as f:
                for part in pd.read_csv(f,usecols=['block_number','from','to','transaction_hash','timestamp'],chunksize=100000):
                    part['original_row_index']=np.arange(nraw,nraw+len(part));nraw+=len(part)
                    for key in ['from','to']:part[key]=part[key].astype(str).str.strip().str.lower()
                    part=part[~part['from'].isin(INVALID)&~part['to'].isin(INVALID)]
                    part['timestamp']=pd.to_numeric(part.timestamp,errors='raise').astype('int64')
                    part['block_number']=pd.to_numeric(part.block_number,errors='raise').astype('int64')
                    part=part[part.timestamp<=row.timestamp]
                    if len(part):
                        max_t=max(max_t,int(part.timestamp.max()));count+=len(part)
                        inc.update(part['to']);out.update(part['from'])
                        tail=pd.concat([tail,part],ignore_index=True) if tail is not None else part
                        tail=tail.sort_values(KEYS,kind='stable').tail(128)
            assert count>=1 and max_t<=row.timestamp
            nodes=sorted(set(tail['from'])|set(tail['to']));mapping={v:i for i,v in enumerate(nodes)}
            a=torch.tensor([inc[v] for v in nodes],dtype=torch.float32);b=torch.tensor([out[v] for v in nodes],dtype=torch.float32)
            x=torch.stack([torch.log1p(a),torch.log1p(b),torch.log1p(a+b)],1)
            edge=torch.tensor([[mapping[v] for v in tail['from']],[mapping[v] for v in tail['to']]],dtype=torch.long)
            graph=Data(x=x,edge_index=edge,y=torch.tensor(float(row.label)),chain_id=torch.tensor(int(row.chain_index)),timestamp=torch.tensor(int(row.timestamp)))
            results.append((row.row_id,graph,dict(sample_id=row.row_id,valid_rows_through_cutoff=count,max_input_timestamp=max_t,cutoff=int(row.timestamp),nodes=len(nodes),edges=len(tail))))
            if j%250==0:print('horizon7',chain,j,'/',len(rows),'elapsed_s',round(time.monotonic()-start),flush=True)
    torch.save(results,cache);return results


def main():
    p=register()
    decision=json.loads((OUT/'selection_decision.json').read_text())['decision']
    assert decision==p['finite_horizon_condition'],'No conditional fitting without material diagnostic change'
    frame=pd.read_csv(PRIVATE/'sample_time_lineage.csv')
    frame['timestamp']=frame.first_valid_tx_time+7*86400
    # Category inheritance is retrospective; inclusion does not require later activity.
    frame=frame[frame.timestamp<=1709251199].sort_values(['timestamp','event_id'],kind='stable').copy()
    chain_index={'ethereum':0,'bsc':1,'polygon':2}
    # Recover actual packed chain coding rather than assume a new order.
    from common import frame_from,packed
    codes=frame_from(packed()).groupby('chain').chain_index.first().to_dict()
    frame['chain_index']=frame.chain.map(codes)
    n=len(frame);a=int(.7*n);b=int(.85*n)
    frame['split']=np.repeat(['train','validation','test'],[a,b-a,n-b])
    frame.to_csv(PRIVATE/'horizon7_population.csv',index=False)
    public=frame[['row_id','chain','label','timestamp','chain_index','split']].rename(columns={'row_id':'sample_id'})
    public.to_csv(OUT/'horizon7_population.csv',index=False)
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(build,c,g) for c,g in frame.groupby('chain')]
        for job in concurrent.futures.as_completed(jobs):results.extend(job.result())
    by_id={i:(g,a) for i,g,a in results};assert set(by_id)==set(frame.row_id)
    graphs=[by_id[i][0] for i in frame.row_id];audits=[by_id[i][1] for i in frame.row_id]
    torch.save(dict(graphs=graphs),PRIVATE/'horizon7_graphs.pt')
    pd.DataFrame(audits).to_csv(OUT/'horizon7_feature_audit.csv',index=False)
    supports=public.groupby(['split','chain'],sort=False).label.agg(n='size',positive='sum',prevalence='mean').reset_index()
    supports.to_csv(OUT/'horizon7_support.csv',index=False)
    write(OUT/'horizon7_dataset_manifest.json',dict(benchmark_id='GoG-317-first7d-v1',n=n,removed=24316-n,
        removed_positive=int(7198-frame.label.sum()),population_sha256=sha(OUT/'horizon7_population.csv'),
        graph_sha256=sha(PRIVATE/'horizon7_graphs.pt'),split_boundaries=[a,b,n],feature_audit_sha256=sha(OUT/'horizon7_feature_audit.csv'),
        test_comparable_to_original=False,paired_cross_cohort_inference=False,label_time_known=False,
        endpoint_policy='observed maximum support; documented exact archive stopping policy unavailable'))
    print('HORIZON7 READY',n,flush=True)


if __name__=='__main__':main()
