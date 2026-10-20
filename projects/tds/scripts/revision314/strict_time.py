"""Reconstruct strict t<cutoff test graphs directly from retained source ZIPs.

Source rows are streamed in chunks; full-history degrees and last-128 topology
are reconstructed independently. Inclusive reconstruction must exactly match
every packed test tensor before strict-time checkpoint sensitivity is admitted.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import zipfile

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from common import FROZEN, OUT, PROJECT, ROOT, RAW_ROOT, SEEDS, load, sha, write_json
from historical_gin import CausalLocalGIN
from historical_models import FraudSAGEBaseline, TGATBaseline, TemporalMemoryBaseline
from historical_metrics import classification_metrics

INVALID={'','nan','none','null'}
KEYS=['timestamp','block_number','transaction_hash','original_row_index']


def make_graph(tail, incoming, outgoing, row):
    nodes=sorted(set(tail['from'])|set(tail['to'])) if len(tail) else ['__empty__']
    mapping={v:i for i,v in enumerate(nodes)}
    ideg=torch.tensor([incoming[x] for x in nodes],dtype=torch.float32)
    odeg=torch.tensor([outgoing[x] for x in nodes],dtype=torch.float32)
    x=torch.stack([torch.log1p(ideg),torch.log1p(odeg),torch.log1p(ideg+odeg)],dim=1)
    edges=torch.tensor([[mapping[v] for v in tail['from']],[mapping[v] for v in tail['to']]],dtype=torch.long)
    return Data(x=x,edge_index=edges,y=torch.tensor(float(row.label)),
                chain_id=torch.tensor(int(row.chain_index)),timestamp=torch.tensor(int(row.timestamp)))


def source_graph(z, filename, row):
    tails=[None,None]
    indeg=[collections.Counter(),collections.Counter()]
    outdeg=[collections.Counter(),collections.Counter()]
    nrows=0;n_equal=0;total=0;available_columns=None
    with z.open(filename) as handle:
        for frame in pd.read_csv(handle,chunksize=100000,usecols=['block_number','from','to','transaction_hash','timestamp']):
            available_columns=list(frame.columns)
            frame['original_row_index']=np.arange(nrows,nrows+len(frame));nrows+=len(frame)
            for key in ['from','to']:
                frame[key]=frame[key].astype(str).str.strip().str.lower()
            frame=frame[~frame['from'].isin(INVALID)&~frame['to'].isin(INVALID)]
            frame['timestamp']=pd.to_numeric(frame.timestamp).astype('int64')
            frame['block_number']=pd.to_numeric(frame.block_number).astype('int64')
            n_equal+=int((frame.timestamp==row.timestamp).sum());total+=len(frame)
            for mode,mask in enumerate([frame.timestamp<=row.timestamp,frame.timestamp<row.timestamp]):
                f=frame.loc[mask]
                indeg[mode].update(f['to']);outdeg[mode].update(f['from'])
                selected=pd.concat([tails[mode],f],ignore_index=True) if tails[mode] is not None else f
                tails[mode]=selected.sort_values(KEYS,kind='stable').tail(128)
    return (make_graph(tails[0],indeg[0],outdeg[0],row),
            make_graph(tails[1],indeg[1],outdeg[1],row),
            dict(event_id=row.event_id,raw_valid_rows=total,rows_at_cutoff=n_equal,
                 retained_equal_timestamp_edges=int((tails[0].timestamp==row.timestamp).sum()),
                 strictly_earlier_rows=total-n_equal,source_columns=','.join(available_columns),
                 strict_max_timestamp=int(tails[1].timestamp.max()) if len(tails[1]) else -1))


def build_chain(chain, rows, packed):
    zip_path=RAW_ROOT/'transactions'/f'{chain}.zip'
    results=[]
    with zipfile.ZipFile(zip_path) as z:
        index={n.rsplit('/',1)[-1].removesuffix('.csv').lower():n for n in z.namelist() if n.endswith('.csv')}
        for count,row in enumerate(rows.itertuples(),1):
            inclusive,strict,audit=source_graph(z,index[row.contract],row)
            original=packed[row.Index]
            audit['inclusive_x_exact']=bool(torch.equal(inclusive.x,original.x))
            audit['inclusive_edges_exact']=bool(torch.equal(inclusive.edge_index,original.edge_index))
            if not audit['inclusive_x_exact'] or not audit['inclusive_edges_exact']:
                raise RuntimeError(f'Inclusive reconstruction mismatch {row.event_id}; no strict sensitivity admitted')
            assert audit['strict_max_timestamp']<row.timestamp
            results.append((row.Index,strict,audit))
            if count%100==0:print('source strict reconstruction',chain,count,'/',len(rows),flush=True)
    return results


def infer(kind,model,graphs,device,tmin,scale):
    model.eval();values=[]
    memory=model.initial_memory(device) if kind=='tgn' else None
    with torch.no_grad():
        for batch in DataLoader(graphs,batch_size=1 if kind=='tgn' else 128,shuffle=False):
            batch=batch.to(device)
            t=(batch.timestamp.float()-float(tmin))/float(scale)
            if kind=='tgn':logits,memory=model.forward_event(batch,t,memory)
            elif kind=='tgat':logits=model(batch,t)
            else:logits=model(batch)
            values.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(values)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.set_num_threads(3)
    _,frame,packed=load();test=frame[frame.split=='test']
    cache=OUT/'supplement/strict_test_graphs.pt'
    if cache.exists():
        p=torch.load(cache,weights_only=False);strict=p['graphs']
    else:
        results=[]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(build_chain,c,r,packed) for c,r in test.groupby('chain')]
            for f in concurrent.futures.as_completed(futures):results.extend(f.result())
        results.sort(key=lambda x:x[0]);strict=[x[1] for x in results]
        pd.DataFrame([x[2] for x in results]).to_csv(OUT/'data/strict_time_source_audit.csv',index=False)
        torch.save(dict(graphs=strict),cache)
    rows=[];device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train=frame[frame.split=='train'];tmin=int(train.timestamp.min());scale=int(train.timestamp.max()-tmin)
    for kind in ['gin','tgat','tgn','fraudsage']:
        for seed in SEEDS:
            cls={'gin':CausalLocalGIN,'tgat':TGATBaseline,'tgn':TemporalMemoryBaseline,'fraudsage':FraudSAGEBaseline}[kind]
            model=cls(input_dim=packed[0].x.shape[1]).to(device)
            path=ROOT/f'archive/tds/data/gog_scimain_v1_preserved_panel/checkpoints/seed{seed}.pt' if kind=='gin' else FROZEN/f'checkpoints/{kind}/seed{seed}.pt'
            cp=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(cp['model_state_dict'])
            p=infer(kind,model,strict,device,tmin,scale)
            pred=test[['event_id','timestamp','label']].reset_index(drop=True).copy();pred['p_mean']=p
            dest=OUT/'supplement'/f'strict_{kind}_seed{seed}.csv';pred.to_csv(dest,index=False)
            base=FROZEN/f'raw_predictions/{"proposed" if kind=="gin" else kind}/seed{seed}_test.csv'
            from common import aligned_prediction
            original=classification_metrics(test.label.to_numpy(int),aligned_prediction(base,frame))
            m=classification_metrics(test.label.to_numpy(int),p)
            rows.append(dict(model=kind,seed=seed,checkpoint_sha256=sha(path),contract='t_edge < contract cutoff; full-history features rebuilt under same restriction',
                             original_auc_pr=original['auc_pr'],auc_pr_delta=m['auc_pr']-original['auc_pr'],**m))
            print('strict sensitivity',kind,seed,m['auc_pr'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'data/strict_time_sensitivity.csv',index=False)
    write_json(OUT/'reports/strict_time_manifest.json',dict(n_test=len(test),inclusive_reconstruction_exact=True,
        strict_graph_sha256=sha(cache),models=['gin','tgat','tgn','fraudsage'],seeds=SEEDS,
        policy='fixed checkpoints; diagnostic sensitivity, not retraining or test-based selection',
        upstream_intra_timestamp_order='block number and transaction hash present; transaction index absent; hash order is deterministic, not physical execution order'))


if __name__=='__main__':main()
