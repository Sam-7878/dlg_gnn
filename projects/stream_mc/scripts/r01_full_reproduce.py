"""Opt-in retraining from a disclosed pseudonymous bounded-snapshot input kit."""
from __future__ import annotations
import argparse
import hashlib
import io
import time
from pathlib import Path
from r01_common import PROJECT,config,write,sha256
from r01_public import verified_archive,rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--confirm-full',action='store_true');parser.add_argument('--seed',type=int);args=parser.parse_args()
    if not args.confirm_full:raise SystemExit('Expensive GPU retraining requires --confirm-full; no training started.')
    import numpy as np
    import torch
    from torch_geometric.data import Data
    import r01_train as training
    cfg=config();cfg['run_id']='selectivestream_r01_public_input_reproduction';cfg['output']='projects/stream_mc/results/public_input_reproduction'
    cfg.pop('config_sha256');dest=PROJECT/'results/public_input_reproduction'
    # A distinct output prevents overwriting or quietly resuming the approved R01 run.
    training.output=lambda:dest
    cost_records=[];cost_context={};dense_fit=training.fit_dense;logistic_class=training.LogisticRegression
    def timed_dense(x,y,*args,**kwargs):
        begin=time.perf_counter();model=dense_fit(x,y,*args,**kwargs)
        cost_records.append({**cost_context,'head':'degree_MLP' if x.shape[1]==11 else 'relational_MLP','seconds':time.perf_counter()-begin,'scope':'actual alias-input retraining probe; not original R01 head-training wall time'})
        return model
    class TimedLogistic(logistic_class):
        def fit(self,x,y,*args,**kwargs):
            begin=time.perf_counter();result=super().fit(x,y,*args,**kwargs)
            cost_records.append({**cost_context,'head':'degree_logistic','seconds':time.perf_counter()-begin,'scope':'actual alias-input retraining probe; not original R01 head-training wall time'})
            return result
    training.fit_dense=timed_dense;training.LogisticRegression=TimedLogistic
    cache={'graphs':{s:[] for s in ('train','validation','test')},'metadata':{s:[] for s in ('train','validation','test')}}
    with verified_archive()[0] as archive:
        metadata=rows(archive,'inputs/snapshot_manifest.csv')
        with np.load(io.BytesIO(archive.read('inputs/bounded_edges.npz')),allow_pickle=False) as arrays:
            edges=arrays['edges'];offsets=arrays['edge_offsets'];nodes=arrays['node_counts']
            for i,m in enumerate(metadata):
                split=m['split_id'];edge=edges[:,offsets[i]:offsets[i+1]].astype(np.int64);n=int(nodes[i]);label=int(m['label'])
                incoming=np.bincount(edge[1],minlength=n).astype(np.float32);outgoing=np.bincount(edge[0],minlength=n).astype(np.float32)
                features=np.stack([np.log1p(incoming),np.log1p(outgoing),np.log1p(incoming+outgoing)],axis=1)
                cache['graphs'][split].append(Data(x=torch.from_numpy(features),edge_index=torch.from_numpy(edge),y=torch.tensor([float(label)]),num_nodes=n))
                cache['metadata'][split].append({'sample_id':m['reference_alias'],'contract_id':m['contract_alias'],'chain':m['chain'],'event_start':int(m['event_start']),'event_end':int(m['snapshot_cutoff']),'label':label})
    seeds=[args.seed] if args.seed is not None else cfg['seeds']
    if any(s not in cfg['seeds'] for s in seeds):raise ValueError('seed outside frozen five-seed family')
    write(dest/'reproduction_config.json',cfg);cfg['config_sha256']=sha256(dest/'reproduction_config.json')
    for backbone in ('GIN','GATv2'):
        for seed in seeds:
            cost_context.update({'seed':seed,'backbone':backbone,'experiment_id':'pooled_snapshot','run_id':cfg['run_id']})
            training.train_one(cfg,cache,'pooled_snapshot',seed,backbone)
    for target in cfg['chains']:
        for seed in seeds:training.train_one(cfg,cache,'loco_'+target,seed,'GIN')
    if cost_records:write(dest/'training_cost_probe.json',cost_records)
    write(dest/'scope.json',{'source_run':'selectivestream_r01_20261011','run_id':cfg['run_id'],'seeds':seeds,'scope':'bounded graph input reproduction; stable lexicographic aliases preserve retrieval ties, not original address/file identities','not_original_provider_edge_lineage_reconstruction':True})


if __name__=='__main__':main()
