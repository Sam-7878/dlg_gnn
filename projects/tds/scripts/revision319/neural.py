"""Frozen R319 seven-day matched-input neural comparison."""
import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

from common import OUT, PRIVATE, HORIZON, SEEDS, PROJECT, sha, write, require_builder, original_module

GIN = original_module('historical_gin').CausalLocalGIN
summary_features = original_module('common').summary_features
ARMS = ('gin0','gin_size','gin_sum','mlp_full','mlp_no_size','mlp_topology_only')


class ModifiedGIN(GIN):
    def __init__(self, arm):
        super().__init__(input_dim=3)
        self.arm=arm
        self.head[0]=nn.Linear(99+(2 if arm=='gin_size' else 48),48)

    def forward(self,d):
        h=F.dropout(F.relu(self.conv1(d.x,d.edge_index)),self.dropout,self.training)
        h=F.dropout(F.relu(self.conv2(h,d.edge_index)),self.dropout,self.training)
        parts=[global_mean_pool(h,d.batch),global_max_pool(h,d.batch)]
        if self.arm=='gin_sum':parts.append(global_add_pool(h,d.batch))
        parts.append(F.one_hot(d.chain_id.long(),num_classes=3).float())
        if self.arm=='gin_size':parts.append(d.size_scaled.reshape(-1,2))
        return self.head(torch.cat(parts,dim=1)).squeeze(-1)


def predict(model,data,device,tabular=False):
    model.eval();ys=[];ps=[];loss=0.;n=0
    with torch.no_grad():
        batches=data.split(128) if tabular else DataLoader(data,batch_size=128,shuffle=False)
        for b in batches:
            if tabular:
                x,y=b[:, :-1].to(device),b[:,-1].to(device)
                z=model(x).squeeze(-1)
            else:
                b=b.to(device);y=b.y.float();z=model(b)
            loss+=F.binary_cross_entropy_with_logits(z,y,reduction='sum').item()
            n+=len(y);ys.extend(y.cpu().numpy());ps.extend(torch.sigmoid(z).cpu().numpy())
    y=np.array(ys,dtype=int);p=np.array(ps)
    return p,dict(loss=loss/n,ap=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=ARMS,required=True);parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    require_builder()
    torch.set_num_threads(4)
    frame=pd.read_csv(PROJECT/'results/317/horizon7_population.csv')
    graphs=torch.load(HORIZON,weights_only=False)['graphs']
    assert len(frame)==len(graphs)
    assert all(int(g.y)==int(y) and int(g.chain_id)==int(c) for g,y,c in zip(graphs,frame.label,frame.chain_index))
    # Counts are computed from the independently audited graph and standardized on train only.
    sz=np.array([[np.log1p(g.num_nodes),np.log1p(g.num_edges)] for g in graphs])
    tr=(frame.split=='train').to_numpy();sc=StandardScaler().fit(sz[tr]);sizes=sc.transform(sz)
    if args.arm=='gin_size':
        for g,v in zip(graphs,sizes):g.size_scaled=torch.tensor(v,dtype=torch.float32).reshape(1,2)
    write(OUT/'gin_size_scaler.json',dict(fit='train',mean=sc.mean_.tolist(),scale=sc.scale_.tolist(),input=['log1p_retained_nodes','log1p_retained_edges']))
    features=None;indices=None
    if args.arm.startswith('mlp'):
        cache=PRIVATE/'horizon7_summary.npy'
        x=np.load(cache) if cache.exists() else np.array([summary_features(g) for g in graphs])
        assert x.shape==(len(graphs),24)
        indices={'mlp_full':list(range(24)),'mlp_no_size':list(range(3,24)),'mlp_topology_only':list(range(12))+[21,22,23]}[args.arm]
        tabsc=StandardScaler().fit(x[tr][:,indices]);features=tabsc.transform(x[:,indices])
        write(OUT/f'{args.arm}_scaler.json',dict(fit='train',indices=indices,mean=tabsc.mean_.tolist(),scale=tabsc.scale_.tolist()))
    labels={s:frame.loc[frame.split==s,'label'].to_numpy(int) for s in ['train','validation','test']}
    posweight=float((labels['train']==0).sum()/labels['train'].sum())
    device=torch.device(args.device);tabular=args.arm.startswith('mlp')
    for seed in SEEDS:
        dest=OUT/'runs'/'horizon7'/args.arm/f'seed{seed}'
        if (dest/'metrics.json').exists():
            m=json.loads((dest/'metrics.json').read_text());assert m['test_sha256']==sha(dest/'test.csv');continue
        dest.mkdir(parents=True,exist_ok=True)
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        if tabular:
            model=nn.Sequential(nn.Linear(len(indices),48),nn.ReLU(),nn.Dropout(.3),nn.Linear(48,1))
            sets={s:torch.tensor(np.column_stack([features[frame.split==s],labels[s]]),dtype=torch.float32) for s in labels}
        else:
            model=GIN(input_dim=3) if args.arm=='gin0' else ModifiedGIN(args.arm)
            sets={s:[graphs[i] for i in frame.index[frame.split==s]] for s in labels}
        model.to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        history=[];best=-1.;best_epoch=0;stale=0;start=time.monotonic()
        for epoch in range(1,61):
            model.train();loss_sum=0.;nnn=0
            batches=torch.randperm(len(sets['train'])).split(64) if tabular else DataLoader(sets['train'],batch_size=64,shuffle=True)
            for b in batches:
                optim.zero_grad(set_to_none=True)
                if tabular:
                    batch=sets['train'][b]
                    x=batch[:,:-1].to(device);y=batch[:,-1].to(device);z=model(x).squeeze(-1)
                else:
                    b=b.to(device);y=b.y.float();z=model(b)
                loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(posweight,device=device))
                loss.backward();optim.step();loss_sum+=loss.item()*len(y);nnn+=len(y)
            valid,vm=predict(model,sets['validation'],device,tabular)
            # Additional train diagnostics run with forked RNG, preserving training shuffle/dropout stream.
            with torch.random.fork_rng(devices=[device.index] if device.type=='cuda' else []):
                _,tm=predict(model,sets['train'],device,tabular)
            history.append(dict(epoch=epoch,optimization_train_loss=loss_sum/nnn,train_loss=tm['loss'],train_ap=tm['ap'],train_roc_auc=tm['roc_auc'],validation_loss=vm['loss'],validation_ap=vm['ap'],validation_roc_auc=vm['roc_auc']))
            if vm['ap']>best+1e-6:
                best=vm['ap'];best_epoch=epoch;stale=0;state=copy.deepcopy({k:v.detach().cpu() for k,v in model.state_dict().items()})
            else:stale+=1
            print(args.arm,seed,epoch,round(vm['ap'],6),flush=True)
            if stale>=5:break
        model.load_state_dict(state)
        valid,vm=predict(model,sets['validation'],device,tabular)
        test,te=predict(model,sets['test'],device,tabular)
        for s,p in [('validation',valid),('test',test)]:
            f=frame.loc[frame.split==s,['sample_id','timestamp','label']].reset_index(drop=True);f['p_mean']=p
            f.to_csv(dest/f'{s}.csv',index=False)
        pd.DataFrame(history).to_csv(dest/'history.csv',index=False)
        checkpoint=PRIVATE/'checkpoints'/args.arm/f'seed{seed}.pt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(state=state,arm=args.arm,seed=seed,selected_epoch=best_epoch,indices=indices),checkpoint)
        write(dest/'metrics.json',dict(arm=args.arm,seed=seed,parameter_count=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,stopped_epoch=epoch,
            validation_ap=best,selected_validation=vm,test=te,train_prevalence=labels['train'].mean(),positive_weight=posweight,
            protocol_sha256=sha(OUT/'protocol_freeze_319.json'),builder_sha256=sha(OUT/'seven_day_builder_equivalence.json'),
            test_sha256=sha(dest/'test.csv'),checkpoint_sha256=sha(checkpoint),elapsed_seconds=time.monotonic()-start))
        print('COMPLETE',args.arm,seed,te['ap'],te['roc_auc'],flush=True)
        del model,optim;torch.cuda.empty_cache()


if __name__=='__main__':main()
