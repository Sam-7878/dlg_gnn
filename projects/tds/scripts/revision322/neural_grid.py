"""Short symmetric validation-only neural tuning sensitivity."""
import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from original_neural import GIN, ModifiedGIN, predict, PROJECT, ORIGINAL, require_builder

SEED=7
EPOCHS=5
GRID=list(product((32,48,64),(.001,.003),(.3,.5)))


class SizedGIN(ModifiedGIN):
    def __init__(self,width,dropout):
        GIN.__init__(self,input_dim=3,hidden_dim=width,dropout=dropout)
        self.arm='gin_size'
        self.head[0]=nn.Linear(2*width+5,width)


def run(cohort,arm):
    require_builder()
    torch.set_num_threads(2)
    device=torch.device('cuda:1')
    frame=pd.read_csv(PROJECT/f'results/317/{cohort}_population.csv')
    mask={s:frame.split.eq(s).to_numpy() for s in ('train','validation','test')}
    labels=frame.label.to_numpy()
    tabular=arm=='mlp'
    if tabular:
        features=np.load(PROJECT/f'private/317/{cohort}_summary.npy')
        assert features.shape==(len(frame),24)
        features=StandardScaler().fit(features[mask['train']]).transform(features)
        sets={s:torch.tensor(np.column_stack((features[m],labels[m])),dtype=torch.float32) for s,m in mask.items()}
    else:
        path=ORIGINAL if cohort=='original' else PROJECT/'private/317/horizon7_graphs.pt'
        packed=torch.load(path,map_location='cpu',weights_only=False)['graphs']
        if cohort=='original':
            graphs=[Data(x=g['x'],edge_index=g['edge_index'],y=torch.tensor(float(g['label'])),
                         chain_id=torch.tensor(g['chain_index']),timestamp=torch.tensor(g['timestamp'])) for g in packed]
        else:
            graphs=packed
        assert len(graphs)==len(frame)
        assert all((int(g.y),int(g.chain_id),int(g.timestamp))==(r.label,r.chain_index,r.timestamp)
                   for g,r in zip(graphs,frame.itertuples(index=False)))
        size=np.log1p([[g.num_nodes,g.num_edges] for g in graphs])
        size=StandardScaler().fit(size[mask['train']]).transform(size)
        for g,v in zip(graphs,size):
            g.size_scaled=torch.tensor(v,dtype=torch.float32).reshape(1,2)
        sets={s:[graphs[i] for i in np.flatnonzero(m)] for s,m in mask.items()}
    pos=labels[mask['train']].mean()
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-pos)/pos,device=device,dtype=torch.float32))
    rows=[]
    best=-1
    for width,lr,dropout in GRID:
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model=(nn.Sequential(nn.Linear(24,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,1))
               if tabular else SizedGIN(width,dropout)).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
        trial_best=-1
        for epoch in range(1,EPOCHS+1):
            model.train()
            data=sets['train']
            batches=torch.randperm(len(data)).split(64) if tabular else DataLoader(data,batch_size=64,shuffle=True)
            for batch in batches:
                opt.zero_grad()
                if tabular:
                    batch=data[batch].to(device)
                    z,y=model(batch[:,:-1]).flatten(),batch[:,-1]
                else:
                    batch=batch.to(device)
                    z,y=model(batch),batch.y.float()
                criterion(z,y).backward()
                opt.step()
            _,valid=predict(model,sets['validation'],device,tabular)
            if valid['ap']>trial_best:
                trial_best=valid['ap']
                trial_epoch=epoch
            if valid['ap']>best:
                best=valid['ap']
                best_model=deepcopy(model)
                selected=len(rows)
        rows.append(dict(cohort=cohort,arm=arm,width=width,learning_rate=lr,dropout=dropout,
                         epoch=trial_epoch,validation_ap=trial_best,selected=False))
        print(cohort,arm,width,lr,dropout,trial_best,flush=True)
    rows[selected]['selected']=True
    scores,test=predict(best_model,sets['test'],device,tabular)
    rows[selected].update(test_ap=test['ap'],test_roc_auc=test['roc_auc'])
    out=PROJECT/'private/322/neural_grid'
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f'{cohort}_{arm}.csv',index=False)
    np.savez_compressed(out/f'{cohort}_{arm}_test.npz',label=labels[mask['test']],score=scores)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('cohort',choices=('original','horizon7'))
    p.add_argument('arm',choices=('gin_size','mlp'))
    a=p.parse_args()
    run(a.cohort,a.arm)
