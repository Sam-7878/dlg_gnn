"""Fixed-configuration sensitivity at additional observation windows."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch_geometric.loader import DataLoader

from original_neural import ModifiedGIN, predict, summary_features, PROJECT

WINDOWS=PROJECT/'private/322/windows'
SEED=7


def run(day,arm,device):
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    frame=pd.read_csv(WINDOWS/f'{day}_population.csv')
    graphs=torch.load(WINDOWS/f'{day}_graphs.pt',map_location='cpu',weights_only=False)
    assert len(graphs)==len(frame)
    assert all(int(g.y)==r.label and int(g.chain_id)==r.chain_index for g,r in zip(graphs,frame.itertuples(index=False)))
    split={name:frame.split.eq(name).to_numpy() for name in ('train','validation','test')}
    label=frame.label.to_numpy()
    size=np.log1p([[g.num_nodes,g.num_edges] for g in graphs])
    record=dict(days=day,arm=arm,seed=SEED,n=int(split['test'].sum()),positive=int(label[split['test']].sum()))
    if arm=='node_lr':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=SEED))
        model.fit(size[split['train'],:1],label[split['train']])
        test=model.predict_proba(size[split['test'],:1])[:,1]
    else:
        tabular=arm=='mlp'
        if tabular:
            features=np.asarray([summary_features(g) for g in graphs])
            features=StandardScaler().fit(features[split['train']]).transform(features)
            sets={key:torch.tensor(np.column_stack((features[mask],label[mask])),dtype=torch.float32) for key,mask in split.items()}
            model=nn.Sequential(nn.Linear(24,48),nn.ReLU(),nn.Dropout(.3),nn.Linear(48,1))
        else:
            scaled=StandardScaler().fit(size[split['train']]).transform(size)
            for g,x in zip(graphs,scaled):
                g.size_scaled=torch.tensor(x,dtype=torch.float32).reshape(1,2)
            sets={key:[graphs[i] for i in np.flatnonzero(mask)] for key,mask in split.items()}
            model=ModifiedGIN('gin_size')
        model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        pos=label[split['train']].mean()
        criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1-pos)/pos,device=device,dtype=torch.float32))
        best=-1
        stale=0
        history=[]
        for epoch in range(1,61):
            model.train()
            training=sets['train']
            batches=torch.randperm(len(training)).split(64) if tabular else DataLoader(training,batch_size=64,shuffle=True)
            for batch in batches:
                optimizer.zero_grad()
                if tabular:
                    batch=training[batch].to(device)
                    z,y=model(batch[:,:-1]).squeeze(-1),batch[:,-1]
                else:
                    batch=batch.to(device)
                    z,y=model(batch),batch.y.float()
                loss=criterion(z,y)
                loss.backward()
                optimizer.step()
            _,valid=predict(model,sets['validation'],device,tabular)
            history.append(dict(epoch=epoch,validation_ap=valid['ap'],validation_roc_auc=valid['roc_auc']))
            print(day,arm,epoch,valid['ap'],flush=True)
            if valid['ap']>best+1e-6:
                best=valid['ap']
                best_state=deepcopy(model.state_dict())
                record.update(best_epoch=epoch,validation_ap=best)
                stale=0
            else:
                stale+=1
                if stale==5:
                    break
        model.load_state_dict(best_state)
        pd.DataFrame(history).to_csv(WINDOWS/f'{day}_{arm}_history.csv',index=False)
        test,_=predict(model,sets['test'],device,tabular)
    y=label[split['test']]
    record.update(prevalence=float(y.mean()),ap=float(average_precision_score(y,test)),roc_auc=float(roc_auc_score(y,test)))
    np.savez_compressed(WINDOWS/f'{day}_{arm}_test.npz',label=y,score=test)
    (WINDOWS/f'{day}_{arm}.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('days',type=int,choices=(1,3,14))
    parser.add_argument('arm',choices=('node_lr','mlp','gin_size'))
    args=parser.parse_args()
    run(args.days,args.arm,torch.device('cuda:1'))
