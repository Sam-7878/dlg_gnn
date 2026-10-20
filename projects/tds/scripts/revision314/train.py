"""Supplemental, validation-selected five-seed experiment matrix.

All experiment choices are frozen before this script evaluates any new test
score. Historical results are never overwritten. Each run saves its complete
epoch trajectory, selected checkpoint and one test prediction pass.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

from common import GRAPH_HASH, OUT, SEEDS, load, sha, summary_features, write_json
from historical_gin import CausalLocalGIN
from historical_models import FraudSAGEBaseline, class_balanced_focal_loss
from historical_metrics import classification_metrics

ARMS = ('gin_focal', 'sage_bce', 'summary_lr', 'summary_mlp', 'gin_no_chain', 'gin_bce', 'sage_focal')


class NoChainGIN(CausalLocalGIN):
    def __init__(self, input_dim):
        super().__init__(input_dim=input_dim)
        self.head[0] = torch.nn.Linear(96, 48)

    def forward(self, data):
        h = F.dropout(F.relu(self.conv1(data.x, data.edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, data.edge_index)), self.dropout, self.training)
        pooled = torch.cat([global_mean_pool(h, data.batch), global_max_pool(h, data.batch)], dim=1)
        return self.head(pooled).squeeze(-1)


def freeze():
    protocol = dict(dataset_sha256=GRAPH_HASH, seeds=SEEDS, arms=ARMS,
                    split_boundaries=[17021,20668,24316], batch_size=64,
                    hidden_dim=48, dropout=.3, learning_rate=.001,weight_decay=.0001,
                    max_epochs=60, early_stopping_patience=5, selection='validation AP; improvement > 1e-6',
                    bce='BCEWithLogits with pos_weight=train negatives/train positives',
                    focal='same weighted BCE multiplied by (1-p_t)^2',
                    summary_features='node/edge count,density,capped in/out/total degree mean/median/max; full-history log-degree feature mean/median/max; chain one-hot',
                    summary_scaler='StandardScaler fit exclusively on train',
                    lr='L2 C=1 max_iter=2000 class_weight=balanced train only',
                    test_policy='one prediction pass after validation-only selection; never tune from test',
                    runtime='current Python/PyTorch differs from historical; rerun both original loss cells for an internally matched supplemental matrix',
                    timestamp_sensitivity='strict earlier timestamp read-only checkpoint stress test; preserve main split')
    path=OUT/'supplement/protocol_freeze.json'
    encoded=json.dumps(protocol,sort_keys=True,indent=2)+'\n'
    if path.exists() and path.read_text()!=encoded:
        raise RuntimeError('Supplemental protocol changed')
    path.write_text(encoded)


def predict(model, data, device, tabular):
    model.eval()
    values=[]
    with torch.no_grad():
        if tabular:
            for batch in data.split(128):
                values.extend(torch.sigmoid(model(batch.to(device)).squeeze(-1)).cpu().tolist())
        else:
            for batch in DataLoader(data,batch_size=128,shuffle=False):
                values.extend(torch.sigmoid(model(batch.to(device))).cpu().tolist())
    return np.array(values)


def run_one(arm, seed, datasets, frame, features, device):
    run=OUT/'runs'/arm/f'seed{seed}'
    run.mkdir(parents=True,exist_ok=True)
    if (run/'metrics.json').exists():
        m=json.loads((run/'metrics.json').read_text())
        assert sha(run/'checkpoint.pt')==m['checkpoint_sha256']
        assert sha(run/'test.csv')==m['test_predictions_sha256']
        return m
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    labels={s:frame.loc[frame.split==s,'label'].to_numpy(int) for s in datasets}
    pw=float((labels['train']==0).sum()/labels['train'].sum())
    start=time.monotonic()
    history=[]; best_epoch=0
    if arm=='summary_lr':
        model=LogisticRegression(C=1, max_iter=2000, class_weight='balanced',random_state=seed)
        model.fit(features['train'],labels['train'])
        valid=model.predict_proba(features['validation'])[:,1]
        test=model.predict_proba(features['test'])[:,1]
        best_ap=float(average_precision_score(labels['validation'],valid))
        state={'coef':model.coef_,'intercept':model.intercept_,'n_iter':model.n_iter_}
        stopped=int(model.n_iter_.max());convergence=stopped<2000
    else:
        tabular=arm=='summary_mlp'
        if tabular:
            model=torch.nn.Sequential(torch.nn.Linear(features['train'].shape[1],48),torch.nn.ReLU(),
                                      torch.nn.Dropout(.3),torch.nn.Linear(48,1))
            data={s:torch.tensor(features[s],dtype=torch.float32) for s in datasets}
        else:
            input_dim=datasets['train'][0].x.shape[1]
            model=NoChainGIN(input_dim) if arm=='gin_no_chain' else FraudSAGEBaseline(input_dim=input_dim) if arm.startswith('sage') else CausalLocalGIN(input_dim=input_dim)
            data=datasets
        model=model.to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best_ap=-1;best_state=None;stale=0
        for epoch in range(1,61):
            model.train();loss_sum=0.;n=0
            if tabular:
                batches=torch.randperm(len(data['train'])).split(64)
            else:
                batches=DataLoader(data['train'],batch_size=64,shuffle=True)
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                if tabular:
                    x=data['train'][batch].to(device)
                    y=torch.tensor(labels['train'][batch.numpy()],dtype=torch.float32,device=device)
                    logits=model(x).squeeze(-1)
                else:
                    batch=batch.to(device);y=batch.y.float();logits=model(batch)
                loss=class_balanced_focal_loss(logits,y,pw) if arm.endswith('focal') else F.binary_cross_entropy_with_logits(logits,y,pos_weight=torch.tensor(pw,device=device))
                loss.backward();optimizer.step()
                loss_sum+=float(loss.detach())*len(y);n+=len(y)
            val=predict(model,data['validation'],device,tabular)
            ap=float(average_precision_score(labels['validation'],val))
            history.append(dict(epoch=epoch,train_loss=loss_sum/n,validation_ap=ap))
            if ap>best_ap+1e-6:
                best_ap=ap;best_epoch=epoch;stale=0
                best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else: stale+=1
            print(f'{arm} seed={seed} epoch={epoch} val_ap={ap:.6f}',flush=True)
            if stale>=5: break
        stopped=epoch;convergence=stale>=5
        model.load_state_dict(best_state)
        valid=predict(model,data['validation'],device,tabular)
        test=predict(model,data['test'],device,tabular)
        state=best_state
        del model,optimizer
        if torch.cuda.is_available():torch.cuda.empty_cache()
    checkpoint=dict(model_state_dict=state,arm=arm,seed=seed,best_epoch=best_epoch,
                    best_validation_ap=best_ap,dataset_sha256=GRAPH_HASH)
    torch.save(checkpoint,run/'checkpoint.pt')
    pd.DataFrame(history,columns=['epoch','train_loss','validation_ap']).to_csv(run/'history.csv',index=False)
    for split,p in [('validation',valid),('test',test)]:
        f=frame.loc[frame.split==split,['event_id','timestamp','label']].reset_index(drop=True)
        f['p_mean']=p;f['seed']=seed;f['arm']=arm
        f.to_csv(run/f'{split}.csv',index=False)
    result=dict(arm=arm,seed=seed,best_epoch=best_epoch,stopped_epoch=stopped,
                max_epoch_hit=(stopped==60 if arm!='summary_lr' else stopped==2000),
                convergence_plateau_or_solver_converged=convergence,best_validation_ap=best_ap,
                dataset_sha256=GRAPH_HASH,test_access_count=1,
                checkpoint_sha256=sha(run/'checkpoint.pt'),test_predictions_sha256=sha(run/'test.csv'),
                elapsed_seconds=time.monotonic()-start,device=str(device),
                **classification_metrics(labels['test'],test))
    write_json(run/'metrics.json',result)
    print('COMPLETE',arm,seed,'AP',result['auc_pr'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arms',default=','.join(ARMS));parser.add_argument('--seeds',default=','.join(map(str,SEEDS)));parser.add_argument('--device',default='cuda:1')
    args=parser.parse_args();freeze();torch.set_num_threads(4)
    _,frame,graphs=load()
    datasets={s:[graphs[i] for i in frame.index[frame.split==s]] for s in ['train','validation','test']}
    print('Building shared summary features',flush=True)
    x=np.asarray([summary_features(g) for g in graphs],dtype=float)
    scaler=StandardScaler().fit(x[frame.split=='train'])
    features={s:scaler.transform(x[frame.split==s]) for s in datasets}
    write_json(OUT/'supplement/summary_scaler.json',dict(fit_split='train',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist()))
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('device',device,flush=True)
    rows=[run_one(a,int(s),datasets,frame,features,device) for a in args.arms.split(',') for s in args.seeds.split(',')]
    pd.DataFrame(rows).to_csv(OUT/'supplement/per_seed_results.csv',index=False)
    print('Supplemental matrix complete',len(rows),flush=True)


if __name__=='__main__':
    main()
