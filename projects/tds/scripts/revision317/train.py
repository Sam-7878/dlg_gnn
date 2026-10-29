"""Bounded preregistered controls, separate outputs and one test access/run."""
import argparse
import random
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool,global_max_pool
from common import OUT,PRIVATE,PROJECT,SEEDS,packed,frame_from,original,sha,write
from protocol import register

hm=original('historical_models');gin=original('historical_gin');metrics=original('historical_metrics')
summary_features=original('common').summary_features


class NoTimeGAT(hm.TGATBaseline):
    def __init__(self):
        super().__init__()
        del self.time
        self.head[0]=torch.nn.Linear(99,48)

    def forward(self,d,t=None):
        h=F.elu(self.conv1(d.x,d.edge_index));h=F.dropout(h,self.dropout,self.training)
        h=F.elu(self.conv2(h,d.edge_index))
        pooled=torch.cat([global_mean_pool(h,d.batch),global_max_pool(h,d.batch),F.one_hot(d.chain_id.long(),3).float()],1)
        return self.head(pooled).squeeze(-1)


def load_data(cohort):
    if cohort=='original':
        gs=packed();frame=frame_from(gs).rename(columns={'row_id':'sample_id','original_split':'split'})
        graphs=[Data(x=g['x'],edge_index=g['edge_index'],y=torch.tensor(float(g['label'])),chain_id=torch.tensor(g['chain_index']),timestamp=torch.tensor(g['timestamp'])) for g in gs]
    else:
        frame=pd.read_csv(OUT/'horizon7_population.csv')
        graphs=torch.load(PRIVATE/'horizon7_graphs.pt',weights_only=False)['graphs']
    features_cache=PRIVATE/f'{cohort}_summary.npy'
    if features_cache.exists():x=np.load(features_cache)
    else:
        x=np.asarray([summary_features(g) for g in graphs]);np.save(features_cache,x)
    assert len(graphs)==len(frame)==len(x)
    scaler=StandardScaler().fit(x[frame.split=='train'])
    write(OUT/f'{cohort}_summary_scaler.json',dict(fit='train',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist()))
    data={s:[graphs[i] for i in frame.index[frame.split==s]] for s in ['train','validation','test']}
    features={s:scaler.transform(x[frame.split==s]) for s in data}
    return frame,data,features


def predict(model,data,device,arm,tmin,scale):
    model.eval();values=[]
    with torch.no_grad():
        batches=data.split(128) if arm=='summary_mlp' else DataLoader(data,batch_size=128,shuffle=False)
        for b in batches:
            b=b.to(device)
            if arm=='summary_mlp':z=model(b).squeeze(-1)
            elif arm.startswith('gat'):z=model(b,(b.timestamp.float()-float(tmin))/float(scale))
            else:z=model(b)
            values.extend(torch.sigmoid(z).cpu().tolist())
    return np.asarray(values)


def run(cohort,arm,seed,frame,data,features,device):
    dest=OUT/'runs'/cohort/arm/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'metrics.json').exists():
        import json
        result=json.loads((dest/'metrics.json').read_text())
        assert sha(dest/'test.csv')==result['test_predictions_sha256']
        return result
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    labels={s:frame.loc[frame.split==s,'label'].to_numpy(int) for s in data}
    pw=float((labels['train']==0).sum()/labels['train'].sum())
    train_t=frame.loc[frame.split=='train','timestamp'];tmin=int(train_t.min());scale=int(train_t.max()-tmin)
    start=time.monotonic();history=[];best_epoch=0;stopped=0;state={};valap=None
    if arm in ['summary_lr','lightgbm','isolation_forest']:
        if arm=='summary_lr':model=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=seed)
        elif arm=='lightgbm':
            sys.path.insert(0,str(PROJECT/'.deps/317'))
            from lightgbm import LGBMClassifier,early_stopping,log_evaluation
            model=LGBMClassifier(n_estimators=300,learning_rate=.05,num_leaves=31,class_weight='balanced',n_jobs=4,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True,metric='average_precision')
        else:model=IsolationForest(n_estimators=200,contamination='auto',max_samples='auto',random_state=seed,n_jobs=4)
        if arm=='lightgbm':
            model.fit(features['train'],labels['train'],eval_set=[(features['validation'],labels['validation'])],eval_metric='average_precision',callbacks=[early_stopping(30,first_metric_only=True,verbose=False),log_evaluation(0)])
            stopped=int(model.best_iteration_)
        elif arm=='isolation_forest':model.fit(features['train'])
        else:model.fit(features['train'],labels['train']);stopped=int(model.n_iter_.max())
        valid=-model.score_samples(features['validation']) if arm=='isolation_forest' else model.predict_proba(features['validation'])[:,1]
        test=-model.score_samples(features['test']) if arm=='isolation_forest' else model.predict_proba(features['test'])[:,1]
        valap=float(average_precision_score(labels['validation'],valid))
        import pickle
        with (dest/'checkpoint.pt').open('wb') as f:pickle.dump(model,f)
    else:
        tabular=arm=='summary_mlp'
        if tabular:
            model=torch.nn.Sequential(torch.nn.Linear(24,48),torch.nn.ReLU(),torch.nn.Dropout(.3),torch.nn.Linear(48,1))
            datasets={s:torch.tensor(features[s],dtype=torch.float32) for s in data}
        else:
            model=NoTimeGAT() if arm=='gat_no_time' else hm.TGATBaseline() if arm=='gat_time' else gin.CausalLocalGIN(input_dim=3)
            datasets=data
        model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        best=-1;stale=0;best_state=None
        for epoch in range(1,61):
            model.train();loss_sum=0;n=0
            batches=torch.randperm(len(datasets['train'])).split(64) if tabular else DataLoader(datasets['train'],batch_size=64,shuffle=True)
            for b in batches:
                optimizer.zero_grad(set_to_none=True)
                if tabular:
                    y=torch.tensor(labels['train'][b.numpy()],dtype=torch.float32,device=device);z=model(datasets['train'][b].to(device)).squeeze(-1)
                else:
                    b=b.to(device);y=b.y.float()
                    z=model(b,(b.timestamp.float()-float(tmin))/float(scale)) if arm.startswith('gat') else model(b)
                loss=F.binary_cross_entropy_with_logits(z,y,pos_weight=torch.tensor(pw,device=device));loss.backward();optimizer.step()
                loss_sum+=float(loss.detach())*len(y);n+=len(y)
            v=predict(model,datasets['validation'],device,arm,tmin,scale);ap=float(average_precision_score(labels['validation'],v))
            history.append(dict(epoch=epoch,train_loss=loss_sum/n,validation_ap=ap))
            if ap>best+1e-6:
                best=ap;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            print(cohort,arm,seed,epoch,'validation_ap',round(ap,6),flush=True)
            if stale>=5:break
        stopped=epoch;valap=best;model.load_state_dict(best_state)
        valid=predict(model,datasets['validation'],device,arm,tmin,scale)
        test=predict(model,datasets['test'],device,arm,tmin,scale)
        torch.save(dict(model_state_dict=best_state,arm=arm,seed=seed,cohort=cohort,tmin=tmin,time_scale=scale),dest/'checkpoint.pt')
        del model,optimizer;torch.cuda.empty_cache()
    pd.DataFrame(history,columns=['epoch','train_loss','validation_ap']).to_csv(dest/'history.csv',index=False)
    for split,p in [('validation',valid),('test',test)]:
        f=frame.loc[frame.split==split,['sample_id','timestamp','label']].reset_index(drop=True);f['p_mean']=p
        f.to_csv(dest/f'{split}.csv',index=False)
    result=dict(cohort=cohort,arm=arm,seed=seed,best_epoch=best_epoch,stopped_epoch=stopped,best_validation_ap=valap,
        test_access_count=1,time_min=tmin,time_scale=scale,device=str(device),elapsed_seconds=time.monotonic()-start,
        checkpoint_sha256=sha(dest/'checkpoint.pt'),test_predictions_sha256=sha(dest/'test.csv'),
        protocol_sha256=sha(OUT/'experiment_preregistration.json'),score_kind='outlier_score' if arm=='isolation_forest' else 'probability')
    result.update(dict(auc_pr=float(average_precision_score(labels['test'],test)),auc_roc=float(roc_auc_score(labels['test'],test))) if arm=='isolation_forest' else metrics.classification_metrics(labels['test'],test))
    write(dest/'metrics.json',result);print('COMPLETE',cohort,arm,seed,result['auc_pr'],flush=True)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',choices=['original','horizon7'],required=True);parser.add_argument('--arms',required=True);parser.add_argument('--device',default='cuda:1');args=parser.parse_args()
    register();torch.set_num_threads(4);device=torch.device(args.device)
    frame,data,features=load_data(args.cohort)
    for arm in args.arms.split(','):
        for seed in SEEDS:run(args.cohort,arm,seed,frame,data,features,device)


if __name__=='__main__':main()
