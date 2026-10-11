"""Fresh R01 models and matched predictions; all relations built from L1 outputs."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from r01_common import ROOT,config,output,write,read,sha256,digest,source_identity
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.level2.relation_builder import HistoricalReferenceIndex
from gog_fraud.streaming.selective import (LocalEncoder,RelationalEncoder,DenseClassifier,seed_all,
    local_predict,fit_calibration,calibrated,fit_threshold,fuse,binary_metrics,router_score,risk_counts)
EXECUTED_SOURCES=source_identity()


def degree_features(graph):
    x=graph.x.numpy();return np.r_[x.mean(0),x.std(0),x.max(0),np.log1p(graph.num_nodes),np.log1p(graph.num_edges)].astype(np.float32)


def load_cache(cfg):
    cache=torch.load(ROOT/cfg['graph_cache'],map_location='cpu',weights_only=False)
    for split,graphs in cache['graphs'].items():
        for graph,meta in zip(graphs,cache['metadata'][split]):
            if int(graph.y.item())!=meta['label']:raise ValueError('cached graph/metadata label mismatch')
            # Recompute bounded degree features; do not consume historical hybrid features.
            edges=graph.edge_index.numpy();n=graph.num_nodes
            incoming=np.bincount(edges[1],minlength=n).astype(np.float32)
            outgoing=np.bincount(edges[0],minlength=n).astype(np.float32)
            graph.x=torch.from_numpy(np.stack([np.log1p(incoming),np.log1p(outgoing),np.log1p(incoming+outgoing)],axis=1))
            graph.y=graph.y.float().reshape(1);graph.num_nodes=n
    return cache


def subset(cache,chains,split):
    ix=[i for i,m in enumerate(cache['metadata'][split]) if m['chain'] in chains]
    return [cache['graphs'][split][i] for i in ix],[cache['metadata'][split][i] for i in ix]


def fit_graph_model(model,graphs,epochs,batch_size,lr,device,relational=False):
    labels=np.array([int(g.y.item()) for g in graphs]);positive=int(labels.sum())
    weight=torch.tensor((len(labels)-positive)/max(1,positive),device=device,dtype=torch.float32)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    criterion=torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    losses=[];start=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0.;count=0
        for batch in DataLoader(graphs,batch_size=batch_size,shuffle=True,num_workers=0):
            batch=batch.to(device);optimizer.zero_grad(set_to_none=True)
            result=model(batch);logits=result if relational else result[0]
            loss=criterion(logits,batch.y.reshape(-1));loss.backward();optimizer.step()
            total+=float(loss.detach())*len(logits);count+=len(logits)
        losses.append(total/count)
        if epoch in (0,epochs-1):print(f'{type(model).__name__} epoch {epoch+1}/{epochs} loss={losses[-1]:.5f}',flush=True)
    model.eval();return {'epochs':epochs,'losses':losses,'training_seconds':time.perf_counter()-start,'training_pos_weight':float(weight.cpu()),'parameter_count':sum(p.numel() for p in model.parameters())}


@torch.inference_mode()
def infer_local(model,graphs,T,device,batch_size=128):
    values=[];emb=[];var=[];start=time.perf_counter()
    for batch in DataLoader(graphs,batch_size=batch_size,shuffle=False):
        p,e,v=local_predict(model,batch.to(device),T);values.append(p);emb.append(e);var.append(v)
    return np.concatenate(values),np.concatenate(emb),np.concatenate(var),(time.perf_counter()-start)*1000/len(graphs)


def relation_graphs(index,embedding,scores,meta,labels=None):
    graphs=[];audits=[]
    for i,(e,p,m) in enumerate(zip(embedding,scores,meta)):
        graph,audit=index.target_graph(e,p,m['event_end'],m['sample_id'])
        if labels is not None:graph.y=torch.tensor([float(labels[i])])
        graphs.append(graph);audits.append(audit)
    return graphs,audits


@torch.inference_mode()
def infer_relation(model,graphs,device,batch_size=256):
    result=[];start=time.perf_counter()
    for batch in DataLoader(graphs,batch_size=batch_size,shuffle=False):result.append(torch.sigmoid(model(batch.to(device))).cpu().numpy())
    return np.concatenate(result),(time.perf_counter()-start)*1000/len(graphs)


def fit_dense(x,y,cfg,device,epochs=None):
    model=DenseClassifier(x.shape[1]).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    xx=torch.from_numpy(np.asarray(x,np.float32)).to(device);yy=torch.from_numpy(np.asarray(y,np.float32)).to(device)
    weight=torch.tensor((len(y)-sum(y))/max(1,sum(y)),device=device);criterion=torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    for _ in range(epochs or cfg['local']['epochs']):
        model.train()
        for ix in torch.randperm(len(y),device=device).split(256):
            opt.zero_grad(set_to_none=True);loss=criterion(model(xx[ix]),yy[ix]);loss.backward();opt.step()
    return model.eval()


@torch.inference_mode()
def dense_predict(model,x,device):return torch.sigmoid(model(torch.from_numpy(np.asarray(x,np.float32)).to(device))).cpu().numpy()


def benefit_features(fast,variance,emb):
    p=np.clip(fast,1e-8,1-1e-8)
    return np.column_stack([p,-p*np.log(p)-(1-p)*np.log1p(-p),variance,np.linalg.norm(emb,axis=1)]).astype(float)


def fit_policies(fast,deep,y,features,cfg,seed,eligible=None):
    fast_map=fit_calibration(fast,y);deep_map=fit_calibration(deep,y)
    pf=calibrated(fast,fast_map);pd=calibrated(deep,deep_map);tau,_=fit_threshold(y,pf)
    best=None
    for weight in cfg['policy']['fusion_weights']:
        fused=fuse(pf,pd,weight)
        if eligible is not None:fused=np.where(eligible,fused,pf)
        threshold,score=fit_threshold(y,fused)
        rank=(score,-abs(weight-.5),weight)
        if best is None or rank>best[0]:best=(rank,weight,threshold)
    weight,full_tau=best[1:];fused=fuse(pf,pd,weight)
    if eligible is not None:fused=np.where(eligible,fused,pf)
    local_loss=-(y*np.log(np.clip(pf,1e-8,1))+(1-y)*np.log(np.clip(1-pf,1e-8,1)))
    full_loss=-(y*np.log(np.clip(fused,1e-8,1))+(1-y)*np.log(np.clip(1-fused,1e-8,1)))
    gain=local_loss-full_loss;oof=np.empty(len(y))
    for train,val in KFold(5,shuffle=True,random_state=seed).split(features):
        model=Ridge(alpha=1).fit(features[train],gain[train]);oof[val]=model.predict(features[val])
    benefit=Ridge(alpha=1).fit(features,gain)
    policies=[]
    for family in cfg['policy']['routers']:
        repeats=cfg['policy']['random_repeats'] if family=='random' else [0]
        for repeat in repeats:
            if family=='random':rankscore=np.random.default_rng(seed*1000+repeat).random(len(y))
            elif family=='learned_benefit':rankscore=oof
            else:rankscore=router_score(pf,tau,family)
            for budget in cfg['policy']['budgets']:
                cut=float(np.quantile(rankscore,1-budget,method='higher')) if 0<budget<1 else None
                route=(rankscore>=cut) if cut is not None else np.full(len(y),budget==1,bool)
                if eligible is not None:route=route & eligible
                final=np.where(route,fused,pf);final_tau,val_f1=fit_threshold(y,final)
                policy={'family':family,'budget':budget,'random_repeat':repeat,'route_cutoff':cut,
                    'fast_threshold':tau,'final_threshold':final_tau,'full_threshold':full_tau,
                    'fast_map':fast_map,'deep_map':deep_map,'weight':weight,
                    'validation_deep_fraction':float(route.mean()),'validation_f1':val_f1,
                    'fit_partition':'source_validation','threshold_control':'same final threshold on direct and escalated decisions',
                    'benefit_coefficients':benefit.coef_.tolist(),'benefit_intercept':float(benefit.intercept_),
                    'benefit_fit':'validation crossfit for selection; refit entire source validation for test',
                    'random_seed':seed*1000+repeat}
                policy['policy_hash']=digest(policy);policies.append(policy)
    controls={'local':{'threshold':tau},'full':{'threshold':full_tau},'threshold_only':{'threshold':next(p['final_threshold'] for p in policies if p['family']=='margin' and p['budget']==cfg['policy']['primary_deep_budget'])}}
    return {'policies':policies,'controls':controls,'fast_map':fast_map,'deep_map':deep_map,'weight':weight,'fit_partition':'source_validation','target_labels_used':False}


def apply_fitted(raw_fast,raw_deep,features,fit,policy=None,control=None,eligible=None):
    fast=calibrated(raw_fast,fit['fast_map']);deep=calibrated(raw_deep,fit['deep_map']);full=fuse(fast,deep,fit['weight'])
    if eligible is not None:full=np.where(eligible,full,fast)
    if control:
        route=np.full(len(fast),control=='full',bool);score=full if control=='full' else fast
        threshold=fit['controls'][control]['threshold']
    else:
        family=policy['family'];budget=policy['budget']
        if family=='random':rank=np.random.default_rng(policy['random_seed']+10**6).random(len(fast))
        elif family=='learned_benefit':rank=features@np.asarray(policy['benefit_coefficients'])+policy['benefit_intercept']
        else:rank=router_score(fast,policy['fast_threshold'],family)
        route=(rank>=policy['route_cutoff']) if policy['route_cutoff'] is not None else np.full(len(fast),budget==1,bool)
        score=np.where(route,full,fast);threshold=policy['final_threshold']
    if eligible is not None:route=route & eligible
    return score,score>=threshold,route


def train_one(cfg,cache,experiment,seed,backbone):
    out=output();name=f'{experiment}_{backbone}_seed{seed}';dest=out/'models'/name
    if (dest/'complete.json').exists():
        record=read(dest/'complete.json')
        if record['config_sha256']!=cfg['config_sha256']:raise ValueError('resume config identity mismatch')
        for filename,expected in record['artifacts'].items():
            if sha256(dest/filename)!=expected:raise ValueError('resume artifact hash mismatch: '+filename)
        return record
    dest.mkdir(parents=True,exist_ok=True);device=torch.device(cfg['device']);seed_all(seed)
    target=experiment.removeprefix('loco_') if experiment.startswith('loco_') else None
    train_chains=[c for c in cfg['chains'] if c!=target];test_chains=[target] if target else cfg['chains']
    parts={s:subset(cache,train_chains if s!='test' else test_chains,s) for s in ('train','validation','test')}
    y={s:np.array([m['label'] for m in parts[s][1]]) for s in parts}
    model=LocalEncoder(backbone).to(device)
    local_training=fit_graph_model(model,parts['train'][0],cfg['local']['epochs'],cfg['local']['batch_size'],cfg['local']['learning_rate'],device)
    torch.save({'state_dict':model.state_dict(),'backbone':backbone,'config':cfg['local'],'training':local_training},dest/'local.pt')
    values={s:infer_local(model,parts[s][0],1,device) for s in parts}
    reference=HistoricalReferenceIndex(values['train'][1],values['train'][0],[m['sample_id'] for m in parts['train'][1]],
        [m['event_end'] for m in parts['train'][1]],k=cfg['relation']['k'],metric=cfg['relation']['metric'])
    np.savez_compressed(dest/'reference.npz',embedding=values['train'][1],score=values['train'][0],
        ids=reference.ids,cutoffs=reference.cutoffs)
    stars={};audit={}
    for s in parts:stars[s],audit[s]=relation_graphs(reference,values[s][1],values[s][0],parts[s][1],y[s] if s=='train' else None)
    deep=RelationalEncoder().to(device)
    deep_training=fit_graph_model(deep,stars['train'],cfg['relational']['epochs'],cfg['relational']['batch_size'],cfg['relational']['learning_rate'],device,True)
    torch.save({'state_dict':deep.state_dict(),'config':cfg['relational'],'training':deep_training},dest/'relational.pt')
    ds={s:infer_relation(deep,stars[s],device)[0] for s in parts}
    # No-reference fallback is the same local raw probability in all inference lanes.
    for s in parts:
        empty=np.array([a['reference_count']==0 for a in audit[s]])
        ds[s][empty]=values[s][0][empty]
    fv=benefit_features(values['validation'][0],values['validation'][2],values['validation'][1])
    fit=fit_policies(values['validation'][0],ds['validation'],y['validation'],fv,cfg,seed,
        np.array([a['reference_count']>0 for a in audit['validation']]))
    fit['experiment_id']=experiment;fit['run_id']=cfg['run_id'];fit['train_chains']=train_chains;fit['test_chains']=test_chains
    fit['model_hash']=digest({'local':sha256(dest/'local.pt'),'relational':sha256(dest/'relational.pt')})
    fit['reference_manifest_hash']=sha256(dest/'reference.npz');write(dest/'policy_fit.json',fit)
    metrics_rows=[]
    for s in ('validation','test'):
        fast,emb,var,_=values[s];features=benefit_features(fast,var,emb)
        frame=pd.DataFrame(parts[s][1]).rename(columns={'chain':'chain_scope','event_end':'snapshot_cutoff'})
        frame['run_id']=cfg['run_id'];frame['experiment_id']=experiment;frame['seed']=seed;frame['backbone']=backbone
        frame['label_source']='retained_snapshot_metadata';frame['label_available_time']=None
        frame['model_hash']=fit['model_hash'];frame['reference_manifest_hash']=fit['reference_manifest_hash'];frame['raw_local_probability']=fast
        frame['counterfactual_full_raw_probability']=ds[s];frame['local_probability']=calibrated(fast,fit['fast_map']);frame['mc_variance']=var
        frame['reference_count']=[a['reference_count'] for a in audit[s]];frame['eligible_reference_count']=[a['eligible_reference_count'] for a in audit[s]]
        frame['selected_reference_ids_hash']=[a['selected_reference_ids_hash'] for a in audit[s]]
        frame['selected_reference_max_cutoff']=[a['selected_reference_max_cutoff'] for a in audit[s]]
        for control in ('local','full','threshold_only'):
            score,pred,route=apply_fitted(fast,ds[s],features,fit,control=control,eligible=np.array([a['reference_count']>0 for a in audit[s]]))
            frame[control+'_probability']=score;frame[control+'_prediction']=pred.astype(int)
            for chain in ['pooled']+test_chains if s=='test' else ['pooled']+train_chains:
                mask=np.ones(len(frame),bool) if chain=='pooled' else frame.chain_scope.eq(chain).to_numpy()
                if mask.any():metrics_rows.append({'experiment_id':experiment,'run_id':cfg['run_id'],'seed':seed,'backbone':backbone,'split':s,'chain_scope':chain,'method':control,'budget':float(control=='full'),'deep_fraction':float(route[mask].mean()),**binary_metrics(y[s][mask],score[mask],pred[mask])})
        for policy in fit['policies']:
            score,pred,route=apply_fitted(fast,ds[s],features,fit,policy=policy,eligible=np.array([a['reference_count']>0 for a in audit[s]]))
            method=policy['family'];budget=policy['budget'];repeat=policy['random_repeat']
            if method=='margin' and budget==cfg['policy']['primary_deep_budget']:
                frame['selective_probability']=score;frame['selective_prediction']=pred.astype(int);frame['escalated']=route
                frame['policy_hash']=policy['policy_hash'];frame['final_threshold']=policy['final_threshold'];frame['route_threshold']=policy['route_cutoff']
            for chain in ['pooled']+test_chains if s=='test' else ['pooled']+train_chains:
                mask=np.ones(len(frame),bool) if chain=='pooled' else frame.chain_scope.eq(chain).to_numpy()
                if mask.any():metrics_rows.append({'experiment_id':experiment,'run_id':cfg['run_id'],'seed':seed,'backbone':backbone,'split':s,'chain_scope':chain,'method':method,'budget':budget,'random_repeat':repeat,'policy_hash':policy['policy_hash'],'deep_fraction':float(route[mask].mean()),'budget_error':float(route[mask].mean()-budget),**binary_metrics(y[s][mask],score[mask],pred[mask]),**risk_counts(y[s][mask],pred[mask],route[mask])})
        frame.drop(columns=['sorted_path'],errors='ignore').to_parquet(dest/f'{s}_predictions.parquet',index=False)
        np.savez_compressed(dest/f'{s}_features.npz',benefit=features,embedding=emb)
    # Same bounded degree aggregates and same train-only scaling for light controls.
    if experiment=='pooled_snapshot' and backbone=='GIN':
        raw_x={s:np.stack([degree_features(g) for g in parts[s][0]]) for s in parts}
        scaler=StandardScaler().fit(raw_x['train']);xx={s:scaler.transform(raw_x[s]).astype(np.float32) for s in parts}
        lr=LogisticRegression(C=1,class_weight='balanced',max_iter=1000).fit(xx['train'],y['train'])
        mlp=fit_dense(xx['train'],y['train'],cfg,device)
        relational_x={s:np.stack([np.r_[g.x[-1].numpy(),g.x[:-1].mean(0).numpy() if len(g.x)>1 else np.zeros(65)] for g in stars[s]]) for s in parts}
        rm=fit_dense(relational_x['train'],y['train'],cfg,device,epochs=cfg['relational']['epochs'])
        for name,probs in [('degree_logistic',{s:lr.predict_proba(xx[s])[:,1] for s in parts}),('degree_MLP',{s:dense_predict(mlp,xx[s],device) for s in parts}),
            ('relational_MLP',{s:dense_predict(rm,relational_x[s],device) for s in parts}),
            ('knn_mean',{s:np.array([float(g.x[:-1,-1].mean()) if len(g.x)>1 else float(values[s][0][i]) for i,g in enumerate(stars[s])]) for s in parts})]:
            mapping=fit_calibration(probs['validation'],y['validation']);threshold,_=fit_threshold(y['validation'],calibrated(probs['validation'],mapping))
            p=calibrated(probs['test'],mapping);pred=p>=threshold
            pd.DataFrame({'sample_id':[m['sample_id'] for m in parts['test'][1]],'label':y['test'],'raw_probability':probs['test'],'probability':p,'prediction':pred}).to_parquet(dest/f'{name}_predictions.parquet',index=False)
            metrics_rows.append({'experiment_id':experiment,'run_id':cfg['run_id'],'seed':seed,'backbone':backbone,'split':'test','chain_scope':'pooled','method':name,**binary_metrics(y['test'],p,pred)})
            write(dest/f'{name}_fit.json',{'calibration':mapping,'threshold':threshold,'fit_partition':'validation','train_chains':train_chains,'degree_feature_scaler_mean':scaler.mean_.tolist(),'degree_feature_scaler_scale':scaler.scale_.tolist()})
        torch.save({'degree_mlp':mlp.state_dict(),'relational_mlp':rm.state_dict(),'degree_feature_scaler_mean':scaler.mean_,'degree_feature_scaler_scale':scaler.scale_,'logistic_coef':lr.coef_,'logistic_intercept':lr.intercept_},dest/'controls.pt')
        mc_rows=[]
        for T in cfg['mc']['passes']:
            for s in ('validation','test'):
                seed_all(seed*100+T);p,e,v,latency=infer_local(model,parts[s][0],T,device)
                if s=='validation':mapping=fit_calibration(p,y[s]);threshold,_=fit_threshold(y[s],calibrated(p,mapping));write(dest/f'mc_T{T}_fit.json',{'mapping':mapping,'threshold':threshold,'fit_partition':'validation','dropout_modules':'Dropout only; BatchNorm frozen'})
                score=calibrated(p,mapping);pred=score>=threshold
                uncertainty=v if T>1 else router_score(score,threshold,'entropy')
                errors=(pred!=y[s]).astype(int)
                from sklearn.metrics import roc_auc_score
                error_auc=float(roc_auc_score(errors,uncertainty)) if len(np.unique(errors))==2 else None
                mc_rows.append({'seed':seed,'split':s,'T':T,'error_score':'MC population variance' if T>1 else 'deterministic predictive entropy',
                    'error_auroc':error_auc,'latency_ms_per_contract_batched':latency,'batch_size':128,**binary_metrics(y[s],score,pred)})
                pd.DataFrame({'sample_id':[m['sample_id'] for m in parts[s][1]],'label':y[s],'raw_probability':p,'probability':score,'prediction':pred,'variance':v,'error_score':uncertainty}).to_parquet(dest/f'mc_T{T}_{s}.parquet',index=False)
        pd.DataFrame(mc_rows).to_csv(dest/'mc_metrics.csv',index=False)
    pd.DataFrame(metrics_rows).to_csv(dest/'metrics.csv',index=False)
    record={'experiment_id':experiment,'run_id':cfg['run_id'],'seed':seed,'backbone':backbone,'train_chains':train_chains,
        'test_chains':test_chains,'config_sha256':cfg['config_sha256'],'local_training':local_training,'relational_training':deep_training,
        'model_fit_data_max_time':max(m['event_end'] for m in parts['train'][1]),
        'calibration_data_max_time':max(m['event_end'] for m in parts['validation'][1]),
        'label_available_time':'unknown','strict_online_eligible':False,
        'source_hashes_at_process_start':EXECUTED_SOURCES,'artifacts':{p.name:sha256(p) for p in dest.iterdir() if p.is_file() and p.name!='complete.json'}}
    write(dest/'complete.json',record);print(f'{name} COMPLETE',flush=True);return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',choices=['pooled','loco','all'],default='all');args=parser.parse_args()
    cfg=config();cache=load_cache(cfg)
    if args.campaign in ('pooled','all'):
        for backbone in ('GIN','GATv2'):
            for seed in cfg['seeds']:train_one(cfg,cache,'pooled_snapshot',seed,backbone)
    if args.campaign in ('loco','all'):
        for target in cfg['chains']:
            for seed in cfg['seeds']:train_one(cfg,cache,'loco_'+target,seed,'GIN')


if __name__=='__main__':main()
