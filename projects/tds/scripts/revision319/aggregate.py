"""Saved-score seven-day controls and new standalone exploratory family."""
import json
import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score
from common import OUT,PROJECT,SEEDS,sha,write,require_builder
from metrics import point,operating,ap

MODELS=['gin0','gin_size','gin_sum','mlp_full','mlp_no_size','mlp_topology_only','summary_lr','gat_time','lightgbm','chain_prior','negative_only_iforest']
ORIGINAL={'summary_lr':'summary_lr','gat_time':'gat_time'}


def load(arm,seed,split):
    if arm in ORIGINAL:p=PROJECT/'results/317/predictions'/f'horizon7_{ORIGINAL[arm]}_seed{seed}_{split}.csv'
    elif arm in ('lightgbm','chain_prior','negative_only_iforest'):p=OUT/'predictions'/f'horizon7_{arm}_seed{seed}_{split}.csv'
    else:p=OUT/'runs/horizon7'/arm/f'seed{seed}/{split}.csv'
    return pd.read_csv(p).rename(columns={'score':'p_mean'}),sha(p)


def temp_fit(y,p):
    z=logit(np.clip(p,1e-12,1-1e-12))
    def nll(t):
        a=z/np.exp(t)
        return float(np.mean(np.logaddexp(0,a)-y*a))
    fit=minimize_scalar(nll,bounds=(-4,4),method='bounded',options={'xatol':1e-9})
    return float(np.exp(fit.x))


def main():
    require_builder();frame=pd.read_csv(PROJECT/'results/317/horizon7_population.csv')
    expected={s:frame.loc[frame.split==s,['sample_id','timestamp','label']].reset_index(drop=True) for s in ['validation','test']}
    chain=frame.loc[frame.split=='test','chain'].reset_index(drop=True)
    records=[];cal=[];ops=[];bychain=[];hist=[];scores={};shas={}
    for arm in MODELS:
        seeds=SEEDS if arm not in ('chain_prior','negative_only_iforest') else (7,)
        for seed in seeds:
            v,vs=load(arm,seed,'validation');t,ts=load(arm,seed,'test')
            for name,f in [('validation',v),('test',t)]:
                assert f[['sample_id','timestamp','label']].equals(expected[name]),(arm,seed,name)
            y=t.label.to_numpy(int);score=t.p_mean.to_numpy(float);vy=v.label.to_numpy(int);vp=v.p_mean.to_numpy(float)
            shas[f'{arm}_{seed}_validation']=vs;shas[f'{arm}_{seed}_test']=ts
            probabilistic=arm!='negative_only_iforest'
            raw=point(y,score,probability=probabilistic)
            records.append(dict(arm=arm,seed=seed,source='new R319' if arm not in ORIGINAL else 'preserved R317',n=len(y),positive=int(y.sum()),
                prevalence=float(y.mean()),score_kind='probability' if probabilistic else 'outlier score',**raw))
            scores[(arm,seed)]=score
            if probabilistic:
                temperature=temp_fit(vy,vp)
                calibrated=expit(logit(np.clip(score,1e-12,1-1e-12))/temperature)
                for state,s in [('raw',score),('temperature_scaled',calibrated)]:
                    cal.append(dict(arm=arm,seed=seed,state=state,temperature=1. if state=='raw' else temperature,
                        validation_fit='NLL, validation only',**point(y,s)))
            for row in operating(y,score,[math.ceil(.01*len(y)),math.ceil(.05*len(y)),50,100]):
                ops.append(dict(arm=arm,seed=seed,n=len(y),positive=int(y.sum()),tie_policy='stable sample order; bounds provided',**row))
            for c in ['ethereum','bsc','polygon']:
                ids=(chain==c).to_numpy();yc=y[ids];sc=score[ids]
                d=dict(arm=arm,seed=seed,chain=c,n=int(ids.sum()),positive=int(yc.sum()),negative=int(len(yc)-yc.sum()),prevalence=float(yc.mean()))
                d['ap']=float(average_precision_score(yc,sc)) if yc.sum() and yc.sum()!=len(yc) else np.nan
                d['roc_auc']=point(yc,sc,False)['roc_auc'] if yc.sum() and yc.sum()!=len(yc) else np.nan
                d['support_note']='point only; low support' if min(d['positive'],d['negative'])<20 else 'conditional; fixed checkpoints'
                bychain.append(d)
            path=OUT/'runs/horizon7'/arm/f'seed{seed}/history.csv'
            if path.exists():
                h=pd.read_csv(path);h.insert(0,'arm',arm);h.insert(1,'seed',seed);hist.append(h)
    pd.DataFrame(records).to_csv(OUT/'seven_day_matched_models.csv',index=False)
    pd.DataFrame(cal).to_csv(OUT/'seven_day_modelwise_calibration.csv',index=False)
    pd.DataFrame(ops).to_csv(OUT/'seven_day_operating_metrics.csv',index=False)
    pd.DataFrame(bychain).to_csv(OUT/'seven_day_chain_support.csv',index=False)
    pd.concat(hist,ignore_index=True).to_csv(OUT/'seven_day_training_histories.csv',index=False)
    write(OUT/'score_provenance_319.json',dict(scores=shas,all_ids_match=True,validation_only_temperature=True))
    # Four comparisons were frozen before outcomes. Contract resampling and UTC-day blocks.
    pairs=[('gin_size','gin0'),('gin_sum','gin0'),('mlp_full','mlp_no_size'),('mlp_full','mlp_topology_only')]
    rng=np.random.default_rng(319);y=expected['test'].label.to_numpy(int);n=len(y)
    days=(expected['test'].timestamp//86400).to_numpy();blocks=[np.where(days==u)[0] for u in np.unique(days)]
    stats=[]
    for a,b in pairs:
        aa=np.stack([scores[(a,s)] for s in SEEDS]);bb=np.stack([scores[(b,s)] for s in SEEDS])
        def delta(ids,A=aa,B=bb):return float(np.mean([ap(y[ids],u[ids])-ap(y[ids],v[ids]) for u,v in zip(A,B)]))
        observed=delta(np.arange(n));ordinary=[];blocked=[];extreme=0;valid_blocks=0
        for _ in range(2000):
            ids=rng.integers(0,n,n)
            if 0<y[ids].sum()<len(ids):ordinary.append(delta(ids))
            picks=rng.integers(0,len(blocks),len(blocks));ids=np.concatenate([blocks[i] for i in picks])
            if 0<y[ids].sum()<len(ids):blocked.append(delta(ids));valid_blocks+=1
        for _ in range(4000):
            swap=rng.integers(0,2,n,dtype=np.int8).astype(bool);d=[]
            for u,v in zip(aa,bb):
                d.append(ap(y,np.where(swap,v,u))-ap(y,np.where(swap,u,v)))
            extreme+=abs(np.mean(d))>=abs(observed)-1e-15
        stats.append(dict(a=a,b=b,delta_ap=observed,bootstrap_low=np.quantile(ordinary,.025),bootstrap_high=np.quantile(ordinary,.975),
            utc_day_block_low=np.quantile(blocked,.025),utc_day_block_high=np.quantile(blocked,.975),valid_blocks=valid_blocks,
            score_swap_p=(extreme+1)/4001,unit='same test contracts; five fixed seed checkpoints',family='R319 separate four-comparison exploratory'))
    order=sorted(range(4),key=lambda i:stats[i]['score_swap_p']);running=0
    for rank,i in enumerate(order):
        running=max(running,(4-rank)*stats[i]['score_swap_p']);stats[i]['holm_p']=min(1,running)
    pd.DataFrame(stats).to_csv(OUT/'new_statistics_319.csv',index=False)
    print('R319 aggregate complete',flush=True)


if __name__=='__main__':main()
