"""Retrospective lifetime/composition checks, distinct from admissible features."""
import argparse
import json
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import OUT,PROJECT,SEEDS,population,require_builder,write,sha
from metrics import conditional_summary,point,ap,auc


def preserved_matrix(cohort,arm):
    panel=('additional' if arm=='gat_time' else 'matched') if cohort=='original' else 'horizon7'
    frames=[pd.read_csv(PROJECT/f'results/317/predictions/{panel}_{arm}_seed{s}_test.csv',float_precision='round_trip') for s in SEEDS]
    ids=frames[0][['sample_id','timestamp','label']]
    assert all(ids.equals(f[['sample_id','timestamp','label']]) for f in frames)
    return ids,np.stack([f.score.to_numpy() for f in frames])


def existing():
    assert (OUT/'protocol_freeze_319.json').exists()
    lineage=pd.read_csv(PROJECT/'results/317/sample_time_lineage.csv').set_index('sample_id')
    records=[];support=[];oracle=[];association=[]
    for cohort in ['original','horizon7']:
        pop=population(cohort);test=pop[pop.split=='test'].reset_index(drop=True)
        val=pop[pop.split=='validation'].reset_index(drop=True)
        life=lineage.loc[test.sample_id].observed_lifetime_seconds.to_numpy()/86400
        masks={'all':np.ones(len(test),bool),'L<1d':life<1,'1<=L<7d':(life>=1)&(life<7),'L>=7d':life>=7}
        if cohort=='horizon7':masks['continued_after_7d']=life>=7
        y=test.label.to_numpy(int)
        for group,mask in masks.items():
            support.append(dict(cohort=cohort,subgroup=group,n=int(mask.sum()),positive=int(y[mask].sum()),
                negative=int((1-y[mask]).sum()),prevalence=float(y[mask].mean()) if mask.any() else np.nan,
                selection='RETROSPECTIVE/FUTURE-CONDITIONED; not operational eligibility' if group!='all' else 'frozen test population'))
        for arm in ['gin_bce','summary_mlp','summary_lr','gat_time']:
            ids,matrix=preserved_matrix(cohort,arm)
            assert ids.equals(test[['sample_id','timestamp','label']])
            for group,mask in masks.items():
                result=conditional_summary(y[mask],matrix[:,mask],seed=319)
                records.append(dict(cohort=cohort,arm=arm,subgroup=group,score_source='preserved R317',
                                    scope='RETROSPECTIVE SUBGROUP DIAGNOSTIC; not eligibility',**result))
        score=-np.log1p(life)
        for chain in ['all','ethereum','bsc','polygon']:
            mask=np.ones(len(test),bool) if chain=='all' else test.chain.to_numpy()==chain
            oracle.append(dict(cohort=cohort,chain=chain,feature='FULL_ARCHIVE_OBSERVED_SPAN',
                scope='ORACLE/RETROSPECTIVE DIAGNOSTIC ONLY',orientation='negative log1p(days), fixed before evaluation',
                operational_feature=False,performance_upper_bound=False,**conditional_summary(y[mask],score[mask])))
        for split,g in pop.groupby('split'):
            L=lineage.loc[g.sample_id].observed_lifetime_seconds.to_numpy()/86400
            for label in [0,1]:
                a=L[g.label.to_numpy()==label]
                association.append(dict(cohort=cohort,split=split,label=label,n=len(a),median_days=float(np.median(a)),
                    fraction_span_under_1d=float(np.mean(a<1)),fraction_span_under_7d=float(np.mean(a<7)),
                    scope='FULL_ARCHIVE OBSERVED SPAN, RETROSPECTIVE ASSOCIATION'))
    pd.DataFrame(records).to_csv(OUT/'lifetime_shortcut_audit.csv',index=False)
    pd.DataFrame(support).to_csv(OUT/'subgroup_support.csv',index=False)
    pd.DataFrame(oracle).to_csv(OUT/'oracle_lifetime_baselines.csv',index=False)
    pd.DataFrame(association).to_csv(OUT/'lifetime_label_association.csv',index=False)
    composition()
    print('Retrospective subgroup/oracle and first-time composition diagnostics complete',flush=True)


def wilson(positive,n):
    z=1.959963984540054;phat=positive/n;den=1+z*z/n
    center=(phat+z*z/(2*n))/den
    delta=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/den
    return center-delta,center+delta


def composition():
    f=pd.read_csv(PROJECT/'results/317/sample_time_lineage.csv')
    dt=pd.to_datetime(f.first_valid_tx_time,unit='s',utc=True).dt.tz_localize(None)
    f['month']=dt.dt.to_period('M').astype(str);f['quarter']=dt.dt.to_period('Q').astype(str)
    rows=[]
    for unit in ['month','quarter']:
        for chain in ['all','ethereum','bsc','polygon']:
            g=f if chain=='all' else f[f.chain==chain]
            for period,h in g.groupby(unit):
                lo,hi=wilson(int(h.label.sum()),len(h));L=h.observed_lifetime_seconds/86400
                distance=(1709251199-h.first_valid_tx_time)/86400
                rows.append(dict(period_unit=unit,period=period,chain=chain,n=len(h),positive=int(h.label.sum()),
                    prevalence=h.label.mean(),wilson95_low=lo,wilson95_high=hi,
                    observed_span_median_days=L.median(),observed_span_q25_days=L.quantile(.25),observed_span_q75_days=L.quantile(.75),
                    days_to_archive_endpoint_median=distance.median(),eligible7_n=int((distance>=7).sum()),eligible30_n=int((distance>=30).sum()),
                    source_category_count=int(h.category.nunique()),low_support=len(h)<20))
    allrows=pd.DataFrame(rows)
    allrows[allrows.period_unit=='month'].to_csv(OUT/'first_time_monthly_composition.csv',index=False)
    allrows[allrows.period_unit=='quarter'].to_csv(OUT/'first_time_quarterly_composition.csv',index=False)
    chains=f.groupby(['month','chain']).label.agg(n='size',positive='sum').reset_index()
    chains.to_csv(OUT/'first_time_chain_mixture.csv',index=False)


def features():
    require_builder()
    counts=pd.read_csv(OUT/'cutoff_admissible_counts.csv');rows=[]
    summary=pd.read_csv(PROJECT/'results/317/model_summary.csv')
    for cohort in ['original','horizon7']:
        f=population(cohort)
        c=counts[counts.cohort==cohort].set_index('sample_id').loc[f.sample_id]
        assert np.array_equal(c.label,f.label)
        X=np.log1p(c[['retained_nodes','retained_edges','full_history_nodes','full_history_edges']].to_numpy(float))
        chain=np.eye(3)[f.chain_index]
        configs={name:X[:,[i]] for i,name in enumerate(['retained_nodes','retained_edges','full_history_nodes','full_history_edges'])}
        configs['count_only']=X;configs['count_plus_chain']=np.c_[X,chain]
        # Existing cached 24 summaries are read-only and bound to identical graph IDs.
        old_features=np.load(PROJECT/f'private/317/{cohort}_summary.npy')
        configs['degree_summary_only']=old_features[:,3:21]
        for name,x in configs.items():
            train=f.split.to_numpy()=='train';valid=f.split.to_numpy()=='validation';test=f.split.to_numpy()=='test'
            scaler=StandardScaler().fit(x[train]);model=LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=7)
            model.fit(scaler.transform(x[train]),f.label.to_numpy()[train])
            vp=model.predict_proba(scaler.transform(x[valid]))[:,1];tp=model.predict_proba(scaler.transform(x[test]))[:,1]
            r=point(f.label.to_numpy()[test],tp)
            rows.append(dict(cohort=cohort,arm=name,input_dimensions=x.shape[1],fit='training-only balanced LR',
                deterministic_fits=1,validation_ap=ap(f.label.to_numpy()[valid],vp),score_orientation='learned from training labels; no test sign choice',**r))
            for split,mask,p in [('validation',valid,vp),('test',test,tp)]:
                export=f.loc[mask,['sample_id','timestamp','label']].copy();export['score']=p
                dest=OUT/f'predictions/{cohort}_single_{name}_{split}.csv';dest.parent.mkdir(exist_ok=True);export.to_csv(dest,index=False)
            write(OUT/f'feature_fits/{cohort}_{name}.json',dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),test_sign_selection=False,
                protocol_sha256=sha(OUT/'protocol_freeze_319.json')))
    pd.DataFrame(rows).to_csv(OUT/'single_feature_baselines.csv',index=False)
    print('Cutoff-admissible single/count/degree baselines complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['existing','features'],required=True)
    args=parser.parse_args()
    existing() if args.phase=='existing' else features()
