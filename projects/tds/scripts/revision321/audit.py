"""Replay frozen scores and fit the retrospective node-count control."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

parser=argparse.ArgumentParser()
parser.add_argument('project',type=Path)
parser.add_argument('out',type=Path)
args=parser.parse_args()
args.out.mkdir(parents=True,exist_ok=True)
old=args.project/'results/317'
prior=args.project/'results/319'
span=pd.read_csv(old/'sample_time_lineage.csv').set_index('sample_id').observed_lifetime_seconds
counts=pd.read_csv(prior/'cutoff_admissible_counts.csv')
records, draws=[], []
rng=np.random.default_rng(321)
for cohort in ('original','horizon7'):
    pop=pd.read_csv(old/f'{cohort}_population.csv')
    df=pop.merge(counts[counts.cohort==cohort],on=['sample_id','label','chain'],validate='one_to_one')
    assert len(df)==len(pop) and (df.timestamp==df.cutoff).all() and (df.max_input_timestamp<=df.cutoff).all()
    df['span']=df.sample_id.map(span)
    assert df.span.notna().all()
    test=df[df.split=='test']
    train=df[(df.split=='train')&(df.span>=604800)]
    y=test.label.to_numpy()
    life=test.span.to_numpy()
    models={}
    def scores(path):
        frame=pd.read_csv(path)
        assert np.array_equal(frame[['sample_id','timestamp','label']],test[['sample_id','timestamp','label']])
        return frame['score' if 'score' in frame else 'p_mean'].to_numpy()
    for arm in ('retained_nodes','retained_edges','count_only'):
        models[arm]=[scores(prior/f'predictions/{cohort}_single_{arm}_test.csv')]
    panel='matched' if cohort=='original' else 'horizon7'
    seeds=(7,17,27,37,47)
    for arm in ('gin_bce','summary_mlp'):
        models[arm]=[scores(old/f'predictions/{panel}_{arm}_seed{seed}_test.csv') for seed in seeds]
    if cohort=='horizon7':
        for arm in ('gin0','gin_size','gin_sum','mlp_full'):
            models[arm]=[scores(prior/f'runs/horizon7/{arm}/seed{seed}/test.csv') for seed in seeds]
        fit=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000))
        fit.fit(np.log1p(train[['retained_nodes']]),train.label)
        models['nodes_refit']=[fit.predict_proba(np.log1p(test[['retained_nodes']]))[:,1]]
        pd.DataFrame([dict(train_n=len(train),positive=int(train.label.sum()),coefficient=fit[-1].coef_[0,0])]).to_csv(args.out/'conditional_fit.csv',index=False)
    for subgroup,mask in [('all',np.ones(len(y),bool)),('<1',life<86400),('1-7',(life>=86400)&(life<604800)),('>=7',life>=604800)]:
        prevalence=y[mask].mean()
        for arm,values in models.items():
            if arm=='nodes_refit' and subgroup!='>=7':
                continue
            average=np.mean([ap(y[mask],v[mask]) for v in values])
            records.append(dict(cohort=cohort,subgroup=subgroup,arm=arm,n=int(mask.sum()),positive=int(y[mask].sum()),
                prevalence=prevalence,ap=average,roc_auc=np.mean([auc(y[mask],v[mask]) for v in values]),
                lift=average-prevalence,ratio=average/prevalence))
    pos,neg=np.flatnonzero(y==1),np.flatnonzero(y==0)
    for target in (.0293,.10,.30):
        npos=min(len(pos),int(len(neg)*target/(1-target)))
        nneg=min(len(neg),round(npos*(1-target)/target))
        for draw in range(200):
            ix=np.r_[rng.choice(pos,npos,replace=False),rng.choice(neg,nneg,replace=False)]
            for arm in ('retained_nodes','gin_bce','summary_mlp')+(('gin_size',) if cohort=='horizon7' else ()):
                values=models[arm]
                draws.append(dict(cohort=cohort,arm=arm,target=target,draw=draw,n=len(ix),positive=npos,
                    ap=np.mean([ap(y[ix],v[ix]) for v in values]),roc_auc=np.mean([auc(y[ix],v[ix]) for v in values])))
pd.DataFrame(records).to_csv(args.out/'count_and_subgroup_metrics.csv',index=False)
draws=pd.DataFrame(draws)
draws.to_csv(args.out/'prevalence_draws.csv',index=False)
draws.groupby(['cohort','arm','target','n','positive'],as_index=False).agg(
    ap=('ap','mean'),ap_sd=('ap','std'),roc_auc=('roc_auc','mean')
).to_csv(args.out/'prevalence_sensitivity.csv',index=False)
