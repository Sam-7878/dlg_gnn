"""Preregistered LightGBM grid, train-only chain prior and negative-only IF."""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from common import OUT,PRIVATE,PROJECT,SEEDS,require_builder,sha,write


def export(frame,split,score,arm,seed):
    f=frame.loc[frame.split==split,['sample_id','timestamp','label']].reset_index(drop=True).copy()
    f['p_mean']=score
    dest=OUT/'predictions'/f'horizon7_{arm}_seed{seed}_{split}.csv'
    dest.parent.mkdir(parents=True,exist_ok=True);f.to_csv(dest,index=False)
    return sha(dest)


def main():
    require_builder()
    sys.path.insert(0,str(PROJECT/'.deps/317'))
    from lightgbm import LGBMClassifier,early_stopping,log_evaluation
    frame=pd.read_csv(PROJECT/'results/317/horizon7_population.csv')
    x=np.load(PRIVATE.parent/'317/horizon7_summary.npy')
    assert x.shape==(len(frame),24)
    sets={s:frame.split==s for s in ['train','validation','test']}
    labels={s:frame.loc[m,'label'].to_numpy(int) for s,m in sets.items()}
    scaler=StandardScaler().fit(x[sets['train']]);x={s:scaler.transform(x[m]) for s,m in sets.items()}
    write(OUT/'lightgbm_scaler.json',dict(fit='train',mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist()))
    configs=[(n,r) for n in [15,31,63] for r in [.03,.1]]
    grid=[];selected=None;best=-1
    for n,r in configs:
        model=LGBMClassifier(n_estimators=300,num_leaves=n,learning_rate=r,class_weight='balanced',n_jobs=4,random_state=7,
            verbosity=-1,deterministic=True,force_col_wise=True,metric='average_precision')
        model.fit(x['train'],labels['train'],eval_set=[(x['validation'],labels['validation'])],
                  eval_metric='average_precision',callbacks=[early_stopping(30,first_metric_only=True,verbose=False),log_evaluation(0)])
        val=model.predict_proba(x['validation'])[:,1];ap=float(average_precision_score(labels['validation'],val))
        grid.append(dict(num_leaves=n,learning_rate=r,best_iteration=model.best_iteration_,validation_ap=ap))
        if ap>best:best=ap;selected=(n,r);selected_model=model
    pd.DataFrame(grid).to_csv(OUT/'lightgbm_validation_grid.csv',index=False)
    write(OUT/'lightgbm_selected.json',dict(num_leaves=selected[0],learning_rate=selected[1],selection_seed=7,validation_ap=best,
        test_used_for_selection=False,protocol_sha256=sha(OUT/'protocol_freeze_319.json')))
    for seed in SEEDS:
        if seed==7:model=selected_model
        else:
            model=LGBMClassifier(n_estimators=300,num_leaves=selected[0],learning_rate=selected[1],class_weight='balanced',n_jobs=4,
                random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True,metric='average_precision')
            model.fit(x['train'],labels['train'],eval_set=[(x['validation'],labels['validation'])],
                      eval_metric='average_precision',callbacks=[early_stopping(30,first_metric_only=True,verbose=False),log_evaluation(0)])
        v=model.predict_proba(x['validation'])[:,1]
        export(frame,'validation',v,'lightgbm',seed)
        t=model.predict_proba(x['test'])[:,1]
        export(frame,'test',t,'lightgbm',seed)
        print('LightGBM',seed,model.best_iteration_,float(average_precision_score(labels['test'],t)),flush=True)
    prior=frame.loc[sets['train']].groupby('chain').label.mean().to_dict();pooled=labels['train'].mean()
    write(OUT/'chain_prior_fit.json',dict(source='training labels only',by_chain=prior,pooled=float(pooled)))
    for split in ['validation','test']:
        p=frame.loc[sets[split],'chain'].map(prior).fillna(pooled).to_numpy(float)
        export(frame,split,p,'chain_prior',7)
    print('chain prior',float(average_precision_score(labels['test'],p)),flush=True)
    model=IsolationForest(n_estimators=200,max_samples=256,contamination='auto',random_state=7,n_jobs=4)
    model.fit(x['train'][labels['train']==0])
    for split in ['validation','test']:
        p=-model.score_samples(x[split])
        export(frame,split,p,'negative_only_iforest',7)
    print('negative-only IF',float(average_precision_score(labels['test'],p)),float(roc_auc_score(labels['test'],p)),flush=True)


if __name__=='__main__':main()
