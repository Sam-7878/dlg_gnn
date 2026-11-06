"""Same six-cell LightGBM validation grid on both pre-existing cohorts."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

project=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(project/'.deps/317'))
from lightgbm import LGBMClassifier, early_stopping

out=project/'private/322/tree'
out.mkdir(parents=True,exist_ok=True)
rows=[]
for cohort in ('original','horizon7'):
    f=pd.read_csv(project/f'results/317/{cohort}_population.csv')
    x=np.load(project/f'private/317/{cohort}_summary.npy')
    assert x.shape==(len(f),24)
    train,validation,test=(f.split==s for s in ('train','validation','test'))
    x=StandardScaler().fit(x[train]).transform(x)
    chosen=None
    best=-1
    for leaves in (15,31,63):
        for rate in (.03,.1):
            model=LGBMClassifier(n_estimators=300,num_leaves=leaves,learning_rate=rate,class_weight='balanced',
                n_jobs=4,random_state=7,verbosity=-1,deterministic=True,force_col_wise=True,metric='average_precision')
            model.fit(x[train],f.label[train],eval_set=[(x[validation],f.label[validation])],
                      eval_metric='average_precision',callbacks=[early_stopping(30,first_metric_only=True,verbose=False)])
            valid=model.predict_proba(x[validation])[:,1]
            ap=average_precision_score(f.label[validation],valid)
            rows.append(dict(cohort=cohort,leaves=leaves,rate=rate,validation_ap=ap,
                             trees=model.best_iteration_,selected=False))
            if ap>best:
                chosen=model
                best=ap
                selected=len(rows)-1
    rows[selected]['selected']=True
    test_score=chosen.predict_proba(x[test])[:,1]
    rows[selected]['test_ap']=average_precision_score(f.label[test],test_score)
    rows[selected]['test_roc_auc']=roc_auc_score(f.label[test],test_score)
    for split, mask in [('validation',validation),('test',test)]:
        np.savez_compressed(out/f'{cohort}_{split}.npz',label=f.label[mask].to_numpy(),
                            score=chosen.predict_proba(x[mask])[:,1])
pd.DataFrame(rows).to_csv(project/'results/322/lightgbm_grid.csv',index=False)
