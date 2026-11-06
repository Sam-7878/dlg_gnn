"""Check score direction against the estimator APIs, saved arrays and labels."""
from a24_run_studies import load_input, LOCAL, SUP, ROOT, rows, sha
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from pyod.models.ecod import ECOD
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

def check_features():
    expected=rows(SUP/'feature_only_per_seed.csv')
    out=[]
    for name in dict.fromkeys(r['dataset'] for r in expected):
        data,masks,identity=load_input(name)
        x=data.x.numpy()
        labels=data.y.numpy()
        ecod=ECOD(contamination=.1,n_jobs=1).fit(x)
        for r in (r for r in expected if r['dataset']==name):
            seed=int(r['seed'])
            path=LOCAL/'runs'/r['run_id']/'scores.npz'
            assert sha(path)==r['raw_score_sha256']
            assert r['feature_hash']==identity['feature_hash']
            if r['model']=='IForest':
                detector=IsolationForest(n_estimators=100,max_samples=256,max_features=1.,
                                         bootstrap=False,contamination='auto',random_state=seed,n_jobs=4).fit(x)
                native=detector.score_samples(x)
                oriented=-native
            else:
                oriented=ecod.decision_scores_
                assert np.all(oriented>=0)
            with np.load(path) as z:
                assert np.array_equal(labels,z['labels'])
                assert np.array_equal(z['node_ids'],np.arange(len(labels)))
                for key, mask in masks[seed].items():
                    np.testing.assert_array_equal(mask,z[key])
                np.testing.assert_array_equal(oriented,z['scores'])
                test=z['test_mask']
                y,s=labels[test],oriented[test]
                ap,roc=average_precision_score(y,s),roc_auc_score(y,s)
                assert abs(ap-float(r['pr_auc']))<1e-12 and abs(roc-float(r['roc_auc']))<1e-12
                inverse_ap,inverse_roc=average_precision_score(y,-s),roc_auc_score(y,-s)
                assert np.isclose(roc+inverse_roc,1.)
            out.append(dict(dataset=name,model=r['model'],seed=seed,score_sha256=sha(path),
                            same_vector_as_api=True,AP=ap,ROC=roc,AP_negated=inverse_ap,ROC_negated=inverse_roc))
        print(name,'all ten native oriented scores and reported metrics reproduced',flush=True)
        Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':
    check_features()
