"""Standalone CPU replay: only NumPy, pandas, SciPy and scikit-learn required."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,log_loss,brier_score_loss


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tied_ap(y,p):
    if not np.sum(y):return float('nan')
    order=np.argsort(-p,kind='stable');y=np.asarray(y)[order];p=np.asarray(p)[order]
    ends=np.r_[np.flatnonzero(np.diff(p)),len(p)-1];tp=np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/tp[-1])


def ece(y,p,adaptive=False):
    if adaptive:
        groups=np.array_split(np.argsort(p,kind='stable'),15)
        return sum(len(i)/len(y)*abs(y[i].mean()-p[i].mean()) for i in groups if len(i))
    edges=np.linspace(0,1,16);result=0.
    for a,b in zip(edges[:-1],edges[1:]):
        m=(p>=a)&(p<b if b<1 else p<=b)
        if m.any():result+=m.mean()*abs(y[m].mean()-p[m].mean())
    return result


def metrics(y,s,probability=True):
    result=dict(auc_pr=average_precision_score(y,s),auc_roc=roc_auc_score(y,s))
    assert abs(tied_ap(y,s)-result['auc_pr'])<1e-12
    if probability:
        p=np.clip(s,1e-7,1-1e-7)
        result.update(f1=f1_score(y,p>=.5,zero_division=0),ece=ece(y,p),adaptive_ece=ece(y,p,True),
                      nll=log_loss(y,p,labels=[0,1]),brier=brier_score_loss(y,p))
    return result


def run(root):
    root=Path(root);verified=0
    if (root/'manifest.json').exists():
        for e in json.loads((root/'manifest.json').read_text())['files']:
            assert sha(root/e['path'])==e['sha256'],e['path'];verified+=1
    expected=pd.read_csv(root/'per_seed_metrics.csv');registry=pd.read_csv(root/'run_registry.csv')
    assert len(expected)==len(registry)==110
    errors=[]
    for row in expected.itertuples():
        name=f'{row.panel}_{row.arm}_seed{row.seed}_test.csv';path=root/'predictions'/name
        f=pd.read_csv(path,float_precision='round_trip');y=f.label.to_numpy(int);score=f.score.to_numpy(float)
        identity=registry[registry.run_id==f'{row.panel}-{row.arm}-s{row.seed}'].iloc[0]
        assert sha(path)==identity.numeric_test_sha256
        population=pd.read_csv(root/('horizon7_population.csv' if row.panel=='horizon7' else 'original_population.csv'))
        population=population[population.split=='test'].reset_index(drop=True)
        assert f[['sample_id','timestamp','label']].equals(population[['sample_id','timestamp','label']])
        for key,value in metrics(y,score,row.arm!='isolation_forest').items():errors.append(abs(value-getattr(row,key)))
    assert max(errors)<2e-12,max(errors)
    cal=pd.read_csv(root/'modelwise_calibration.csv');cal_errors=[]
    from scipy.optimize import minimize_scalar
    from scipy.special import logit,expit
    for row in cal.itertuples():
        f=pd.read_csv(root/'predictions'/f'{row.panel}_{row.arm}_seed{row.seed}_validation.csv',float_precision='round_trip')
        y=f.label.to_numpy(int);z=logit(np.clip(f.score.to_numpy(),1e-12,1-1e-12))
        def loss(t):
            v=z/np.exp(t);return float(np.mean(np.logaddexp(0,v)-y*v))
        temp=float(np.exp(minimize_scalar(loss,bounds=(-4,4),method='bounded').x))
        assert abs(temp-row.temperature)<1e-9
        f=pd.read_csv(root/'predictions'/f'{row.panel}_{row.arm}_seed{row.seed}_test.csv',float_precision='round_trip')
        p=expit(logit(np.clip(f.score.to_numpy(),1e-12,1-1e-12))/temp)
        for k,v in metrics(f.label.to_numpy(int),p).items():cal_errors.append(abs(v-getattr(row,'ts_'+k)))
    assert max(cal_errors)<2e-12,max(cal_errors)
    return dict(status='PASS',n_test_arrays=len(expected),n_validation_only_calibrators=len(cal),files_verified=verified,
                maximum_point_metric_error=float(max(errors)),maximum_calibrated_metric_error=float(max(cal_errors)),
                raw_data_required=False,weights_required=False,manuscript_sources_required=False,
                scope='local CPU numerical replay, not full retraining or anonymous release verification')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path);args=parser.parse_args()
    here=Path(__file__).resolve().parent
    root=args.root or (here if (here/'predictions').exists() else here.parents[1]/'results/317')
    print(json.dumps(run(root),indent=2))


if __name__=='__main__':main()
