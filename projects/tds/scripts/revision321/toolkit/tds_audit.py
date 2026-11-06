"""Audit contract observation order, temporal admissibility and ranking."""
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('ordering','cutoff','counts','ap'))
    parser.add_argument('csv')
    parser.add_argument('--features',nargs='+',default=['retained_nodes'])
    args=parser.parse_args()
    data=pd.read_csv(args.csv)
    if data.empty:
        parser.error('input is empty')
    if args.command=='cutoff':
        clocks=data[['cutoff','max_input_timestamp']]
        if not np.isfinite(clocks.to_numpy()).all():
            parser.error('cutoff clocks must be finite')
        violations=int((clocks.max_input_timestamp>clocks.cutoff).sum())
        print(json.dumps({'rows':len(data),'violations':violations}))
        return int(violations>0)
    if not data.label.isin([0,1]).all():
        parser.error('labels must be binary')
    if args.command=='ordering':
        if data.sample_id.isna().any() or data.sample_id.duplicated().any():
            parser.error('sample IDs must be unique and present')
        report={}
        for key in ('first_valid_tx_time','last_valid_tx_time'):
            if not np.isfinite(data[key].to_numpy()).all():
                parser.error(f'{key} must be finite')
            ordered=data.sort_values([key,'sample_id']).label.to_numpy()
            parts=np.split(ordered,[int(.7*len(data)),int(.85*len(data))])
            report[key]=[dict(n=len(y),positive=int(y.sum()),prevalence=float(y.mean()) if len(y) else None) for y in parts]
        print(json.dumps(report))
        return 0
    if args.command=='counts':
        train,test=data[data.split=='train'],data[data.split=='test']
        if train.label.nunique()!=2 or test.label.nunique()!=2:
            parser.error('both classes are required in train and test')
        features=data[args.features].to_numpy()
        if (features<0).any() or not np.isfinite(features).all():
            parser.error('counts must be nonnegative and finite')
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',max_iter=2000))
        model.fit(np.log1p(train[args.features]),train.label)
        y,score=test.label,model.predict_proba(np.log1p(test[args.features]))[:,1]
    else:
        y,score=data.label,data.score
    if y.nunique()!=2 or not np.isfinite(score).all():
        parser.error('metrics require both classes and finite scores')
    prevalence=float(y.mean())
    ap=float(average_precision_score(y,score))
    print(json.dumps(dict(n=len(y),prevalence=prevalence,ap=ap,lift=ap-prevalence,roc_auc=roc_auc_score(y,score))))
    return 0
