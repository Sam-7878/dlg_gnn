"""Recompute activity strata, prevalence sensitivity, and budget diagnostics."""
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from replay import measure

def resample(scores):
    """Shared seeded draws using frozen held-out order."""
    rng=np.random.default_rng(321)
    rows=[]
    for cohort in ('original','horizon7'):
        arms=['retained_nodes','gin_bce','summary_mlp']
        if cohort=='horizon7':
            arms+=['gin_size','gin_sum','lightgbm_grid','DOMINANT','CoLA','OCGNN']
        label=scores[cohort,arms[0]][0][0]
        pos,neg=np.flatnonzero(label==1),np.flatnonzero(label==0)
        for target in (.0293,.10,.30):
            npos=min(len(pos),int(len(neg)*target/(1-target)))
            nneg=min(len(neg),round(npos*(1-target)/target))
            k=math.ceil(.01*(npos+nneg))
            for draw in range(200):
                idx=np.r_[rng.choice(pos,npos,replace=False),rng.choice(neg,nneg,replace=False)]
                for arm in arms:
                    results=[]
                    for y,s in scores[cohort,arm]:
                        assert np.array_equal(y,label)
                        rec=measure(y[idx],s[idx])
                        order=np.argsort(-s[idx],kind='stable')
                        rec['capture_top1']=y[idx][order[:k]].sum()
                        results.append(rec)
                    rows.append(dict(cohort=cohort,arm=arm,target=target,draw=draw,n=len(idx),positive=npos,k=k,
                        ap=np.mean([r['ap'] for r in results]),roc_auc=np.mean([r['roc_auc'] for r in results]),
                        capture_top1=np.mean([r['capture_top1'] for r in results])))
    return pd.DataFrame(rows)


def main(root):
    scores=defaultdict(list)
    manifest=json.loads((root/'manifest.json').read_text())
    for r in manifest['runs']:
        with np.load(root/r['array']) as a:
            scores[r['cohort'],r['arm']].append((a['label'],a['score']))
    draws=resample(scores)
    pd.testing.assert_frame_equal(draws,pd.read_csv(root/'prevalence_draws.csv'),atol=1e-12,rtol=0)
    summary=draws.groupby(['cohort','arm','target','n','positive','k'],as_index=False).agg(
        ap=('ap','mean'),ap_sd=('ap','std'),roc_auc=('roc_auc','mean'),
        capture_top1=('capture_top1','mean'),capture_top1_sd=('capture_top1','std'))
    pd.testing.assert_frame_equal(summary,pd.read_csv(root/'prevalence_sensitivity.csv'),atol=1e-12,rtol=0)
    strata=np.load(root/'span_stratum.npy')
    for row in pd.read_csv(root/'activity_strata.csv').to_dict('records'):
        mask=strata==row['group']
        ms=[measure(y[mask],s[mask]) for y,s in scores['horizon7',row['arm']]]
        for metric in ms[0]:
            assert np.isclose(np.mean([m[metric] for m in ms]),row[metric],atol=1e-12,rtol=0),(row,metric)
    print(f"PASS: {len(draws)} matched-prevalence rows and all nine activity-stratum controls")

if __name__=='__main__':
    main(Path(sys.argv[1]))
