"""Exploratory paired contrasts for the completed supplemental controls.

This is a separate family from the original seven pre-specified comparisons.
It is explicitly post-hoc to the supplemental point estimates.
"""
import numpy as np
import pandas as pd

from common import OUT, SEEDS
from statistical_audit import tied_ap, holm
import historical_statistics as statistics


def main():
    statistics.average_precision=tied_ap
    panels={};identity=None
    for arm in ['gin_bce','gin_focal','sage_bce','sage_focal','summary_lr','summary_mlp','gin_no_chain']:
        panels[arm]=[]
        for seed in SEEDS:
            f=pd.read_csv(OUT/'runs'/arm/f'seed{seed}/test.csv')
            if identity is None:identity=f[['event_id','timestamp','label']]
            assert f[['event_id','timestamp','label']].equals(identity)
            panels[arm].append(f.p_mean.to_numpy(float))
    specs=[('gin_bce','summary_lr'),('gin_bce','summary_mlp'),('gin_bce','sage_bce'),
           ('gin_focal','sage_focal'),('gin_focal','gin_bce'),('gin_bce','gin_no_chain')]
    rows=[]
    for i,(left,right) in enumerate(specs):
        r=statistics.paired_panel_comparison(identity.label.to_numpy(int),panels[left],panels[right],n_resamples=10000,seed=314000+i)
        rows.append(dict(comparison=f'{left} vs {right}',family='exploratory supplemental controls; not primary family',**r))
        pd.DataFrame(rows).to_csv(OUT/'supplement/control_statistical_comparisons.csv',index=False)
        print(left,right,r['mean_auc_pr_difference'],r['randomization_p_value_two_sided'],flush=True)
    f=pd.DataFrame(rows);f['holm_adjusted_p']=holm(f.randomization_p_value_two_sided)
    f.to_csv(OUT/'supplement/control_statistical_comparisons.csv',index=False)


if __name__=='__main__':main()
