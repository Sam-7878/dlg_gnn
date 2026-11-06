"""Summarize window-specific test points with preserved seven-day seed seven."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

project=Path(__file__).resolve().parents[2]
results=project/'results/322'
prior=project/'results/319'
source=project/'private/322/windows'
records=[json.loads((source/f'{d}_{arm}.json').read_text()) for d in (1,3,14) for arm in ('node_lr','mlp','gin_size')]
for arm,p in [('node_lr',prior/'predictions/horizon7_single_retained_nodes_test.csv'),
              ('mlp',prior/'runs/horizon7/mlp_full/seed7/test.csv'),
              ('gin_size',prior/'runs/horizon7/gin_size/seed7/test.csv')]:
    f=pd.read_csv(p)
    score=f['score' if 'score' in f else 'p_mean']
    records.append(dict(days=7,arm=arm,seed=7,n=len(f),positive=int(f.label.sum()),
                        prevalence=f.label.mean(),ap=average_precision_score(f.label,score),roc_auc=roc_auc_score(f.label,score)))
frame=pd.DataFrame(records).sort_values(['days','arm'])
frame.to_csv(results/'horizon_sensitivity.csv',index=False)
fig,ax=plt.subplots(figsize=(7.1,3.7))
for arm,name in [('node_lr','Node-count LR'),('mlp','Summary MLP'),('gin_size','GIN-size')]:
    sub=frame[frame.arm==arm]
    ax.plot(sub.days,sub.ap,marker='o',label=name)
sub=frame[frame.arm=='node_lr']
ax.plot(sub.days,sub.prevalence,marker='s',ls='--',color='#333333',label='Positive prevalence')
ax.set(xlabel='Observation window (days)',ylabel='Average precision or prevalence',xticks=sub.days,ylim=(0,1.04))
ax.set_xticklabels([f'{r.days}\n{r.positive:,}/{r.n:,}' for r in sub.itertuples()])
ax.grid(alpha=.2)
ax.legend(loc='lower right',frameon=False,ncols=2)
fig.tight_layout()
fig.savefig(results/'horizon_sensitivity.pdf')
