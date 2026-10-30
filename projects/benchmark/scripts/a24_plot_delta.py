"""Plot frozen seed-paired AP differences with sample SD; no primary value changes."""
from a24_common import *
import os
os.environ.setdefault('MPLBACKEND','Agg')
import matplotlib.pyplot as plt
rs=sorted(rows(PUB/'paired_deltas_a24.csv'),key=lambda r:float(r['mean_delta_AP']));ys=list(range(len(rs)));means=[float(r['mean_delta_AP']) for r in rs];sd=[float(r['sample_SD']) for r in rs]
fig,ax=plt.subplots(figsize=(7.1,5.1));ax.barh(ys,means,xerr=sd,color=['#2864aa' if x>=0 else '#b94646' for x in means],error_kw={'elinewidth':1,'capsize':2});ax.set_yticks(ys,[r['dataset'] for r in rs]);ax.axvline(0,color='#333333',linewidth=.8);ax.set_xlabel('Mean paired AP difference (Aug - Base); error bars: sample SD');ax.set_title('Frozen primary results: five matched seeds per graph',fontsize=11);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);fig.tight_layout()
for v in ['applsci','preprints']:
 out=PAPER/v/'figures/fig_a24_delta_sample_sd.pdf';fig.savefig(out,metadata={'CreationDate':None,'ModDate':None})
print('Primary paired-delta plot regenerated with ddof=1 SD')
