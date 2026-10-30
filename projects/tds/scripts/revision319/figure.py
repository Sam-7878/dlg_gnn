"""Publication vector chart from the frozen first-transfer composition table."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import OUT,sha,write

df=pd.read_csv(OUT/'first_time_quarterly_composition.csv')
all_rows=df[df.chain=='all'].reset_index(drop=True)
labels=all_rows.period.tolist();x=np.arange(len(labels));fig,ax=plt.subplots(2,1,figsize=(9.2,5.8),sharex=True,gridspec_kw={'height_ratios':[1.6,1]})
palette={'all':'#111827','ethereum':'#2563eb','bsc':'#d97706','polygon':'#059669'}
for chain in ['all','ethereum','bsc','polygon']:
    t=df[df.chain==chain].set_index('period').reindex(labels)
    eligible=(t.n>=20).fillna(False).to_numpy()
    if not eligible.any():continue
    xx=x[eligible];p=t.prevalence.to_numpy()[eligible];lo=t.wilson95_low.to_numpy()[eligible];hi=t.wilson95_high.to_numpy()[eligible]
    ax[0].errorbar(xx,p,yerr=[np.maximum(0,p-lo),np.maximum(0,hi-p)],fmt='o-',ms=2.7,lw=1,capsize=1.5,label=chain.title() if chain!='bsc' else 'BSC',color=palette[chain],alpha=.85)
ax[0].set_ylabel('Positive fraction (Wilson 95% interval)');ax[0].set_ylim(-.03,1.03);ax[0].legend(ncol=4,fontsize=8,frameon=False)
for chain in ['ethereum','bsc','polygon']:
    t=df[df.chain==chain].set_index('period').reindex(labels)
    ax[1].plot(x,t.n.fillna(0).to_numpy(),label=chain.title() if chain!='bsc' else 'BSC',lw=1.5,color=palette[chain])
ax[1].set_yscale('symlog',linthresh=10);ax[1].set_ylabel('Contracts by chain');ax[1].set_xlabel('First valid transfer quarter (UTC)')
ax[1].set_xticks(x[::max(1,len(x)//12)]);ax[1].set_xticklabels([labels[i] for i in x[::max(1,len(x)//12)]],rotation=35,ha='right')
ax[1].grid(alpha=.2);ax[0].grid(alpha=.2)
fig.tight_layout();dst=OUT/'figures/first_time_quarterly_composition.pdf';dst.parent.mkdir(parents=True,exist_ok=True);fig.savefig(dst,bbox_inches='tight');plt.close(fig)
write(OUT/'figures/first_time_quarterly_composition.json',dict(source_sha256=sha(OUT/'first_time_quarterly_composition.csv'),pdf_sha256=sha(dst),support_threshold=20,wilson95=True))
from matplotlib.patches import FancyBboxPatch
fig,ax=plt.subplots(figsize=(10,3.3));ax.axis('off')
boxes=[(.02,.59,.21,.29,'Define sample and label time\nUnknown discovery time?\nRetrospective scope only'),(.28,.59,.21,.29,'Recover first/last times\nand archive provenance\nAudit sorting and lifetimes'),(.54,.59,.21,.29,'Material composition change?\nYes: separate horizon cohort\nNo: report diagnostic limits'),(.79,.59,.19,.29,'Fix primary panel\nMatch inputs and loss\nKeep archival runs separate'),(.16,.10,.30,.26,'Validate calibration and ranking\nValidation-only fitting; AP ties\nDeclare multiplicity and dependence'),(.56,.10,.30,.26,'Archive code, predictions, hashes\nand source-to-tensor audit\nState admissibility and limits')]
for px,py,w,h,s in boxes:
    ax.add_patch(FancyBboxPatch((px,py),w,h,boxstyle='round,pad=0.008',fc='#eef4fa',ec='#365c7c'))
    ax.text(px+w/2,py+h/2,s,ha='center',va='center',fontsize=9)
for a,b in [((.235,.735),(.275,.735)),((.495,.735),(.535,.735)),((.755,.735),(.785,.735)),((.465,.23),(.55,.23))]:
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#365c7c'})
ax.plot([.88,.88,.10,.10],[.57,.45,.45,.23],color='#365c7c',linewidth=1)
ax.annotate('',xy=(.15,.23),xytext=(.10,.23),arrowprops={'arrowstyle':'->','color':'#365c7c'})
fig.tight_layout();decision=OUT/'figures/validity_decision_tree_319.pdf';fig.savefig(decision);plt.close(fig)
write(OUT/'figures/validity_decision_tree_319.json',dict(pdf_sha256=sha(decision),internal_release_workflow_removed=True))
print(dst)
