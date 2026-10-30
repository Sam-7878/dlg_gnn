"""Tie-aware saved-score metrics and conditional contract-bootstrap intervals."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,log_loss,f1_score


def ap(y,s):
    y=np.asarray(y);s=np.asarray(s)
    if y.sum()==0:return np.nan
    order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    tp=np.cumsum(ys)[ends]
    return float(np.sum(np.diff(np.r_[0,tp])*tp/(ends+1))/tp[-1])


def auc(y,s):
    y=np.asarray(y);s=np.asarray(s);positive=y.sum();negative=len(y)-positive
    if positive==0 or negative==0:return np.nan
    order=np.argsort(s,kind='stable');ys=y[order];ss=s[order]
    ends=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    pos=np.diff(np.r_[0,np.cumsum(ys)[ends]])
    neg=np.diff(np.r_[0,ends+1])-pos
    return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(positive*negative))


def ece(y,p,adaptive=False):
    if adaptive:
        return sum(len(i)/len(y)*abs(y[i].mean()-p[i].mean()) for i in np.array_split(np.argsort(p,kind='stable'),15) if len(i))
    result=0.
    for a,b in zip(np.linspace(0,1,16)[:-1],np.linspace(0,1,16)[1:]):
        mask=(p>=a)&(p<b if b<1 else p<=b)
        if mask.any():result+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return float(result)


def point(y,s,probability=True):
    y=np.asarray(y,int);s=np.asarray(s,float)
    r=dict(ap=float(average_precision_score(y,s)),roc_auc=float(roc_auc_score(y,s)))
    assert abs(r['ap']-ap(y,s))<1e-12 and abs(r['roc_auc']-auc(y,s))<1e-12
    if probability:
        p=np.clip(s,1e-7,1-1e-7)
        r.update(ece=ece(y,p),adaptive_ece=ece(y,p,True),nll=float(log_loss(y,p,labels=[0,1])),
                 brier=float(brier_score_loss(y,p)),f1_at_05=float(f1_score(y,p>=.5,zero_division=0)))
    return r


def conditional_summary(y,matrix,seed=319,draws=2000):
    y=np.asarray(y,int);matrix=np.atleast_2d(matrix)
    n=len(y);positive=int(y.sum());negative=n-positive
    r=dict(n=n,positive=positive,negative=negative,prevalence=positive/n if n else np.nan,
           ap=np.nan,roc_auc=np.nan,ap_ci95_low=np.nan,ap_ci95_high=np.nan,
           roc_ci95_low=np.nan,roc_ci95_high=np.nan,ci_draws_valid=0,
           interval_unit='contract observations; fixed fitted models; no actor/future-deployment inference')
    if min(positive,negative)==0:
        r['support_status']='ONE_CLASS_OR_EMPTY: point AP/ROC not reported';return r
    r.update(ap=float(np.mean([ap(y,s) for s in matrix])),roc_auc=float(np.mean([auc(y,s) for s in matrix])))
    if min(positive,negative)<20:
        r['support_status']='LOW_SUPPORT: point only, CI not reported';return r
    rng=np.random.default_rng(seed);aps=[];aucs=[]
    for _ in range(draws):
        ids=rng.integers(0,n,n);yy=y[ids]
        if yy.sum()==0 or yy.sum()==n:continue
        aps.append(np.mean([ap(yy,s[ids]) for s in matrix]));aucs.append(np.mean([auc(yy,s[ids]) for s in matrix]))
    a=np.quantile(aps,[.025,.975]);b=np.quantile(aucs,[.025,.975])
    r.update(ap_ci95_low=float(a[0]),ap_ci95_high=float(a[1]),roc_ci95_low=float(b[0]),roc_ci95_high=float(b[1]),
             ci_draws_valid=len(aps),support_status='CONDITIONAL_EVENT_BOOTSTRAP')
    return r


def operating(y,s,budgets):
    y=np.asarray(y,int);s=np.asarray(s,float);order=np.argsort(-s,kind='stable');rows=[]
    for k in sorted(set(budgets)):
        if k>len(y):continue
        cutoff=s[order[k-1]];above=s>cutoff;tied=s==cutoff
        slots=k-int(above.sum());nt=int(tied.sum());pt=int(y[tied].sum());base=int(y[above].sum())
        capture=int(y[order[:k]].sum())
        rows.append(dict(budget=k,captured=capture,precision=capture/k,recall=capture/y.sum(),
                         boundary_tie_size=nt,tie_expected_captured=base+slots*pt/nt,
                         tie_min_captured=base+max(0,slots-(nt-pt)),tie_max_captured=base+min(slots,pt)))
    return rows
