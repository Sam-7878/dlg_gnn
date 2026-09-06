"""Matched prediction inference and support-explicit calibration diagnostics."""
from __future__ import annotations
import numpy as np
from scipy.stats import beta, binomtest
from sklearn.metrics import average_precision_score


def metric_vector(y, score, pred):
    y=np.asarray(y); pred=np.asarray(pred)
    tp=np.sum((y==1)&(pred==1)); fn=np.sum((y==1)&(pred==0))
    fp=np.sum((y==0)&(pred==1)); tn=np.sum((y==0)&(pred==0))
    den=float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return np.array([2*tp/max(1,2*tp+fp+fn), average_precision_score(y,score),
        (tp*tn-fp*fn)/den if den else 0.,tp/max(1,tp+fn)])


METRICS=('f1','pr_auc','mcc','fraud_recall')


def paired_bootstrap(y, fast, selective, fast_pred, selective_pred, resamples=2000, seed=20260906):
    """Class-stratified, prediction-paired bootstrap, conditional on this model/split."""
    y=np.asarray(y); rng=np.random.default_rng(seed)
    classes=[np.flatnonzero(y==c) for c in (0,1)]
    if any(len(c)==0 for c in classes): raise ValueError('bootstrap requires two classes')
    delta=metric_vector(y,selective,selective_pred)-metric_vector(y,fast,fast_pred)
    samples=np.empty((resamples,4))
    for b in range(resamples):
        ix=np.concatenate([rng.choice(c,len(c),replace=True) for c in classes])
        samples[b]=metric_vector(y[ix],selective[ix],selective_pred[ix])-metric_vector(y[ix],fast[ix],fast_pred[ix])
    bounds=np.quantile(samples,[.025,.975],axis=0)
    return {m:{'delta':float(delta[i]),'ci_low':float(bounds[0,i]),'ci_high':float(bounds[1,i])} for i,m in enumerate(METRICS)},samples


def mcnemar(y, fast_pred, selective_pred):
    a=np.asarray(fast_pred)==y; b=np.asarray(selective_pred)==y
    improved=int((~a&b).sum()); worsened=int((a&~b).sum()); n=improved+worsened
    return {'fast_wrong_selective_correct':improved,'fast_correct_selective_wrong':worsened,
        'p_value':float(binomtest(improved,n,.5).pvalue) if n else 1.,
        'paired_error_rate_reduction':(improved-worsened)/len(y),
        'matched_error_odds_ratio':improved/worsened if worsened else None,
        'odds_undefined_reason':'zero denominator' if not worsened else ''}


def holm(pvalues):
    p=np.asarray(pvalues); order=np.argsort(p); adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
    return adjusted


def exact_upper(errors, support, alpha=.05):
    if not 0 <= errors <= support: raise ValueError('invalid binomial support')
    if support==0: return None
    return 1. if errors==support else float(beta.ppf(1-alpha,errors+1,support-errors))


def reliability(y, scores, bins=10, adaptive=False):
    y=np.asarray(y); scores=np.asarray(scores)
    if adaptive:
        # Keep tied scores together; never create a fictitious additional coverage point.
        edges=np.unique(np.r_[0,np.quantile(scores,np.linspace(0,1,bins+1)),1])
    else: edges=np.linspace(0,1,bins+1)
    assignments=np.clip(np.searchsorted(edges,scores,side='right')-1,0,len(edges)-2)
    rows=[]
    for i in range(len(edges)-1):
        mask=assignments==i; n=int(mask.sum())
        if not n: continue
        k=int(y[mask].sum()); observed=k/n
        lo=0. if k==0 else float(beta.ppf(.025,k,n-k+1))
        hi=1. if k==n else float(beta.ppf(.975,k+1,n-k))
        rows.append({'bin':i,'N':n,'N_positive':k,'confidence':float(scores[mask].mean()),
            'observed_fraction':observed,'ci_low':lo,'ci_high':hi,'edge_low':float(edges[i]),'edge_high':float(edges[i+1])})
    return rows


def calibration_metrics(y,scores):
    y=np.asarray(y); scores=np.asarray(scores); clip=np.clip(scores,1e-12,1-1e-12)
    def ece(yy,pp,bins=10,adaptive=False):
        return sum(r['N']*abs(r['confidence']-r['observed_fraction']) for r in reliability(yy,pp,bins,adaptive))/len(yy)
    return {'N':len(y),'N_positive':int(y.sum()),'nll':float(-np.mean(y*np.log(clip)+(1-y)*np.log1p(-clip))),
        'brier':float(np.mean((scores-y)**2)),'ece_10':ece(y,scores), 'ece_20':ece(y,scores,20),
        'adaptive_ece':ece(y,scores,10,True),
        'classwise_ece':(ece(y,scores)+ece(1-y,1-scores))/2,
        'classwise_definition':'macro binary one-vs-rest equal-width 10-bin ECE'}
