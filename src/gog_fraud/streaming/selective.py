"""Reusable deterministic selective models, calibration and routing semantics."""
from __future__ import annotations
import contextlib
import random
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GINConv,GATv2Conv,global_mean_pool,global_max_pool
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score,roc_auc_score,confusion_matrix,matthews_corrcoef)


def seed_all(seed,threads=1):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


class LocalEncoder(nn.Module):
    def __init__(self,backbone='GIN',hidden=32,dropout=.2):
        super().__init__();self.backbone=backbone;self.dropout=nn.Dropout(dropout)
        if backbone=='GIN':
            self.layers=nn.ModuleList([GINConv(nn.Sequential(nn.Linear(i,hidden),nn.ReLU(),nn.Linear(hidden,hidden))) for i in (3,hidden)])
        elif backbone=='GATv2':
            self.layers=nn.ModuleList([GATv2Conv(i,hidden//4,heads=4,dropout=dropout) for i in (3,hidden)])
        else:raise ValueError('unknown local backbone')
        self.head=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))

    def forward(self,data):
        x=data.x
        for layer in self.layers:x=self.dropout(torch.relu(layer(x,data.edge_index)))
        batch=getattr(data,'batch',None)
        if batch is None:batch=torch.zeros(len(x),dtype=torch.long,device=x.device)
        emb=torch.cat((global_mean_pool(x,batch),global_max_pool(x,batch)),dim=1)
        return self.head(emb).reshape(-1),emb


class RelationalEncoder(nn.Module):
    def __init__(self,input_dim=65,hidden=32,heads=4,dropout=.2):
        super().__init__();self.norm=nn.LayerNorm(input_dim)
        self.layers=nn.ModuleList([GATv2Conv(i,hidden//heads,heads=heads,dropout=dropout,add_self_loops=True) for i in (input_dim,hidden)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden),nn.LayerNorm(hidden)])
        self.dropout=nn.Dropout(dropout);self.head=nn.Linear(hidden,1)

    def forward(self,data):
        x=self.norm(data.x)
        for layer,norm in zip(self.layers,self.norms):x=self.dropout(torch.nn.functional.elu(norm(layer(x,data.edge_index))))
        logits=self.head(x).reshape(-1)
        indices=data.ptr[1:]-1 if hasattr(data,'ptr') else torch.tensor([len(x)-1],device=x.device)
        return logits[indices]


class DenseClassifier(nn.Module):
    def __init__(self,input_dim,hidden=32):
        super().__init__();self.net=nn.Sequential(nn.Linear(input_dim,hidden),nn.ReLU(),nn.Dropout(.2),nn.Linear(hidden,1))
    def forward(self,x):return self.net(x).reshape(-1)


@contextlib.contextmanager
def dropout_only(model,enabled):
    """Freeze all stateful layers; turn on only declared dropout mechanisms."""
    states={m:m.training for m in model.modules()};model.eval()
    for module in model.modules():
        if isinstance(module,(nn.Dropout,GATv2Conv)):module.train(enabled)
    try:yield
    finally:
        for module,state in states.items():module.training=state


@torch.inference_mode()
def local_predict(model,data,T=1):
    with dropout_only(model,T>1):
        outputs=[model(data) for _ in range(T)]
        scores=torch.stack([torch.sigmoid(logits) for logits,_ in outputs])
        embedding=torch.stack([emb for _,emb in outputs]).mean(0)
    return scores.mean(0).cpu().numpy(),embedding.cpu().numpy(),scores.var(0,unbiased=False).cpu().numpy()


def log_odds(scores):return logit(np.clip(np.asarray(scores,float),1e-6,1-1e-6))


def fit_calibration(scores,y):
    if len(np.unique(y))<2:raise ValueError('calibration partition lacks both classes')
    model=LogisticRegression(C=1,solver='lbfgs').fit(log_odds(scores)[:,None],y)
    return {'a':float(model.coef_[0,0]),'b':float(model.intercept_[0])}


def calibrated(scores,mapping):return expit(mapping['a']*log_odds(scores)+mapping['b'])


def fuse(fast,deep,weight):return expit(weight*log_odds(fast)+(1-weight)*log_odds(deep))


def fit_threshold(y,scores):
    scores=np.asarray(scores);y=np.asarray(y,dtype=int)
    if not y.sum():raise ValueError('undefined positive support for threshold selection')
    order=np.argsort(scores,kind='stable');p=scores[order];yy=y[order]
    candidates=np.r_[np.unique(p),np.nextafter(p.max(),np.inf)]
    positions=np.searchsorted(p,candidates,'left');cumulative=np.r_[0,np.cumsum(yy)]
    tp=cumulative[-1]-cumulative[positions];den=cumulative[-1]+len(y)-positions
    f1=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0)
    ix=np.flatnonzero(f1==f1.max())[-1]
    return float(candidates[ix]),float(f1[ix])


def binary_metrics(y,score,pred):
    y=np.asarray(y,int);pred=np.asarray(pred,int)
    tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel());pos=int(y.sum());neg=len(y)-pos
    return {'N':len(y),'N_positive':pos,'tp':tp,'tn':tn,'fp':fp,'fn':fn,
        'ap':float(average_precision_score(y,score)) if pos and neg else None,
        'roc_auc':float(roc_auc_score(y,score)) if pos and neg else None,
        'f1':2*tp/(2*tp+fp+fn) if pos and (2*tp+fp+fn) else None,
        'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/pos if pos else None,
        'mcc':float(matthews_corrcoef(y,pred)) if pos and neg else None,
        'fpr':fp/neg if neg else None,'specificity':tn/neg if neg else None,
        'alert_rate':float(pred.mean()),'fp_per_1000':1000*fp/len(y),
        'undefined_reason':'' if pos and neg else 'negative-only slice' if not pos else 'positive-only slice'}


def router_score(fast,threshold,family):
    if family=='margin':return -abs(fast-threshold)
    if family=='entropy':
        p=np.clip(fast,1e-12,1-1e-12);return -(p*np.log(p)+(1-p)*np.log1p(-p))
    raise ValueError('router score requires fitted benefit or recorded random generator')


def risk_counts(y,pred,route):
    y=np.asarray(y);pred=np.asarray(pred);direct=~np.asarray(route,bool)
    errors=int((direct&(y!=pred)).sum());miss=int((direct&(y==1)&(pred==0)).sum())
    definitions={'selective_error':(errors,int(direct.sum())),
        'direct_fraud_FNR':(miss,int((direct&(y==1)).sum())),
        'population_direct_fraud_miss':(miss,int((y==1).sum())),
        'false_omission_rate':(miss,int((direct&(pred==0)).sum()))}
    result={'coverage':float(direct.mean()),'N_direct':int(direct.sum()),'N_deep':int((~direct).sum())}
    for name,(num,den) in definitions.items():result.update({name:num/den if den else None,name+'_numerator':num,name+'_denominator':den})
    return result
