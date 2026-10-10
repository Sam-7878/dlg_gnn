"""A08 common raw-score evaluation; targets never enter detector fitting."""
import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_score(scores, n):
    scores=np.asarray(scores)
    if scores.shape!=(n,) or not np.isfinite(scores).all(): raise ValueError('score must be finite aligned N-vector')
    return scores.astype(np.float64)


def confusion(y, pred):
    tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum());tn=int(((y==0)&~pred).sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else None
    f1=2*tp/(2*tp+fp+fn) if tp+fn else None
    return {'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1}


def evaluate_scores(labels, scores, node_ids, val_mask, test_mask):
    y=np.asarray(labels);s=validate_score(scores,len(y));ids=np.asarray(node_ids)
    val=np.asarray(val_mask);test=np.asarray(test_mask)
    if y.ndim!=1 or val.dtype!=bool or test.dtype!=bool or val.shape!=y.shape or test.shape!=y.shape or (val&test).any(): raise ValueError('invalid metric population')
    if not np.isin(y[val|test],[0,1]).all() or not val.any() or not test.any(): raise ValueError('invalid metric targets')
    vy,vs=y[val],s[val];ty,ts=y[test],s[test];ti=ids[test]
    op={'threshold_source':'validation','candidate_rule':'distinct raw scores plus reject_all','operator':'>=','tie_rule':'largest threshold','validation_total':len(vy),'validation_positives':int(vy.sum())}
    if not vy.sum(): op.update(threshold_kind='undefined',reason='UNDEFINED_VALIDATION_SUPPORT');thresholded=None
    else:
        # Descending grouped scores; equal-score groups cannot be split.
        order=np.argsort(-vs,kind='stable');sort_s,sort_y=vs[order],vy[order]
        ends=np.flatnonzero(np.r_[sort_s[1:]!=sort_s[:-1],True]);cum=np.cumsum(sort_y);tp=cum[ends];fp=ends+1-tp;fn=vy.sum()-tp
        f1=2*tp/(2*tp+fp+fn);best=int(np.argmax(f1)) # descending threshold resolves ties
        threshold=float(sort_s[ends[best]])
        op.update(threshold_kind='finite',threshold=threshold,validation_confusion=confusion(vy,vs>=threshold))
        thresholded=confusion(ty,ts>=threshold)
    budgets={}
    for q in (.01,.05):
        k=max(1,math.ceil(q*len(ty)));order=np.lexsort((ti,-ts));selection=order[:k];positive=int(ty[selection].sum())
        budgets[str(q)]={'k':k,'precision':positive/k,'recall':positive/int(ty.sum()) if ty.sum() else None,
                         'selected_node_ids':ti[selection].tolist(),'score_boundary_ties':int((ts==ts[selection[-1]]).sum())}
    return {'ap':float(average_precision_score(ty,ts)) if ty.sum() else None,
            'roc_auc':float(roc_auc_score(ty,ts)) if len(np.unique(ty))==2 else None,
            'test_total':len(ty),'test_positives':int(ty.sum()),'test_prevalence':float(ty.mean()),
            'operating_point':op,'thresholded':thresholded,'alert_budgets':budgets,
            'undefined_policy':'AP/recall/F1 undefined with no positives; ROC needs both classes; no predicted positives precision=0'}


def validate_run_identity(record, input_hash, split_hash, config_hash):
    if record.get('input_manifest_hash')!=input_hash or record.get('split_hash')!=split_hash or record.get('model_config_hash')!=config_hash:
        raise ValueError('run/input/split/config identity mismatch')
    if record.get('status')=='SUPPORTED_EXACT' and not record.get('raw_scores_sha256'): raise ValueError('raw score evidence required; scalar metrics cannot reconstruct scores')
