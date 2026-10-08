#!/usr/bin/env python3
"""Diagnostic-only one-step production-path comparison for CONAD Eq.(1)."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr,spearmanr
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import benchmark_8x10_pipeline as legacy
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.models.pygod.shared_reconstruction import SharedCONAD
OUT=ROOT/'evaluation/benchmark/v2/diagnostics/conad_dominant/conad_eq1_integration_a04.json'


class AuditReference(SharedCONAD):
    def init_model(self,**kwargs):
        model=super().init_model(**kwargs)
        self._initial={name:p.detach().cpu().clone() for name,p in model.named_parameters()}
        return model
    def _training_extra_loss(self,data,components):
        loss=super()._training_extra_loss(data,components)
        grads=torch.autograd.grad(loss,self.model.parameters(),retain_graph=True,allow_unused=True)
        self.contrastive_gradient_norm=sum(float(g.detach().square().sum()) for g in grads if g is not None)**0.5
        return loss


class AuditPaperEq1(AuditReference):
    def _forward_components(self,data):
        if self.model.training:
            x_aug,edge_aug,labels=self._sparse_data_augmentation(data)
            augmented_graph=self._message_graph(data,x_aug.dtype,edge_index=edge_aug,cache=False)
            self.model(x_aug,augmented_graph)
            h_aug=self.model.emb
        components=super(SharedCONAD,self)._forward_components(data)
        if self.model.training:
            h=self.model.emb
            distance=torch.linalg.vector_norm(h-h_aug,ord=2,dim=-1)
            label=labels.to(distance.dtype)
            # Published Siamese Eq.(1): unsquared Euclidean + unsquared hinge.
            loss=((1-label)*distance+label*torch.relu(self.margin_loss_func.margin-distance)).mean()
            self._contrastive_extra=(1-self.eta)*loss
        return components


def checksum(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run(cls,data):
    seed_everything(42,deterministic=True)
    det=cls(epoch=1,gpu=0 if torch.cuda.is_available() else -1,verbose=0,batch_size=0,margin=0.5)
    graph=data.clone();det.fit(graph)
    delta=sum(float((p.detach().cpu()-det._initial[name]).square().sum()) for name,p in det.model.named_parameters())**0.5
    scores=det.decision_function(graph)
    scores=scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
    return {'contrastive_gradient_norm':det.contrastive_gradient_norm,
            'encoder_parameter_delta_after_one_step':delta,
            'final_parameter_checksum':checksum(det.model),
            'raw_score_sha256':hashlib.sha256(np.asarray(scores).tobytes()).hexdigest()},scores


def main():
    legacy.DATA_ROOT='/mnt/d/_Work/_data/DLG';legacy.DATASET_SEED=42
    seed_everything(42,deterministic=True)
    data=legacy.load_planetoid('Cora')
    reference,a=run(AuditReference,data)
    eq1,b=run(AuditPaperEq1,data)
    assert reference['contrastive_gradient_norm']==0.0
    assert eq1['contrastive_gradient_norm']>0.0
    assert reference['final_parameter_checksum']!=eq1['final_parameter_checksum']
    assert reference['raw_score_sha256']!=eq1['raw_score_sha256']
    result={'dataset':'Cora-Syn','seed':42,'epochs':1,'scope':'APPENDIX_DIAGNOSTIC_ONLY',
        'paper_equation':'(1-y)*||h-h_aug||_2 + y*max(0,margin-||h-h_aug||_2)',
        'reference':reference,'experimental_paper_eq1':eq1,
        'score_pearson':float(pearsonr(a,b).statistic),'score_spearman':float(spearmanr(a,b).statistic),
        'primary_ranking_eligible':False}
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
