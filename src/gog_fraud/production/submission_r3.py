"""One calibrated execution contract shared by R3 offline and streaming lanes."""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.special import expit, logit
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, matthews_corrcoef, roc_auc_score
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from evidence.experiment_identity import digest, file_digest
from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig
from gog_fraud.models.level2.model import Level2Model, Level2ModelConfig

CONFIG = Path('configs/sci_v3_submission_r3/primary_operating_point.yaml')


def read_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg['base'] = yaml.safe_load(Path(cfg['source_config']).read_text())
    cfg['selection'] = yaml.safe_load(Path(cfg['selection_config']).read_text())['calibration']
    return cfg


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def load_models(cfg, seed, device):
    path = Path(cfg['checkpoint_root']) / f'seed{seed}'
    models = []
    for name, model_type, config_type in [('level1', Level1Model, Level1ModelConfig), ('level2', Level2Model, Level2ModelConfig)]:
        payload = torch.load(path / f'{name}.pt', map_location='cpu', weights_only=False)
        model = model_type(config_type(**payload['config']))
        model.load_state_dict(payload['state_dict']); model.to(device).eval(); models.append(model)
    return models


@torch.inference_mode()
def local_predict(model, batch, T):
    model.train(T > 1)
    outputs = [model(batch) for _ in range(T)]
    score = torch.stack([out.score.reshape(-1) for out in outputs])
    embedding = torch.stack([out.embedding for out in outputs]).mean(0)
    model.eval()
    return score.mean(0).cpu().numpy(), embedding.cpu().numpy(), score.var(0, unbiased=False).cpu().numpy()


def infer_local(model, graphs, T, device):
    scores, embeddings, variances = [], [], []
    for batch in DataLoader(graphs, batch_size=128, shuffle=False):
        score, emb, var = local_predict(model, batch.to(device), T)
        scores.append(score); embeddings.append(emb); variances.append(var)
    return np.concatenate(scores), np.concatenate(embeddings), np.concatenate(variances)


class FrozenRelations:
    """Each target sees k frozen train neighbors; no target-to-target messages."""
    def __init__(self, embeddings, scores, k):
        self.embeddings = embeddings
        self.scores = scores
        self.k = min(k, len(scores))
        self.tree = cKDTree(embeddings)

    def graphs(self, embeddings, scores):
        _, neighbors = self.tree.query(embeddings, k=self.k, workers=1)
        neighbors = np.asarray(neighbors).reshape(len(scores), self.k)
        output = []
        for emb, score, indices in zip(embeddings, scores, neighbors):
            features = np.column_stack((self.embeddings[indices], self.scores[indices]))
            features = np.concatenate((features, np.r_[emb, score][None]), axis=0).astype('float32')
            refs = np.arange(self.k)
            edge_index = np.stack((np.r_[refs, np.full(self.k, self.k)], np.r_[np.full(self.k, self.k), refs]))
            output.append(Data(x=torch.from_numpy(features), edge_index=torch.from_numpy(edge_index).long()))
        return output

    @torch.inference_mode()
    def predict(self, model, embeddings, scores, device):
        values = []
        for start in range(0, len(scores), 128):
            graphs = self.graphs(embeddings[start:start+128], scores[start:start+128])
            batch = Batch.from_data_list(graphs).to(device)
            model.eval()
            values.append(model(batch).score.reshape(-1)[batch.ptr[1:] - 1].cpu().numpy())
        return np.concatenate(values)


def logits(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError('expected finite probabilities in [0,1]')
    return logit(np.clip(scores, 1e-6, 1-1e-6))


def fit_map(scores, labels):
    if set(np.unique(labels)) != {0, 1}: raise ValueError('two validation classes required')
    model = LogisticRegression(C=1., solver='lbfgs').fit(logits(scores)[:, None], labels)
    return {'coefficient': float(model.coef_[0,0]), 'intercept': float(model.intercept_[0])}


def calibrate(scores, mapping):
    return expit(mapping['coefficient'] * logits(scores) + mapping['intercept'])


def best_threshold(labels, scores):
    """Exact grouped F1 search, including equal-score ties, in O(n log n)."""
    candidates = np.unique(np.r_[0., scores, 1.])
    order = np.argsort(scores); sorted_scores = scores[order]; y = np.asarray(labels)[order]
    positions = np.searchsorted(sorted_scores, candidates, side='left')
    cumulative = np.r_[0, np.cumsum(y)]
    tp = cumulative[-1] - cumulative[positions]
    denom = cumulative[-1] + len(y) - positions
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom > 0)
    indices = np.flatnonzero(f1 == f1.max())
    chosen = indices[np.argmin(np.abs(candidates[indices] - .5))]
    return float(candidates[chosen]), float(f1[chosen])


def select_policy(valid_raw_fast, valid_raw_deep, labels, cfg, T):
    fm, dm = fit_map(valid_raw_fast, labels), fit_map(valid_raw_deep, labels)
    fast, deep = calibrate(valid_raw_fast, fm), calibrate(valid_raw_deep, dm)
    ft, _ = best_threshold(labels, fast)
    best = None
    for budget in cfg['selection']['deep_budget_grid']:
        cutoff = float(np.quantile(np.abs(fast-ft), budget, method='higher'))
        route = np.abs(fast-ft) <= cutoff
        for weight in cfg['selection']['fast_weight_grid']:
            fusion = expit(weight*logits(fast) + (1-weight)*logits(deep))
            final = np.where(route, fusion, fast)
            threshold, f1 = best_threshold(labels, final)
            rank = (f1, -float(route.mean()), -abs(weight-.5))
            if best is None or rank > best[0]:
                best = (rank, {'fast_map': fm, 'deep_map': dm, 'fast_threshold': ft,
                    'route_cutoff': cutoff, 'fast_weight': weight, 'final_threshold': threshold,
                    'validation_f1': f1, 'validation_deep_rate': float(route.mean()),
                    'requested_budget': budget, 'mc_T': T})
    selection = best[1]
    full = expit(selection['fast_weight']*logits(fast) + (1-selection['fast_weight'])*logits(deep))
    selection['full_threshold'], _ = best_threshold(labels, full)
    selection['selection_partition'] = 'validation'
    selection['policy_config_id'] = digest(selection)
    return selection


def apply_policy(raw_fast, raw_deep, selection, policy='primary'):
    fast = calibrate(raw_fast, selection['fast_map'])
    route = np.abs(fast-selection['fast_threshold']) <= selection['route_cutoff']
    if policy == 'full_deep': route = np.ones(len(fast), dtype=bool)
    if policy == 'direct_only': route = np.zeros(len(fast), dtype=bool)
    final = fast.copy()
    if route.any():
        deep = calibrate(np.asarray(raw_deep)[route], selection['deep_map'])
        final[route] = expit(selection['fast_weight']*logits(fast[route]) + (1-selection['fast_weight'])*logits(deep))
    threshold = selection[{'primary':'final_threshold','full_deep':'full_threshold','direct_only':'fast_threshold'}[policy]]
    return final, route, (final >= threshold).astype(int)


def metrics(y, score, prediction):
    y = np.asarray(y, dtype=int); prediction = np.asarray(prediction, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0,1]).ravel()
    pos, neg = int(y.sum()), int(len(y)-y.sum())
    return {'N':len(y), 'N_positive':pos,'N_negative':neg,
        'tp':int(tp),'tn':int(tn),'fp':int(fp),'fn':int(fn),
        'f1': float(2*tp/(2*tp+fp+fn)) if pos and (2*tp+fp+fn) else None,
        'precision':float(tp/(tp+fp)) if tp+fp else None,
        'fraud_recall':float(tp/pos) if pos else None,'fnr':float(fn/pos) if pos else None,
        'mcc':float(matthews_corrcoef(y,prediction)) if pos and neg else None,
        'pr_auc':float(average_precision_score(y,score)) if pos and neg else None,
        'roc_auc':float(roc_auc_score(y,score)) if pos and neg else None,
        'balanced_accuracy':float((tp/pos+tn/neg)/2) if pos and neg else None,
        'undefined_reason': '' if pos and neg else 'single-class support'}
