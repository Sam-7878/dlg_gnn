"""Submission R4 production pipeline: timestamp-aware relation filtering, policy selection, and audit tracking."""
from __future__ import annotations
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from evidence.experiment_identity import digest, file_digest
from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig
from gog_fraud.models.level2.model import Level2Model, Level2ModelConfig


def _resolve_cfg_path(p):
    path = Path(p)
    if path.exists():
        return path
    s = str(p).replace('\\', '/')
    for prefix in ('configs/stream_mc/', 'configs/dlg_gnn/', 'configs/benchmark/', 'configs/graph_rag/'):
        if s.startswith('configs/'):
            cand = Path(prefix + s[len('configs/'):])
            if cand.exists():
                return cand
        cand = Path(prefix + s)
        if cand.exists():
            return cand
    return path


def read_config(path='configs/stream_mc/sci_v3_submission_r4/primary_operating_point.yaml'):
    path = _resolve_cfg_path(path)
    cfg = yaml.safe_load(path.read_text())
    base_cfg = yaml.safe_load(_resolve_cfg_path(cfg['source_config']).read_text())
    if 'source_config' in base_cfg:
        root_base = yaml.safe_load(_resolve_cfg_path(base_cfg['source_config']).read_text())
        for k in ('level1', 'level2', 'bounded_graph', 'method_identity'):
            if k in root_base and k not in base_cfg:
                base_cfg[k] = root_base[k]
    cfg['base'] = base_cfg
    cfg['selection'] = yaml.safe_load(_resolve_cfg_path(cfg['selection_config']).read_text())['calibration']
    return cfg


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True))


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def load_models(cfg, seed, device):
    path = Path(cfg['checkpoint_root']) / f'seed{seed}'
    models = []
    for name, model_type, config_type in (
        ('level1', Level1Model, Level1ModelConfig),
        ('level2', Level2Model, Level2ModelConfig)):
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


def infer_local(model, graphs, T, device, batch_size=256):
    scores, embeddings, variances = [], [], []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        score, emb, var = local_predict(model, batch.to(device), T)
        scores.append(score); embeddings.append(emb); variances.append(var)
    return np.concatenate(scores), np.concatenate(embeddings), np.concatenate(variances)


class TimestampedFrozenRelations:
    """Cutoff-safe Level-2 relational reasoning: target i only sees train neighbors with t_ref <= t_target."""
    def __init__(self, embeddings, scores, timestamps, chains=None, contract_ids=None, k=8):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.scores = np.asarray(scores, dtype=np.float32)
        self.timestamps = np.asarray(timestamps, dtype=np.int64)
        self.chains = np.asarray(chains) if chains is not None else None
        self.contract_ids = np.asarray(contract_ids) if contract_ids is not None else None
        self.k = int(k)

    def graphs(self, target_embeddings, target_scores, target_timestamps,
               target_chains=None, target_contract_ids=None, source_chains=None):
        target_embeddings = np.asarray(target_embeddings, dtype=np.float32)
        target_scores = np.asarray(target_scores, dtype=np.float32)
        target_timestamps = np.asarray(target_timestamps, dtype=np.int64)
        
        output_graphs = []
        audit_records = []
        
        for i in range(len(target_scores)):
            t_i = target_timestamps[i]
            # Hard invariant: t_reference <= t_target
            mask = self.timestamps <= t_i
            if source_chains is not None:
                mask = mask & np.isin(self.chains, source_chains)
            
            n_eligible = int(mask.sum())
            assert n_eligible >= 1, f"Zero eligible historical references for target cutoff {t_i}"
            k_i = min(self.k, n_eligible)
            
            cand_emb = self.embeddings[mask]
            cand_scores = self.scores[mask]
            cand_times = self.timestamps[mask]
            
            # Efficient exact nearest neighbors from eligible candidates
            diffs = cand_emb - target_embeddings[i]
            dists = np.einsum('ij,ij->i', diffs, diffs)
            nn_local = np.argpartition(dists, k_i - 1)[:k_i]
            nn_local = nn_local[np.argsort(dists[nn_local])]
            
            sel_times = cand_times[nn_local]
            violations = int((sel_times > t_i).sum())
            assert violations == 0, f"Temporal violation: selected future reference for target {t_i}"
            
            audit = {
                'target_cutoff': int(t_i),
                'eligible_reference_count': n_eligible,
                'selected_neighbor_count': k_i,
                'max_selected_reference_cutoff': int(sel_times.max()),
                'temporal_violation_count': 0
            }
            if target_contract_ids is not None:
                audit['target_contract_id'] = str(target_contract_ids[i])
            if target_chains is not None:
                audit['target_chain'] = str(target_chains[i])
            audit_records.append(audit)
            
            # Star graph: k_i references + 1 target center at index k_i
            features = np.column_stack((cand_emb[nn_local], cand_scores[nn_local]))
            features = np.concatenate((features, np.r_[target_embeddings[i], target_scores[i]][None]), axis=0).astype('float32')
            refs = np.arange(k_i)
            edge_index = np.stack((np.r_[refs, np.full(k_i, k_i)], np.r_[np.full(k_i, k_i), refs]))
            output_graphs.append(Data(x=torch.from_numpy(features), edge_index=torch.from_numpy(edge_index).long()))
            
        return output_graphs, audit_records

    @torch.inference_mode()
    def predict(self, model, target_embeddings, target_scores, target_timestamps, device,
                target_chains=None, target_contract_ids=None, source_chains=None, batch_size=128):
        values = []
        all_audits = []
        for start in range(0, len(target_scores), batch_size):
            end = start + batch_size
            tch = target_chains[start:end] if target_chains is not None else None
            tcid = target_contract_ids[start:end] if target_contract_ids is not None else None
            graphs, audits = self.graphs(
                target_embeddings[start:end], target_scores[start:end], target_timestamps[start:end],
                target_chains=tch, target_contract_ids=tcid, source_chains=source_chains
            )
            batch = Batch.from_data_list(graphs).to(device)
            model.eval()
            values.append(model(batch).score.reshape(-1)[batch.ptr[1:] - 1].cpu().numpy())
            all_audits.extend(audits)
        return np.concatenate(values), all_audits


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
    budgets = cfg.get('risk', {}).get('budgets', [0.1, 0.2, 0.3, 0.42])
    weights = np.arange(0, 1.05, 0.1)
    for budget in budgets:
        cutoff = float(np.quantile(np.abs(fast-ft), budget, method='higher'))
        route = np.abs(fast-ft) <= cutoff
        for weight in weights:
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


def apply_policy(fast_raw, deep_raw, selection, policy='primary'):
    fast_cal = calibrate(fast_raw, selection['fast_map'])
    if policy == 'direct_only':
        return fast_cal, np.zeros(len(fast_raw), dtype=bool), (fast_cal >= selection['fast_threshold']).astype(int)
    if policy == 'full_deep':
        deep_cal = calibrate(deep_raw, selection['deep_map'])
        w = selection['fast_weight']
        full = expit(w * logits(fast_cal) + (1 - w) * logits(deep_cal))
        return full, np.ones(len(fast_raw), dtype=bool), (full >= selection['full_threshold']).astype(int)
    # Primary policy: selective escalation on validation margin
    route = np.abs(fast_cal - selection['fast_threshold']) <= selection['route_cutoff']
    deep_cal = np.where(route, calibrate(deep_raw, selection['deep_map']), fast_cal)
    w = selection['fast_weight']
    final_score = np.where(route, expit(w * logits(fast_cal) + (1 - w) * logits(deep_cal)), fast_cal)
    return final_score, route, (final_score >= selection['final_threshold']).astype(int)


def metrics(labels, scores, preds):
    labels, scores, preds = (np.asarray(x) for x in (labels, scores, preds))
    pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(labels, preds, labels=[0,1]).ravel())
    base = {
        'N': len(labels), 'N_positive': pos, 'N_negative': neg,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'f1': float(f1_score(labels, preds, zero_division=0)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'fraud_recall': float(recall_score(labels, preds, zero_division=0)),
        'fnr': float(fn / pos) if pos > 0 else 0.0,
        'mcc': float(matthews_corrcoef(labels, preds)),
        'pr_auc': float(average_precision_score(labels, scores)) if pos > 0 else float('nan'),
        'roc_auc': float(roc_auc_score(labels, scores)) if pos > 0 and neg > 0 else float('nan'),
        'balanced_accuracy': float(balanced_accuracy_score(labels, preds)) if pos > 0 and neg > 0 else float('nan'),
        'undefined_reason': '' if (pos > 0 and neg > 0) else 'single-class support'
    }
    return base
