"""Check score direction against the estimator APIs, saved arrays, labels, and feature-space dispersion across all 12 cells."""
from a24_run_studies import load_input, LOCAL, SUP, ROOT, rows, sha
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from pyod.models.ecod import ECOD
import json
import sys
from pathlib import Path
import numpy as np

def check_features():
    expected = rows(SUP / 'feature_only_per_seed.csv')
    per_seed = []
    cells_12 = []
    for name in dict.fromkeys(r['dataset'] for r in expected):
        data, masks, identity = load_input(name)
        x = data.x.numpy()
        labels = data.y.numpy()
        N, d = x.shape
        stds = x.std(axis=0)
        near_const = int(np.sum(stds < 1e-6))
        valid = stds >= 1e-8
        z_std = (x[:, valid] - x[:, valid].mean(axis=0)) / stds[valid]
        rms_z = np.linalg.norm(z_std, axis=1) / np.sqrt(valid.sum())
        max_abs_z = np.max(np.abs(z_std), axis=1)

        ecod = ECOD(contamination=.1, n_jobs=1).fit(x)
        for r in (r for r in expected if r['dataset'] == name):
            seed = int(r['seed'])
            path = LOCAL / 'runs' / r['run_id'] / 'scores.npz'
            assert sha(path) == r['raw_score_sha256']
            assert r['feature_hash'] == identity['feature_hash']
            if r['model'] == 'IForest':
                detector = IsolationForest(n_estimators=100, max_samples=256, max_features=1.,
                                           bootstrap=False, contamination='auto', random_state=seed, n_jobs=4).fit(x)
                native = detector.score_samples(x)
                oriented = -native
            else:
                oriented = ecod.decision_scores_
                assert np.all(oriented >= 0)
            with np.load(path) as z:
                assert np.array_equal(labels, z['labels'])
                assert np.array_equal(z['node_ids'], np.arange(len(labels)))
                for key, mask in masks[seed].items():
                    np.testing.assert_array_equal(mask, z[key])
                np.testing.assert_array_equal(oriented, z['scores'])
                test = z['test_mask']
                y, s = labels[test], oriented[test]
                ap, roc = average_precision_score(y, s), roc_auc_score(y, s)
                assert abs(ap - float(r['pr_auc'])) < 1e-12 and abs(roc - float(r['roc_auc'])) < 1e-12
                inverse_ap, inverse_roc = average_precision_score(y, -s), roc_auc_score(y, -s)
                assert np.isclose(roc + inverse_roc, 1.)
                eval_mask = (z['val_mask'] | z['test_mask']) if name == 'DGraphFin' else np.ones(N, dtype=bool)
                pos_mask = eval_mask & (labels == 1)
                neg_mask = eval_mask & (labels == 0)
                s_pos = float(np.mean(oriented[pos_mask]))
                s_neg = float(np.mean(oriented[neg_mask]))
                rms_pos = float(np.mean(rms_z[pos_mask]))
                rms_neg = float(np.mean(rms_z[neg_mask]))
                maxz_pos = float(np.mean(max_abs_z[pos_mask]))
                maxz_neg = float(np.mean(max_abs_z[neg_mask]))
            per_seed.append(dict(
                dataset=name, model=r['model'], seed=seed, score_sha256=sha(path),
                same_vector_as_api=True, AP=float(ap), ROC=float(roc),
                AP_negated=float(inverse_ap), ROC_negated=float(inverse_roc),
                score_pos_mean=s_pos, score_neg_mean=s_neg,
                rms_z_pos=rms_pos, rms_z_neg=rms_neg,
                max_abs_z_pos=maxz_pos, max_abs_z_neg=maxz_neg
            ))
        for model in ['IForest', 'ECOD']:
            sub = [row for row in per_seed if row['dataset'] == name and row['model'] == model]
            assert len(sub) == 5
            s_pos_m = float(np.mean([row['score_pos_mean'] for row in sub]))
            s_neg_m = float(np.mean([row['score_neg_mean'] for row in sub]))
            r_pos = sub[0]['rms_z_pos']
            r_neg = sub[0]['rms_z_neg']
            cells_12.append(dict(
                dataset=name, model=model, N=N, d=d, near_constant_dims=near_const,
                api_call_site='projects/benchmark/scripts/a24_run_studies.py:78-79',
                api_convention='-det.score_samples(x)' if model == 'IForest' else 'det.decision_scores_',
                rms_z_pos=r_pos, rms_z_neg=r_neg, rms_z_ratio=float(r_pos / r_neg),
                max_abs_z_pos=sub[0]['max_abs_z_pos'], max_abs_z_neg=sub[0]['max_abs_z_neg'],
                score_pos_mean=s_pos_m, score_neg_mean=s_neg_m,
                score_diff_pos_minus_neg=float(s_pos_m - s_neg_m),
                AP_mean=float(np.mean([row['AP'] for row in sub])),
                AP_sd=float(np.std([row['AP'] for row in sub], ddof=1)),
                ROC_mean=float(np.mean([row['ROC'] for row in sub])),
                ROC_sd=float(np.std([row['ROC'] for row in sub], ddof=1)),
                AP_negated_mean=float(np.mean([row['AP_negated'] for row in sub])),
                ROC_negated_mean=float(np.mean([row['ROC_negated'] for row in sub])),
            ))
        print(name, 'all ten native oriented scores, metrics, and feature-dispersion diagnostics reproduced', flush=True)
    Path(sys.argv[1]).write_text(json.dumps({'cells_12': cells_12, 'per_seed': per_seed}, indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    check_features()

