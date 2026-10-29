"""Portable, CPU-only verification of a staged/public TDS 315 numeric bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tied_ap(y, score):
    order = np.argsort(-score, kind='stable')
    y, score = np.asarray(y)[order], np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
    tp = np.cumsum(y)[ends]
    return float(np.sum(np.diff(np.r_[0, tp])*tp/(ends+1))/y.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--statistics', action='store_true', help='Also replay one full 10000-replicate comparison')
    args = parser.parse_args()
    base = args.bundle.resolve()
    manifest = json.loads((base / 'manifest.json').read_text())
    for e in manifest['files']:
        p = (base / e['path']).resolve()
        assert p.is_relative_to(base), e['path']
        assert sha(p) == e['sha256'], e['path']
    lineage = json.loads((base / 'prediction_lineage.json').read_text())
    rows, panels, identity = [], {}, None
    for e in lineage['files']:
        f = pd.read_csv(base / e['path'])
        ids = f[['row_id', 'label']]
        if identity is None:
            identity = ids
        assert ids.equals(identity), e['path']
        assert len(f) == 3648 and f.label.sum() == 107
        assert f.p_mean.between(0, 1).all() and np.isfinite(f.p_mean).all()
        ap = average_precision_score(f.label, f.p_mean)
        assert np.isclose(ap, e['ap'], atol=1e-12, rtol=0), e['path']
        assert np.isclose(tied_ap(f.label.to_numpy(), f.p_mean.to_numpy()), ap, atol=1e-12, rtol=0)
        panels[(e['panel'], e['arm'], e['seed'])] = f.p_mean.to_numpy()
        rows.append(e)
    assert len(rows) == 65
    summary = pd.read_csv(base / 'data/supplemental_summary.csv')
    for o in summary.itertuples():
        values = [e['ap'] for e in rows if e['panel'] == 'supplemental' and e['arm'] == o.arm]
        assert len(values) == 5
        assert np.isclose(np.mean(values), o.auc_pr_mean, atol=1e-12, rtol=0)
        assert np.isclose(np.std(values, ddof=1), o.auc_pr_std, atol=1e-12, rtol=0)
    stats = pd.read_csv(base / 'statistics/statistical_comparisons_corrected.csv')
    if args.statistics:
        import importlib.util
        spec = importlib.util.spec_from_file_location('paired', base / 'scientific/historical_statistics.py')
        h = importlib.util.module_from_spec(spec); spec.loader.exec_module(h)
        h.average_precision = tied_ap
        seeds = [7, 17, 27, 37, 47]
        deep = np.mean([panels['primary', 'proposed', s] for s in seeds], axis=0)
        mc = np.mean([panels['primary', 'mc_dropout', s] for s in seeds], axis=0)
        out = h.paired_panel_comparison(identity.label.to_numpy(int), [mc], [deep], n_resamples=10000, seed=20260916)
        for key, value in out.items():
            assert np.isclose(value, stats.iloc[6][key], atol=1e-12, rtol=0), key
    print(json.dumps(dict(status='PASS', files_hashed=len(manifest['files']), prediction_files=65,
                          samples_per_file=3648, positives_per_file=107, supplemental_arms=7,
                          full_representative_statistics_replayed=args.statistics,
                          anonymous_access_tested=False)))


if __name__ == '__main__':
    main()
