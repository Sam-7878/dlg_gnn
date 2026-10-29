"""Audit frozen predictions, checkpoints and statistics without training."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from paths import ROOT, INPUT, PRIMARY, RESULTS, REPORTS
from freeze import sha, write

SEEDS = (7, 17, 27, 37, 47)


def main():
    sealed = json.loads((RESULTS / 'input_hash_manifest.json').read_text())
    changes = []
    for e in sealed['files']:
        actual = sha(ROOT / e['path'])
        if actual != e['sha256']:
            assert e['path'] in ['projects/tds/scripts/revision314/common.py',
                                 'projects/tds/scripts/revision314/seal_packages.py'], e['path']
            changes.append(dict(path=e['path'], before=e['sha256'], after=actual,
                                reason='canonical path repair only'))
    primary = json.loads((PRIMARY / 'reproducibility_manifest.json').read_text())
    for e in primary['entries']:
        assert sha(PRIMARY / e['path']) == e['sha256'], e['path']
    checkpoints = json.loads((PRIMARY / 'checkpoints_manifest.json').read_text())
    for e in checkpoints['entries']:
        path = (ROOT / f'archive/tds/data/gog_scimain_v1_preserved_panel/checkpoints/seed{e["seed"]}.pt'
                if e['model_key'] == 'proposed' else PRIMARY / f'checkpoints/{e["model_key"]}/seed{e["seed"]}.pt')
        assert sha(path) == e['checkpoint_sha256'], path
    expected = pd.read_csv(PRIMARY / 'raw_predictions/proposed/seed7_test.csv')[['event_id', 'timestamp', 'label']]
    assert len(expected) == 3648 and expected.label.sum() == 107 and expected.event_id.is_unique
    rows = []
    sources = []
    for path in sorted((INPUT / 'runs').glob('*/seed*/test.csv')):
        m = json.loads((path.parent / 'metrics.json').read_text())
        assert sha(path) == m['test_predictions_sha256'], path
        assert sha(path.parent / 'checkpoint.pt') == m['checkpoint_sha256'], path
        assert m['test_access_count'] == 1 and not m['max_epoch_hit'], path
        f = pd.read_csv(path)
        assert f[['event_id', 'timestamp', 'label']].equals(expected), path
        ap = average_precision_score(f.label, f.p_mean)
        assert np.isclose(ap, m['auc_pr'], atol=1e-12, rtol=0), path
        rows.append(dict(arm=path.parent.parent.name, seed=int(path.parent.name[4:]),
                         n=len(f), positive=int(f.label.sum()), ap=ap, sha256=sha(path)))
        sources.append(path)
    assert len(rows) == 35
    f = pd.DataFrame(rows)
    summary = f.groupby('arm').ap.agg(['mean', 'std']).reset_index()
    original = pd.read_csv(INPUT / 'data/supplemental_summary.csv')
    for row in summary.itertuples():
        o = original.loc[original.arm == row.arm].iloc[0]
        assert np.isclose(row.mean, o.auc_pr_mean, atol=1e-12, rtol=0)
        assert np.isclose(row.std, o.auc_pr_std, atol=1e-12, rtol=0)
    RESULTS.mkdir(parents=True, exist_ok=True)
    f.to_csv(RESULTS / 'supplemental_ap_per_seed.csv', index=False)
    summary.to_csv(RESULTS / 'supplemental_ap_summary.csv', index=False)
    primary_rows = []
    panels = {}
    for kind in ['proposed', 'mc_dropout', 'temperature_scaled', 'tgn', 'tgat', 'fraudsage']:
        panels[kind] = []
        for seed in SEEDS:
            path = PRIMARY / f'raw_predictions/{kind}/seed{seed}_test.csv'
            p = pd.read_csv(path)
            assert p[['event_id', 'timestamp', 'label']].equals(expected), path
            assert np.isfinite(p.p_mean).all() and p.p_mean.between(0, 1).all()
            panels[kind].append(p.p_mean.to_numpy())
            primary_rows.append(dict(method=kind, seed=seed, n=len(p), positive=int(p.label.sum()),
                                     ap=average_precision_score(p.label, p.p_mean), sha256=sha(path)))
            sources.append(path)
    pd.DataFrame(primary_rows).to_csv(RESULTS / 'primary_ap_per_seed.csv', index=False)
    # Import pure numeric helpers; importing the training runtime is unnecessary.
    sys.path.insert(0, str(ROOT / 'projects/tds/scripts/revision314'))
    import historical_statistics as h
    def tied_ap(y, score):
        order = np.argsort(-score, kind='stable')
        y, score = np.asarray(y)[order], np.asarray(score)[order]
        ends = np.r_[np.flatnonzero(score[:-1] != score[1:]), len(score)-1]
        tp = np.cumsum(y)[ends]
        return float(np.sum(np.diff(np.r_[0, tp])*tp/(ends+1))/y.sum())
    h.average_precision = tied_ap
    def holm(p):
        p = np.asarray(p); order = np.argsort(p, kind='stable'); out = np.empty_like(p)
        out[order] = np.minimum(1, np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
        return out
    for name, size in [('statistical_comparisons_corrected', 7), ('control_statistical_comparisons', 6)]:
        stats = pd.read_csv(INPUT / f'supplement/{name}.csv')
        assert len(stats) == size
        assert np.allclose(holm(stats.randomization_p_value_two_sided), stats.holm_adjusted_p, atol=1e-14)
        for col in ['n_bootstrap', 'n_bootstrap_valid', 'n_class_stratified_bootstrap', 'n_randomization']:
            assert (stats[col] == 10000).all()
        counts = stats.randomization_p_value_two_sided.to_numpy()*10001-1
        assert np.allclose(counts, np.rint(counts), atol=1e-9)
    stats = pd.read_csv(INPUT / 'supplement/statistical_comparisons_corrected.csv')
    y = expected.label.to_numpy(int)
    deep = np.mean(panels['proposed'], axis=0)
    mc = np.mean(panels['mc_dropout'], axis=0)
    replay = h.paired_panel_comparison(y, [mc], [deep], n_resamples=10000, seed=20260916)
    for key, value in replay.items():
        assert np.isclose(value, stats.iloc[6][key], atol=1e-12, rtol=0), (key, value, stats.iloc[6][key])
    write(RESULTS / 'representative_statistics_replay.json', dict(
        comparison=stats.iloc[6]['comparison'], seed=20260916, **replay,
        exact_corrected_csv_match=True, scope='one original comparison; all three 10000-resample protocols'))
    write(RESULTS / 'prediction_sources.json', dict(files=[dict(
        path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in sources],
        aligned_identity_sha256=__import__('hashlib').sha256(expected.to_csv(index=False).encode()).hexdigest()))
    write(REPORTS / 'evidence_audit.json', dict(status='PASS', input_files_verified=len(sealed['files']),
        source_path_repairs=changes, primary_artifacts_verified=len(primary['entries']),
        primary_checkpoints_verified=len(checkpoints['entries']), supplemental_checkpoints_verified=35,
        supplemental_predictions_verified=35, primary_predictions_verified=30,
        test_samples=3648, test_positives=107, supplemental_ap_mean_sample_sd_match=True,
        original_family_size=7, exploratory_family_size=6, holm_recomputed=True,
        representative_full_resampling_replay=True, new_training_performed=False))
    print('PASS: frozen inputs, 20+35 checkpoints, 35 supplemental and 30 primary predictions; 10000-resample replay')


if __name__ == '__main__':
    main()
