"""Verify local per-run vector coverage, cryptocurrency tensor identities, DGraphFin population reconciliation, and supplementary inventories."""
from a24_inputs import load_input, tensor_hash
from a24_common import ROOT, LOCAL, PUB, REG, SUP, rows, sha, located
from collections import Counter
import json
import hashlib
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    registry = rows(REG)
    missing = [r for r in registry if r['raw_score_status'] == 'MISSING_NOT_STORED_BY_SOURCE_RUN']
    assert len(missing) == 310 and len({r['dataset'] for r in missing}) == 10
    assert sum(r['dataset'] == 'LANL-RedTeam' for r in missing) == 35
    source_dirs = set()
    for r in missing:
        sp = located(r['source_path'])
        assert sp.exists()
        source_dirs.add(sp.parent)
        rec = json.loads(sp.read_text(encoding='utf-8'))
        for k, v in rec.items():
            if 'score' in k.lower() and isinstance(v, str):
                assert not v.endswith(('.npy', '.npz'))
    for sd in source_dirs:
        assert len(list(sd.glob('*.np*'))) == 0
    missing_audit = dict(
        total_missing_runs=310,
        primary_missing_runs=275,
        external_lanl_missing_runs=35,
        verified_source_json_records=310,
        score_files_in_source_dirs=0,
        source_directories=[str(sd.relative_to(ROOT)).replace('\\', '/') for sd in sorted(source_dirs)],
    )

    archive = []
    for r in registry:
        if r['raw_score_status'] != 'ARCHIVED_A05_SCORE':
            continue
        p = located(r['raw_score_path'])
        raw = located(r['source_path'])
        a = json.loads(raw.read_text(encoding='utf-8'))
        assert r['metric_file_hash'] == sha(raw)
        score = np.load(p, allow_pickle=False)
        assert a['raw_score_sha256'] == r['raw_score_hash'] == array_sha(score)
        assert a['raw_score_file_sha256'] == sha(p)
        assert score.size == int(r['nodes']) and np.all(np.isfinite(score))
        assert a['run_id'] == r['run_id'] and np.isclose(float(r['pr_auc']), a['pr_auc'], rtol=0, atol=2e-16)
        archive.append(dict(dataset=r['dataset'], model=r['model'], seed=int(r['seed']),
                            score_sha256=a['raw_score_sha256'], file_sha256=sha(p)))
    assert sum(r['dataset'] == 'BitcoinOTC' for r in archive) == 35 and len(archive) == 40

    # Verify DGraphFin three-level population and prevalence arithmetic (A27-02)
    dg_graph, dg_masks, dg_id = load_input('DGraphFin')
    dg_y = dg_graph.y.numpy()
    dg_train = dg_masks[42]['train_mask']
    dg_val = dg_masks[42]['val_mask']
    dg_test = dg_masks[42]['test_mask']
    dg_eval = dg_val | dg_test
    assert dg_graph.num_nodes == 1225601 and int(dg_y.sum()) == 15509
    assert int(dg_train.sum()) == 857899 and int(dg_y[dg_train].sum()) == 10857
    assert int(dg_eval.sum()) == 367702 and int(dg_y[dg_eval].sum()) == 4652
    assert int(dg_val.sum()) == 183862 and int(dg_y[dg_val].sum()) == 2326
    assert int(dg_test.sum()) == 183840 and int(dg_y[dg_test].sum()) == 2326
    dgraphfin_reconciliation = dict(
        raw_upstream_nodes=3700550,
        raw_upstream_edges=4300999,
        model_input_N=1225601,
        model_input_E=746271,
        model_input_d=17,
        total_positives=15509,
        train_N=857899,
        train_positives=10857,
        whole_tensor_prevalence=float(15509 / 1225601),
        labeled_eval_pool_N=367702,
        eval_pool_positives=4652,
        val_N=183862,
        val_positives=2326,
        test_N=183840,
        test_positives=2326,
        labeled_pool_prevalence=float(4652 / 367702),
        test_prevalence=float(2326 / 183840),
        dense_float32_4N2_bytes_model_input=4 * (1225601 ** 2),
        dense_float32_4N2_tb_decimal=round(4 * (1225601 ** 2) / 1e12, 3),
        dense_float32_4N2_gb_labeled_pool=round(4 * (367702 ** 2) / 1e9, 1),
    )

    graph_prior = {r['dataset_id']: r for r in rows(ROOT / 'projects/benchmark/publication/a23/dataset_graph_identity_canonical.csv')}
    bindings = {r['run_id']: r for r in rows(PUB / 'score_input_binding_105.csv')}
    crypto = []
    for d in ['Ethereum', 'BSC', 'Polygon']:
        graph, masks, identity = load_input(d)
        rebuild = LOCAL / 'raw_rebuild' / d.lower() / 'observables.npz'
        with np.load(rebuild, allow_pickle=False) as z:
            np.testing.assert_array_equal(z['x'], graph.x.numpy())
            np.testing.assert_array_equal(z['edge_index'], graph.edge_index.numpy())
        edges = graph.edge_index.numpy()
        packed = edges[0] * graph.num_nodes + edges[1]
        assert len(np.unique(packed)) == edges.shape[1] and np.all(edges[0] != edges[1])
        np.testing.assert_array_equal(np.sort(packed), np.sort(edges[1] * graph.num_nodes + edges[0]))
        local_runs = ROOT / 'local_storage/benchmark/a08_data_repair/runs'
        matches = [r for r in registry if r['dataset'] == d]
        assert len(matches) == 35
        for r in matches:
            folder = local_runs / r['run_id']
            score_file = folder / 'scores.npz'
            meta = json.loads((folder / 'run_manifest.json').read_text(encoding='utf-8'))
            b = bindings[r['run_id']]
            assert identity['input_manifest_sha256'] == meta['input_manifest_hash'] == r['input_manifest_hash'] == b['input_manifest_hash']
            assert sha(score_file) == r['raw_score_hash'] == b['scores_hash']
            assert sha(folder / 'metrics.json') == r['metric_json_hash'] == b['metrics_hash']
            with np.load(score_file, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['labels'], graph.y.numpy())
                seed = int(r['seed'])
                for key in ['val_mask', 'test_mask']:
                    np.testing.assert_array_equal(z[key], masks[seed][key])
                test = z['test_mask']
                assert np.isclose(average_precision_score(z['labels'][test], z['scores'][test]), float(r['pr_auc']), atol=1e-12, rtol=0)
        prior = graph_prior[d]
        path = ROOT / identity['source_path']
        construction = json.loads((path.parent / 'input_manifest.json').read_text(encoding='utf-8'))
        crypto.append(dict(
            dataset=d, dataset_id=construction['dataset_id'], N=identity['N'],
            E=identity['E'], d=identity['F'], old_N=int(prior['model_input_N']),
            old_E=int(prior['model_input_E']), archived_rebuild_sha256=sha(rebuild),
            exact_primary_score_matches=len(matches),
            builder_code_hashes=construction['builder_code_hashes'],
            **{k: identity[k] for k in ('source_sha256', 'input_manifest_sha256', 'feature_hash', 'edge_hash', 'label_hash')}
        ))
        print(d, '35 primary APs and exact graph binding verified', flush=True)

    alerts = []
    expected = [r for r in rows(SUP / 'real_domain_alert_budget_per_seed.csv') if r['dataset'] == 'Elliptic']
    assert len(expected) == 10
    _, masks, identity = load_input('Elliptic')
    for r in expected:
        p = LOCAL / 'runs' / r['run_id'] / 'scores.npz'
        assert sha(p) == r['raw_score_sha256']
        with np.load(p, allow_pickle=False) as z:
            s, y, test = z['scores'], z['labels'], z['test_mask']
            assert np.all(s >= 0) and np.isfinite(s).all()
            assert tensor_hash(y) == identity['label_hash']
            np.testing.assert_array_equal(test, masks[int(r['seed'])]['test_mask'])
            ap = average_precision_score(y[test], s[test])
            roc = roc_auc_score(y[test], s[test])
            budget = int(np.ceil(.01 * test.sum()))
            top = np.lexsort((z['node_ids'][test], -s[test]))[:budget]
            precision = y[test][top].mean()
            lift = precision / y[test].mean()
            assert np.isclose(ap, float(r['pr_auc'])) and np.isclose(roc, float(r['roc_auc']))
            assert np.isclose(precision, float(r['P_at_1pct'])) and np.isclose(lift, float(r['lift_1pct']))
            alerts.append(dict(model=r['model'], seed=int(r['seed']), AP=float(ap), ROC=float(roc), P_at_1pct=float(precision), lift=float(lift)))

    supplementary = []
    sup_records = []
    for path in sorted((LOCAL / 'runs').glob('*/result.json')):
        r = json.loads(path.read_text(encoding='utf-8'))
        assert sha(path.parent / 'scores.npz') == r['raw_score_sha256']
        supplementary.append(r['raw_score_sha256'])
        sup_records.append(r)
    assert len(supplementary) == 113
    breakdown = []
    for (camp, ds), cnt in sorted(Counter((r['campaign'], r['dataset']) for r in sup_records).items()):
        models = sorted({r['model'] for r in sup_records if r['campaign'] == camp and r['dataset'] == ds})
        breakdown.append(dict(campaign=camp, dataset=ds, count=cnt, models=models))

    reddit_aug_path = ROOT / 'local_storage/benchmark/a27/reddit_dlg_aug_measured.json'
    reddit_aug = json.loads(reddit_aug_path.read_text(encoding='utf-8')) if reddit_aug_path.exists() else None

    print('40 earlier archived vectors, 310 missing verified on disk, 113 supplementary vectors broken down, DGraphFin reconciled', flush=True)
    Path(sys.argv[1]).write_text(json.dumps(dict(
        primary_registry_sha256=sha(REG),
        archived_A05=archive,
        missing_primary=275,
        missing_external=35,
        missing_runs_disk_audit=missing_audit,
        supplementary_verified=113,
        supplementary_breakdown=breakdown,
        dgraphfin_reconciliation=dgraphfin_reconciliation,
        reddit_dlg_aug_rtx3090_measurement=reddit_aug,
        crypto=crypto,
        elliptic_alerts=alerts,
    ), indent=2) + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
