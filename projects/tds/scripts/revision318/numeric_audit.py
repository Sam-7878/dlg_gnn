"""Read-only R317 evidence audit. Writes only R318 reports; never fits a detector."""
import hashlib
import importlib.util
import json
import re
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
OUT = PROJECT / 'results/318'
DATA = PROJECT / 'results/317'
ARCHIVE = PROJECT / 'publication/317/TDS_317_numeric_release_candidate.zip'
SEALED_SHA = 'a4f05674eef4d05913d897d3caa8e085026e7939e0d1d2c82c171d95a672e218'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def main():
    assert sha(ARCHIVE) == SEALED_SHA
    preserved = []
    with tempfile.TemporaryDirectory(prefix='tds318_numeric_') as tmp:
        cold = Path(tmp)
        with zipfile.ZipFile(ARCHIVE) as z:
            for n in z.namelist():
                assert not Path(n).is_absolute() and '..' not in Path(n).parts
            z.extractall(cold)
        spec = importlib.util.spec_from_file_location('sealed317_replay', cold / 'replay.py')
        replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(replay)
        result = replay.run(cold)
        result.update(archive_sha256=SEALED_SHA, extracted_to_fresh_directory=True,
                      anonymous_public_download_verified=False, full_retraining_performed=False)
        write('numeric_cold_replay.json', result)
        manifest = json.loads((cold / 'manifest.json').read_text())
        for e in manifest['files']:
            # All archived numeric inputs, predictions, tables and figures still match locally.
            local = DATA / e['path']
            if local.is_file():
                assert sha(local) == e['sha256'], e['path']
                preserved.append(e)

    summary = pd.read_csv(DATA / 'model_summary.csv', float_precision='round_trip')
    registry = pd.read_csv(DATA / 'run_registry.csv')
    cached = {}
    def scores(panel, arm, seed):
        key = (panel, arm, int(seed))
        if key not in cached:
            cached[key] = pd.read_csv(DATA / f'predictions/{panel}_{arm}_seed{int(seed)}_test.csv', float_precision='round_trip')
        return cached[key]

    # Independent sklearn aggregation, independent of the sealed replay's aggregation.
    key_rows = []
    for row in summary.itertuples():
        vals = []
        for seed in (7, 17, 27, 37, 47):
            f = scores(row.panel, row.arm, seed)
            vals.append((average_precision_score(f.label, f.score), roc_auc_score(f.label, f.score)))
        values = np.asarray(vals)
        for j, name in enumerate(['auc_pr', 'auc_roc']):
            assert abs(values[:, j].mean() - getattr(row, name + '_mean')) < 2e-12
            assert abs(values[:, j].std(ddof=1) - getattr(row, name + '_std')) < 2e-12
        key_rows.append(dict(panel=row.panel, arm=row.arm, ap_mean=values[:, 0].mean(),
                             ap_seed_sd=values[:, 0].std(ddof=1), roc_auc_mean=values[:, 1].mean(),
                             n=len(f), positive=int(f.label.sum())))
    pd.DataFrame(key_rows).to_csv(OUT / 'independent_model_metrics.csv', index=False)

    def operating(y, p, k):
        order = np.argsort(-p, kind='stable')
        cutoff = p[order[k-1]]
        above, tied = p > cutoff, p == cutoff
        slots = k-int(above.sum())
        nt, pt, pa = int(tied.sum()), int(y[tied].sum()), int(y[above].sum())
        captured = int(y[order[:k]].sum())
        return dict(captured=captured, precision=captured/k, recall=captured/y.sum(),
                    boundary_tie_size=nt, tie_expected_captured=pa+slots*pt/nt,
                    tie_min_captured=pa+max(0, slots-(nt-pt)), tie_max_captured=pa+min(slots, pt))
    operational = pd.read_csv(DATA / 'operational_metrics.csv', float_precision='round_trip')
    for r in operational.itertuples():
        f = (pd.read_csv(DATA / 'predictions/chain_prior_test.csv', float_precision='round_trip')
             if r.panel == 'control' else scores(r.panel, r.arm, r.seed))
        actual = operating(f.label.to_numpy(), f.score.to_numpy(), r.budget)
        for k, v in actual.items():
            assert abs(v-getattr(r, k)) < 2e-12, (r.panel, r.arm, k)

    chain = pd.read_csv(DATA / 'chain_stratified_ap.csv', float_precision='round_trip')
    pop = pd.read_csv(DATA / 'original_population.csv')
    pop = pop[pop.split == 'test'].reset_index(drop=True)
    for r in chain.itertuples():
        mask = pop.chain.to_numpy() == r.chain
        f = scores(r.panel, r.arm, 7)
        assert mask.sum() == r.n and int(f.label.to_numpy()[mask].sum()) == r.positive
        vals = [average_precision_score(scores(r.panel, r.arm, s).label.to_numpy()[mask],
                                         scores(r.panel, r.arm, s).score.to_numpy()[mask]) for s in (7,17,27,37,47)]
        assert abs(np.mean(vals)-r.ap_mean) < 2e-12
        if r.positive < 2:
            assert r.status == 'INFERENTIAL_SUPPORT_INSUFFICIENT' and pd.isna(r.ci95_low)

    # Check selection sensitivity from preserved lineage, using its original tie keys privately.
    lineage = pd.read_csv(DATA / 'sample_time_lineage.csv')
    private = pd.read_csv(PROJECT / 'private/317/sample_time_lineage.csv')
    joined = lineage.merge(private[['row_id', 'event_id', 'contract']], left_on='sample_id', right_on='row_id', validate='one_to_one')
    assert len(joined) == 24316 and joined.contract.notna().all()
    sorting = pd.read_csv(DATA / 'first_last_ordering_diagnostics.csv')
    for order in ['first_valid_tx_time', 'last_valid_tx_time']:
        f = joined.sort_values([order, 'event_id'], kind='stable').copy()
        f['split'] = np.repeat(['train', 'validation', 'test'], [17021, 3647, 3648])
        for split, g in f.groupby('split'):
            r = sorting[(sorting.ordering == order) & (sorting.split == split) & (sorting.chain == 'all')].iloc[0]
            assert len(g) == r.n and int(g.label.sum()) == r.positive
            assert abs(g.label.mean()-r.prevalence) < 1e-14
    medians = lineage.groupby('label').observed_lifetime_seconds.median()/86400
    assert abs(medians[1]-.670625) < 1e-9 and abs(medians[0]-608.4473611111111) < 1e-9
    assert int(lineage.raw_rows.sum()) == 268282924
    horizon = pd.read_csv(DATA / 'horizon7_population.csv')
    support = horizon.groupby('split').label.agg(['size','sum'])
    assert (int(support.loc['test','size']), int(support.loc['test','sum'])) == (3627, 2037)
    stats = pd.read_csv(DATA / 'statistics_and_holm.csv')
    assert sorted(stats.groupby('family').size().tolist()) == [6, 7]
    block = pd.read_csv(DATA / 'time_block_sensitivity.csv').iloc[0]
    assert block.n_blocks == 19 and block.ci95_low < 0 < block.ci95_high
    # Scan actual evidence bytes and record aggregate counts, never publish the mapping.
    with zipfile.ZipFile(ARCHIVE) as z:
        address_files = [n for n in z.namelist() if n.endswith(('.csv','.json')) and re.search(rb'0x[0-9a-fA-F]{40}(?![0-9a-fA-F])', z.read(n))]
        linkage_files = [n for n in z.namelist() if n.endswith('.csv') and b'sample_id' in z.read(n).split(b'\n')[0]]
        forbidden = [n for n in z.namelist() if n.endswith(('.pt','.pth','.bib','.cls')) or '/paper/' in n or '/author_records/' in n]
    assert not address_files and not forbidden
    write('identifier_linkage_audit.json', dict(direct_address_files=address_files,
          sample_id_bearing_csv_count=len(linkage_files), private_mapping_join_count=len(joined),
          sample_ids_pseudonymous_not_anonymous=True, upstream_reconstruction_linkage_possible=True,
          formal_privacy_guarantee=False, redistribution_authorized=False,
          forbidden_payloads=forbidden, mapping_disclosed=False))
    write('numeric_preservation.json', dict(status='PASS', preserved_files=preserved,
          arrays=110, validation_only_calibrators=95, model_groups_checked=len(summary),
          topk_rows_checked=len(operational), chain_rows_checked=len(chain),
          selection_support_and_lifetime_checked=True, transfer_count_from_preserved_lineage=268282924,
          raw_archives_rescanned=False, checkpoint_retraining=False,
          original_numeric_archive_sha256=SEALED_SHA))
    print(json.dumps(dict(status='PASS', arrays=110, calibrators=95, groups=len(summary),
                         topk_rows=len(operational), chain_rows=len(chain), preserved_files=len(preserved))))


if __name__ == '__main__':
    main()
