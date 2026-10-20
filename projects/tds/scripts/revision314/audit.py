"""Produce data-contract, loss, state, calibration and statistical audits."""
from __future__ import annotations

import itertools
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from common import FROZEN, GRAPH, OUT, ROOT, RAW_ROOT, load, sha, write_json


def utc(t):
    return datetime.fromtimestamp(int(t), timezone.utc).isoformat()


def tie_row(frame, scope):
    counts = frame.timestamp.value_counts()
    tied = counts[counts > 1]
    return dict(scope=scope, n=len(frame), distinct_timestamps=len(counts),
                tied_timestamps=len(tied), samples_in_ties=int(tied.sum()),
                fraction_in_ties=float(tied.sum()/len(frame)),
                max_samples_per_timestamp=int(counts.max()))


def main():
    payload, frame, graphs = load()
    out = OUT/'data'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    groups = [(s, f) for s, f in frame.groupby('split', sort=False)]
    test = frame[frame.split == 'test']
    groups.extend((f'test_bin_{i+1}', test.iloc[ix]) for i, ix in enumerate(np.array_split(np.arange(len(test)), 6)))
    for scope, f in groups:
        for chain in ['all', 'ethereum', 'bsc', 'polygon']:
            d = f if chain == 'all' else f[f.chain == chain]
            rows.append(dict(scope=scope, chain=chain, n=len(d),
                             fraction=len(d)/len(f), n_positive=int(d.label.sum()),
                             prevalence=float(d.label.mean()) if len(d) else None,
                             start_timestamp=int(d.timestamp.min()) if len(d) else None,
                             end_timestamp=int(d.timestamp.max()) if len(d) else None,
                             start_utc=utc(d.timestamp.min()) if len(d) else None,
                             end_utc=utc(d.timestamp.max()) if len(d) else None,
                             duration_seconds=int(d.timestamp.max()-d.timestamp.min()) if len(d) else None))
    pd.DataFrame(rows).to_csv(out/'split_composition.csv', index=False)
    ties = [tie_row(frame, 'all')]
    ties.extend(tie_row(f, c) for c, f in frame.groupby('chain'))
    pd.DataFrame(ties).to_csv(out/'timestamp_tie_audit.csv', index=False)
    overlap = []
    for a, b in itertools.combinations(['train', 'validation', 'test'], 2):
        fa, fb = frame[frame.split == a], frame[frame.split == b]
        for namespaced in [True, False]:
            ka = set(zip(fa.chain, fa.contract)) if namespaced else set(fa.contract)
            kb = set(zip(fb.chain, fb.contract)) if namespaced else set(fb.contract)
            overlap.append(dict(left=a, right=b, entity='chain:contract' if namespaced else 'contract_hex_cross_chain',
                                n_left=len(ka), n_right=len(kb), intersection=len(ka & kb),
                                new_entity_fraction_right=len(kb-ka)/len(kb)))
    past=set(zip(frame[frame.split != 'test'].chain, frame[frame.split != 'test'].contract))
    pos=test[test.label == 1]
    overlap.append(dict(left='train+validation',right='test_positive',entity='chain:contract',
                        n_left=len(past),n_right=len(pos),intersection=sum((c,a) in past for c,a in zip(pos.chain,pos.contract)),
                        new_entity_fraction_right=sum((c,a) not in past for c,a in zip(pos.chain,pos.contract))/len(pos)))
    pd.DataFrame(overlap).to_csv(out/'entity_overlap.csv', index=False)
    label_path=RAW_ROOT/'labels.csv'
    labels=pd.read_csv(label_path)
    labels['Chain']=labels.Chain.str.lower();labels['Contract']=labels.Contract.str.lower()
    m=frame.merge(labels, left_on=['chain','contract'],right_on=['Chain','Contract'],validate='one_to_one')
    assert len(m)==len(frame) and np.array_equal(m.label, (m.Category==0).astype(int))
    write_json(OUT/'reports/data_contract.json',dict(
        graph_sha256=sha(GRAPH), n=len(frame), sample_unit='one token/contract-local graph observed at its final recorded transaction timestamp',
        label_unit='token/contract', label_file=str(label_path),label_file_sha256=sha(label_path),
        label_field='Category',binary_mapping='Category == 0 -> 1; Category != 0 -> 0',
        label_join_verified=len(m),contract_entities_unique=not frame.duplicated(['chain','contract']).any(),
        cutoff_semantics='last valid transaction timestamp, not label discovery time',
        feature_semantics='log1p degree on full contract history up to cutoff, sampled at nodes retained after the 128-edge cap',
        neighborhood_semantics='all contract endpoints then most recent 128 edges; no k-hop expansion parameter exists',
        split_order='timestamp then sample_id; deterministic ordering is not a claim about physical intra-block ordering',
        retained_graph_future_violation_count=int((frame.max_input_edge_timestamp > frame.timestamp).sum()),
        same_timestamp_cutoff_count=int((frame.max_input_edge_timestamp == frame.timestamp).sum()),
        limitation='Packed graph omits per-edge times and endpoint address mapping; strict-time sensitivity needs source ZIP reconstruction. Historical upstream symlinks are currently dangling.'))
    corrected=OUT/'supplement/statistical_comparisons_corrected.csv'
    stats=pd.read_csv(corrected if corrected.exists() else FROZEN/'statistical_comparisons.csv')
    p=stats.randomization_p_value_two_sided.to_numpy()
    order=np.argsort(p,kind='stable'); adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    stats['holm_adjusted_p']=adjusted
    stats['holm_reject_0_05']=adjusted<.05
    stats.to_csv(OUT/'supplement/holm_adjusted_statistics.csv',index=False)
    for src in ['experiments/round4/train_sci_main.py','experiments/round7/train_comparable_models.py',
                'experiments/round5/models.py','src/data_generation/build_gog_chronological_dataset.py',
                'experiments/round7/statistics.py']:
        data=(Path(__file__).parent/'original_source'/src.rsplit('/',1)[-1]).read_bytes()
        import hashlib
        write_json(OUT/'reports'/('source_'+src.rsplit('/',1)[-1]+'.json'),dict(commit='fbccefc',path=src,sha256=hashlib.sha256(data).hexdigest()))
    print('Data and Holm audit complete', flush=True)


if __name__ == '__main__':
    main()
