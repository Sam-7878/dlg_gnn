"""Recompute ORIGINAL trace claims; cheap new state probe has a separate identity."""
from __future__ import annotations
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
import tarfile
from r02_common import ROOT,R01,R02,VERIFY,RUN,read,write,sha256,csv_write,original_unchanged
from r01_common import config
import numpy as np
import pandas as pd
from gog_fraud.streaming.selective import binary_metrics


def main():
    VERIFY.mkdir(parents=True,exist_ok=True);cfg=config();rows=[];source_matrix=[]
    original_unchanged()
    expected_events=pd.read_parquet(ROOT/cfg['raw_events']).sort_values(['event_time','block_number','transaction_index','sample_id'],kind='stable').reset_index(drop=True)
    expected_ids=np.array([hashlib.sha256(hashlib.sha256(i.encode()).hexdigest().encode()).hexdigest() for i in expected_events.sample_id])
    for completed in sorted((R01/'runtime').glob('*/complete.json')):
        record=read(completed);trace_path=completed.parent/'event_trace.csv';trace=pd.read_csv(trace_path,low_memory=False);measured=trace[trace.measured]
        n=len(trace);assert np.array_equal(trace.sequence_id,np.arange(n)) and np.array_equal(trace.event_id_hash,expected_ids[:n])
        assert np.array_equal(trace.event_time,expected_events.event_time.to_numpy()[:n]) and np.array_equal(trace.label,expected_events.label.to_numpy()[:n])
        assert n==record['N_input'] and len(measured)==record['N_measured'] and int((~trace.measured).sum())==record['warmup_events']
        result={'population':record['population'],'policy':record['policy'],'repeat_id':record['repeat_id'],'N_input':n,'N_measured':len(measured),
            'deep_fraction':float(measured.escalated.mean()),'route_requested_fraction':float(measured.route_requested.mean()),
            'p50_ms':float(measured.end_to_end_ms.quantile(.5)),'p95_ms':float(measured.end_to_end_ms.quantile(.95)),'p99_ms':float(measured.end_to_end_ms.quantile(.99)),
            'throughput_events_per_second':len(measured)/record['wall_seconds'],'trace_sha256':sha256(trace_path),
            'throughput_scope':'recomputed from observed N and recorded total wall duration; per-event scoring timings exclude logging/checkpoint; no independent total start/end clock persisted'}
        for key in ('deep_fraction','p50_ms','p95_ms','p99_ms','throughput_events_per_second'):assert np.isclose(result[key],record[key],atol=1e-8,rtol=0),(completed.parent.name,key)
        assert int((~measured.accepted).sum())==record['rejected_events']
        assert (trace.selected_reference_max_cutoff.fillna(-1)<=trace.snapshot_cutoff).all()
        valid=measured.final_score.notna();metrics=binary_metrics(measured.loc[valid,'label'].to_numpy(),measured.loc[valid,'final_score'].to_numpy(),measured.loc[valid,'final_label'].to_numpy())
        from r02_numeric import same_metrics
        same_metrics(pd.Series(record['metrics']),metrics);result.update({k:metrics[k] for k in ('ap','f1','recall','fp_per_1000')})
        system=pd.read_csv(completed.parent/'system_trace.csv')
        assert trace.resident_contracts.max()<=5000 and trace.cache_entries.max()<=10000 and (trace.queue_length<=512).all()
        assert np.array_equal(trace.set_index('sequence_id').loc[system.sequence_id,'resident_contracts'].to_numpy(),system.resident_contracts.to_numpy())
        rows.append(result)
        source_matrix.append({'artifact':str(trace_path.relative_to(ROOT)),'original_source_version':'runtime_sources_v1.tar','measurement_id':record['run_id']+'/'+completed.parent.name,
            'verification':'raw identity/order/count/latency quantiles/metrics recomputed','amended_source_not_retroactively_timed':True})
    assert len(rows)==18 and sum(r['population']=='long' for r in rows)==3
    csv_write(VERIFY/'runtime_recomputed.csv',rows)
    write(VERIFY/'event_population_audit.json',{'verification_id':RUN,'N_prefix_runs':15,'N_long_runs':3,'per_long_events':100000,'event_order_and_provider_derived_labels_match':True,
        'raw_input_sha256':sha256(ROOT/cfg['raw_events']),'populations_separate':True,'labels':'inherited contract labels, not transaction fraud truth','long_repeats_per_policy':1,
        'throughput_limitation':'N/recorded wall duration; raw per-event trace alone cannot independently recover logging/checkpoint/loop wall overhead'})
    # Offline selected score/cost identities and all 540 trace sums.
    runtime=pd.read_csv(R01/'runtime/offline_v2/budget_runtime.csv');batches=pd.read_csv(R01/'runtime/offline_v2/offline_batch_trace.csv')
    keys=['seed','family','budget','random_repeat','runtime_repeat'];groups=batches.groupby(keys)
    assert len(runtime)==540 and len(groups)==540
    for r in runtime.itertuples():
        group=groups.get_group(tuple(getattr(r,k) for k in keys))
        assert int(group.N_batch.sum())==r.N and np.isclose(group.N_deep.sum()/r.N,r.deep_fraction,atol=1e-12)
        for field in ('local_ms','retrieval_ms','relational_ms','wall_ms'):assert np.isclose(group[field].sum(),getattr(r,field),atol=1e-8,rtol=0)
        assert np.isclose(group.wall_ms.sum()/r.N,r.latency_ms_per_contract,atol=1e-10)
        assert r.measurement_run_id=='selectivestream_r01_20261011_offline_v2' and bool(r.deterministic_algorithms) and r.torch_threads==1
    source_matrix.append({'artifact':'runtime/offline_v2','original_source_version':'runtime_sources_v2.tar','measurement_id':'selectivestream_r01_20261011_offline_v2','verification':'540 batch sums, N/deep/latency identities','amended_source_not_retroactively_timed':True})
    from r01_train import apply_fitted
    from sklearn.metrics import average_precision_score
    parity=[]
    for directory in sorted((R01/'models').glob('pooled_snapshot_GIN_seed*')):
        fit=read(directory/'policy_fit.json');frame=pd.read_parquet(directory/'test_predictions.parquet');seed=int(directory.name.split('seed')[1])
        with np.load(directory/'test_features.npz',allow_pickle=False) as features:benefit=features['benefit']
        for policy in fit['policies']:
            name=f"seed{seed}_{policy['family']}_q{policy['budget']}_random{policy['random_repeat']}.npz"
            path=R01/'runtime/offline_v2/budget_predictions'/name
            expected_score,expected_label,expected_route=apply_fitted(frame.raw_local_probability.to_numpy(),frame.counterfactual_full_raw_probability.to_numpy(),benefit,fit,policy=policy,eligible=frame.reference_count.to_numpy()>0)
            with np.load(path,allow_pickle=False) as actual:
                difference=float(np.max(abs(expected_score-actual['probability'])))
                assert difference<=1e-6 and np.array_equal(expected_label,actual['prediction']) and np.array_equal(expected_route,actual['escalated'])
                assert np.array_equal(frame.label,actual['label']) and np.array_equal(actual['sample_key'],[hashlib.sha256(i.encode()).hexdigest() for i in frame.sample_id])
                ap_actual=float(average_precision_score(actual['label'],actual['probability']));ap_canonical=float(average_precision_score(frame.label,expected_score))
                summary=runtime[(runtime.seed==seed)&(runtime.family==policy['family'])&(runtime.budget==policy['budget'])&(runtime.random_repeat==policy['random_repeat'])]
                assert len(summary)==3 and np.allclose(summary.ap,ap_actual,atol=1e-10,rtol=0)
                assert np.allclose(summary.deep_fraction,actual['escalated'].mean(),atol=1e-12)
            parity.append({'cell':name,'N':len(frame),'score_max_abs_diff':difference,'label_disagreements':0,'route_disagreements':0,'actual_selected_ap':ap_actual,'canonical_composed_ap':ap_canonical,
                'ap_float_tie_delta':ap_actual-ap_canonical,'actual_raw_sha256':sha256(path),'policy_hash':policy['policy_hash'],'source_version':'offline_v2','verification_id':RUN})
    assert len(parity)==180;csv_write(VERIFY/'selected_execution_parity.csv',parity)
    # Recompute cap/churn evidence from the original sampled pressure trace.
    state=read(R01/'stress/bounded_state_audit.json');pressure=pd.read_csv(R01/'stress/state_pressure_trace.csv')
    total=cfg['stress']['unique_contracts']+cfg['stress']['churn_events']
    expected_sample_sequence=sorted(set(range(0,total,500))|{cfg['stress']['unique_contracts']-1,total-1})
    assert np.array_equal(pressure.sequence_id,expected_sample_sequence) and int(pressure.resident_contracts.max())==state['resident_peak']==5000
    assert int(pressure.sequence_id.iloc[-1])==74999 and int(pressure.capacity_evictions.iloc[-1])==state['capacity_evictions']==70000
    assert pressure.resident_contracts.le(5000).all() and pressure.capacity_evictions.is_monotonic_increasing
    probe=R02/'new_measurements/state_probe';probe.mkdir(parents=True,exist_ok=True)
    from r01_stress import state_stress
    state_stress(cfg,probe)
    new=read(probe/'bounded_state_audit.json');invariant_fields=['N_unique','N_additional_churn','resident_peak','capacity_evictions','edge_cap','node_cap','ttl_expiry_count','cache_entries','cache_evictions','queue_high_water','queue_retries','queue_processed_events','duplicate_window_detected','late_quarantine']
    assert all(new[k]==state[k] for k in invariant_fields)
    write(VERIFY/'state_boundary_verification.json',{'verification_id':RUN,'original_report_sha256':sha256(R01/'stress/bounded_state_audit.json'),'original_pressure_trace_sha256':sha256(R01/'stress/state_pressure_trace.csv'),
        'original_trace_recomputed':'resident peak5000,75000-event sequence,70000 capacity evictions',
        'new_probe_measurement_id':'selectivestream_r02_20261012_state_probe','new_probe_report_sha256':sha256(probe/'bounded_state_audit.json'),
        'new_probe_reason':'original sampled pressure trace does not record every TTL/cache/queue action; reexecute cheap original state fixture, not expensive100k predictive replay',
        'new_probe_invariants_equal_original':invariant_fields,'new_times_rss_not_old_measurements':True,'churn_scope':'25000 initial distinct then50000 additional distinct contracts causing sustained eviction; not50000 repeated accesses to the first25000 keys',
        'memory_scope':'logical containers/payload, serialized state, immutable index/cache and RSS/GPU/source reader are distinct; not a lifetime whole-process bound'})
    restart=read(R01/'stress/restart_audit.json');diff=pd.read_csv(R01/'stress/checkpoint_diff.csv')
    assert len(diff)==restart['N_compared']==109 and np.isclose(diff.score_abs_diff.max(),restart['score_max_abs_diff'],atol=1e-15)
    assert not diff.label_disagreement.any() and diff.state_version_equal.all() and diff.accepted.all()
    checkpoints=[]
    for phase,cursor,code in [('before',19,91),('during',19,92),('after',30,93)]:
        fixture=next(r for r in restart['failures'] if r['phase']==phase);assert fixture['exit_code']==fixture['expected_exit']==code
        part=diff[diff.phase==phase];assert np.array_equal(part.sequence_id,np.arange(cursor+1,60)) and (part.recovery_cursor==cursor).all()
        path=R01/'stress'/f'{phase}.checkpoint'
        with path.open('rb') as handle:payload=pickle.load(handle)
        assert payload['cursor']==cursor and payload['identity']==restart['model_policy_reference_identity'] and 'rng' in payload and 'router_rng' in payload
        assert payload['state'].invariants()
        checkpoints.append({'phase':phase,'cursor':cursor,'actual_recorded_exit':code,'serialized_checkpoint_sha256':sha256(path),'N_replayed_suffix':len(part)})
    write(VERIFY/'restart_failure_model_verification.json',{'verification_id':RUN,'original_restart_report_sha256':sha256(R01/'stress/restart_audit.json'),'original_diff_sha256':sha256(R01/'stress/checkpoint_diff.csv'),
        'N_raw_pairs_checked':109,'checkpoints':checkpoints,'scope':'original recorded local os._exit results, exact suffix/state/identity/RNG presence and score guard; no new crash invocation, no distributed/disk-loss certification'})
    csv_write(VERIFY/'source_measurement_matrix.csv',source_matrix)
    original_unchanged();print('R02 ORIGINAL SYSTEM TRACE VERIFICATION PASS',len(rows),flush=True)


if __name__=='__main__':main()
