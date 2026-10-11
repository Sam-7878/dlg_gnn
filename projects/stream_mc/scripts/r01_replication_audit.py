"""Compare actual alias-input reruns with the original, not with a target table."""
from __future__ import annotations
import argparse
from r01_common import PROJECT, output, write, sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['retrain','replay','all'],default='all');args=parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    out=output();report={'scope':'actual same-host seed11 alias-input reconstruction/rerun; distinct identities, not independent external review or prospective source reconstruction','score_tolerance':1e-6}
    if args.mode in ('retrain','all'):
        replica=PROJECT/'results/public_input_reproduction/models';records=[]
        paths=sorted(replica.glob('*/complete.json'))
        assert len(paths)==5,'expected actual seed11 pooled GIN/GATv2 plus three source-only fits'
        for completed in paths:
            new=completed.parent;old=out/'models'/new.name
            for split in ('validation','test'):
                a=pd.read_parquet(old/f'{split}_predictions.parquet');b=pd.read_parquet(new/f'{split}_predictions.parquet')
                assert len(a)==len(b) and np.array_equal(a.label,b.label) and np.array_equal(a.chain_scope,b.chain_scope)
                assert np.array_equal(a.snapshot_cutoff,b.snapshot_cutoff)
                columns=['raw_local_probability','counterfactual_full_raw_probability','local_probability','selective_probability','full_probability','threshold_only_probability']
                score=max(float(np.max(np.abs(a[c].to_numpy()-b[c].to_numpy()))) for c in columns)
                changes=sum(int(np.sum(a[c].to_numpy()!=b[c].to_numpy())) for c in ['local_prediction','selective_prediction','full_prediction','threshold_only_prediction','escalated','reference_count','eligible_reference_count'])
                assert score<=1e-6 and changes==0,(new.name,split,score,changes)
                records.append({'model_id':new.name,'split':split,'N':len(a),'max_score_difference':score,'decision_route_reference_count_disagreements':changes,'original_prediction_sha256':sha256(old/f'{split}_predictions.parquet'),'replica_prediction_sha256':sha256(new/f'{split}_predictions.parquet')})
            for name in ('local.pt','relational.pt'):
                a=torch.load(old/name,map_location='cpu',weights_only=False)['state_dict'];b=torch.load(new/name,map_location='cpu',weights_only=False)['state_dict']
                assert a.keys()==b.keys() and all(torch.equal(a[k],b[k]) for k in a),'parameter tensor difference: '+new.name+'/'+name
        report['retraining']={'N_completed_models':5,'seeds':[11],'prediction_comparisons':records,'local_and_relational_parameter_tensors_bit_identical':True,'checkpoint_file_bytes_not_equal_expected':'new timing/run/alias metadata, not numerical weight tensors',
            'light_head_training_probe_sha256':sha256(PROJECT/'results/public_input_reproduction/training_cost_probe.json'),'original_light_head_training_wall_time_still_unrecorded':True}
    if args.mode in ('replay','all'):
        old=out/'runtime/prefix_margin_repeat0/event_trace.csv';new=PROJECT/'results/public_alias_replay/runtime/prefix_margin_repeat0/event_trace.csv'
        a=pd.read_csv(old);b=pd.read_csv(new)
        assert len(a)==len(b)==525 and np.array_equal(a.event_id_hash,b.event_id_hash)
        exact=['sequence_id','event_time','snapshot_cutoff','label','final_label','route_requested','escalated','accepted','fallback','state_version_before','state_version_after','reference_count','measured']
        for c in exact:assert a[c].fillna('').equals(b[c].fillna('')),'replay field changed: '+c
        score=max(float(np.max(np.abs(a[c].to_numpy()-b[c].to_numpy()))) for c in ('local_score','final_score'))
        assert score<=1e-6,'replay numeric mismatch'
        report['replay']={'N_input':525,'N_measured':500,'max_score_difference':score,'decision_route_state_event_order_disagreements':0,'original_trace_sha256':sha256(old),'replica_trace_sha256':sha256(new),
            'ignored_intentionally':'different run/reference/contract-alias identity, input string byte accounting, timing/ingestion timestamps and allocator instrumentation',
            'timing_not_expected_identical':True}
    target=out/'audits'/('alias_replication_'+args.mode+'.json');write(target,report)
    print('ACTUAL ALIAS REPLICATION PASS',args.mode,flush=True)


if __name__=='__main__':main()
