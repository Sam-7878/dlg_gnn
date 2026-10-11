"""Synthetic state boundaries and actual model replay across abrupt checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from r01_common import ROOT,config,output,write,read,sha256
import numpy as np
import pandas as pd
import psutil
import torch
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.streaming.selective_state import BoundedContractState,BoundedHotCache,BoundedQueue,load_checkpoint
from r01_replay import events,engine


def synthetic(i,contract=None,timestamp=None,unique_nodes=False):
    return {'sequence_id':i,'event_id':f'synthetic_event_{i}','chain':'synthetic','contract_id':contract or f'contract_{i}',
        'timestamp':i if timestamp is None else timestamp,'source':f'src_{i}' if unique_nodes else 'a','target':f'dst_{i}' if unique_nodes else 'b'}


def state_stress(cfg,dest):
    proc=psutil.Process();rows=[];s=cfg['state'];state=BoundedContractState(s['max_contracts'],s['max_edges'],s['max_nodes'],s['ttl_seconds']);start=time.perf_counter();peak=proc.memory_info().rss
    for i in range(cfg['stress']['unique_contracts']+cfg['stress']['churn_events']):
        state.update(synthetic(i));resident=len(state.states)
        if resident>s['max_contracts']:raise AssertionError('resident cap exceeded')
        if i%500==0 or i+1 in (cfg['stress']['unique_contracts'],cfg['stress']['unique_contracts']+cfg['stress']['churn_events']):
            rss=proc.memory_info().rss;peak=max(peak,rss)
            rows.append({'sequence_id':i,'synthetic':True,'resident_contracts':resident,'capacity_evictions':state.counters['capacity_evictions'],'rss_mib':rss/2**20,
                'state_payload_budget_bytes':state.payload_bytes(),'state_serialized_bytes':len(pickle.dumps(state,protocol=5)),'elapsed_seconds':time.perf_counter()-start})
    assert state.invariants()
    edge=BoundedContractState(max_edges=128,max_nodes=128)
    for i in range(400):edge.update(synthetic(i,contract='edge_cap'))
    assert len(edge.states[('synthetic','edge_cap')]['edges'])==128
    nodes=BoundedContractState(max_edges=128,max_nodes=128)
    for i in range(400):nodes.update(synthetic(i,contract='node_cap',unique_nodes=True))
    assert nodes.invariants() and len(nodes.states[('synthetic','node_cap')]['nodes'])==128
    ttl=BoundedContractState(ttl=s['ttl_seconds']);ttl.update(synthetic(0,contract='ttl',timestamp=0))
    ttl.update(synthetic(1,contract='new',timestamp=s['ttl_seconds']+1));assert ('synthetic','ttl') not in ttl.states
    ttl.update(synthetic(2,contract='ttl',timestamp=s['ttl_seconds']+2));assert ttl.states[('synthetic','ttl')]['version']==1
    cache=BoundedHotCache(s['cache_entries'],s['cache_bytes'])
    for i in range(25000):cache.get(i,lambda i=i:np.full(65,i,np.float32))
    assert len(cache.values)==10000 and cache.bytes<=s['cache_bytes'];before=cache.misses;cache.get(0,lambda:np.zeros(65,np.float32));assert cache.misses==before+1
    queue=BoundedQueue(s['queue_limit']);input_rows=[synthetic(i) for i in range(2000)];processed=[];waits=[]
    for event in input_rows:
        while not queue.put(event):
            e,wait=queue.pop();processed.append(e['event_id']);waits.append(wait)
    while queue.items:e,wait=queue.pop();processed.append(e['event_id']);waits.append(wait)
    assert processed==[e['event_id'] for e in input_rows] and queue.high_water==512 and queue.retries>0
    duplicate=BoundedContractState();duplicate.update(synthetic(0,contract='dup',timestamp=5));dupe=synthetic(1,contract='dup',timestamp=5);dupe['event_id']='synthetic_event_0'
    assert duplicate.update(dupe)['reason']=='duplicate';assert duplicate.update(synthetic(2,contract='late',timestamp=4))['reason']=='late'
    tiny=BoundedContractState(capacity=2)
    for i in range(5):tiny.update(synthetic(i))
    assert tiny.counters['capacity_evictions']==3
    pd.DataFrame(rows).to_csv(dest/'state_pressure_trace.csv',index=False)
    report={'synthetic_not_fraud_evidence':True,'N_unique':cfg['stress']['unique_contracts'],'N_additional_churn':cfg['stress']['churn_events'],'resident_peak':max(r['resident_contracts'] for r in rows),
        'capacity_evictions':state.counters['capacity_evictions'],'state_invariants':state.invariants(),'tiny_cap_evictions':tiny.counters['capacity_evictions'],
        'edge_cap':len(edge.states[('synthetic','edge_cap')]['edges']),'edge_truncations':edge.counters['edge_truncations'],'node_cap':len(nodes.states[('synthetic','node_cap')]['nodes']),'node_truncations':nodes.counters['node_truncations'],
        'ttl_expiry_count':ttl.counters['ttl_evictions'],'ttl_reentry_version':ttl.states[('synthetic','ttl')]['version'],
        'cache_entries':len(cache.values),'cache_bytes':cache.bytes,'cache_evictions':cache.evictions,'cache_misses':cache.misses,'cache_miss_reloads_immutable_source':True,
        'queue_high_water':queue.high_water,'queue_retries':queue.retries,'queue_input_events':len(input_rows),'queue_processed_events':len(processed),'queue_p99_wait_ms':float(np.quantile(waits,.99)),
        'late_quarantine':duplicate.counters['late'],'duplicate_window_detected':duplicate.counters['duplicates'],'elapsed_seconds':time.perf_counter()-start,'rss_peak_mib':peak/2**20,
        'memory_claim':'bounded logical payload and serialized state; allocator RSS is measured, not asserted O(1) for the full process',
        'ttl_cache_invalidation':'reference cache is immutable model-version keyed; TTL removes contract state, not immutable reference features',
        'trace_sha256':sha256(dest/'state_pressure_trace.csv')}
    write(dest/'bounded_state_audit.json',report);print('STATE STRESS COMPLETE',report,flush=True)


def restart_events(cfg):
    # Tail timestamps exercise eligible references; input sequence is fixture-local.
    source=list(events(cfg))[-60:]
    for i,event in enumerate(source):event['sequence_id']=i
    return source


def crash_worker(cfg,phase,path,policy):
    eng=engine(cfg,policy);eng.restore(path)
    for event in restart_events(cfg)[:31]:
        if event['sequence_id']>eng.state.cursor:eng.process(event)
    eng.checkpoint(path,fail_at=phase)
    raise AssertionError('failure fixture did not terminate')


def restart(cfg,dest):
    source=restart_events(cfg);eng=engine(cfg,'full');reference=[]
    for event in source:
        reference.append(eng.process(event))
        if event['sequence_id']==19:eng.checkpoint(dest/'baseline.checkpoint')
    assert sum(r['escalated'] for r in reference)>0,'restart fixture must execute the relational stage'
    rows=[];failures=[]
    for phase,expected_cursor in [('before',19),('during',19),('after',30)]:
        path=dest/f'{phase}.checkpoint';shutil.copy2(dest/'baseline.checkpoint',path)
        result=subprocess.run([sys.executable,__file__,'--crash',phase,'--checkpoint',str(path)],cwd=ROOT,capture_output=True,text=True,env={**__import__('os').environ,'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONPATH':str(ROOT/'src')})
        failures.append({'phase':phase,'exit_code':result.returncode,'expected_exit':{'before':91,'during':92,'after':93}[phase]})
        assert result.returncode==failures[-1]['expected_exit'],result.stderr
        recovered=engine(cfg,'full');cursor=recovered.restore(path);assert cursor==expected_cursor
        for event in source[cursor+1:]:
            actual=recovered.process(event);expected=reference[event['sequence_id']]
            a,b=actual['final_score'],expected['final_score'];delta=abs(a-b) if a is not None and b is not None else 0. if a is b else np.inf
            rows.append({'phase':phase,'sequence_id':event['sequence_id'],'event_id_hash':hashlib.sha256(event['event_id'].encode()).hexdigest(),
                'recovery_cursor':cursor,'score_abs_diff':delta,'score_bit_identical':a==b,'label_disagreement':actual['final_label']!=expected['final_label'],
                'accepted':actual['accepted'],'state_version_equal':actual['version_after']==expected['version_after']})
        # Check identity rejection using actual serialized payload.
        try:load_checkpoint(path,{'wrong':'identity'})
        except ValueError:pass
        else:raise AssertionError('checkpoint identity mismatch accepted')
    frame=pd.DataFrame(rows);frame.to_csv(dest/'checkpoint_diff.csv',index=False)
    report={'failure_model':'abrupt os._exit before write, after half write+fsync, after atomic replace+directory fsync; not power failure or multi-consumer exactly-once',
        'failures':failures,'N_compared':len(rows),'score_max_abs_diff':float(frame.score_abs_diff.max()),'score_tolerance':1e-6,
        'score_bit_identical_count':int(frame.score_bit_identical.sum()),'label_disagreements':int(frame.label_disagreement.sum()),
        'state_version_mismatches':int((~frame.state_version_equal).sum()),'replayed_input_loss_count':int((~frame.accepted).sum()),'identity_negative_fixture':True,
        'source_schema':'real provider-derived retained event records; labels are inherited contract labels', 'source_sha256':sha256(ROOT/cfg['raw_events']),
        'model_policy_reference_identity':eng.predictor.identity,'policy_scope':'full path, with actual eligible relational execution; not every policy/failure workload',
        'N_reference_relational_executions':sum(r['escalated'] for r in reference),'fixture_sequence':'last60 real chronological events, sequence renumbered only for restart fixture',
        'checkpoint_contains_rng_state':True,'diff_sha256':sha256(dest/'checkpoint_diff.csv')}
    assert report['score_max_abs_diff']<=1e-6 and report['label_disagreements']==0 and report['state_version_mismatches']==0 and report['replayed_input_loss_count']==0
    write(dest/'restart_audit.json',report);print('RESTART COMPLETE',report,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--crash',choices=['before','during','after']);parser.add_argument('--checkpoint');parser.add_argument('--mode',choices=['state','restart','all'],default='all');args=parser.parse_args();cfg=config();dest=output()/'stress';dest.mkdir(exist_ok=True)
    if args.crash:crash_worker(cfg,args.crash,Path(args.checkpoint),'full');return
    if args.mode in ('state','all'):state_stress(cfg,dest)
    if args.mode in ('restart','all'):restart(cfg,dest)


if __name__=='__main__':main()
