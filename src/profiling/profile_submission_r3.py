"""Policy-identical event replay with durable state/RNG restoration and bounded traces."""
from __future__ import annotations
import argparse
import copy
import json
import os
import pickle
import random
import sys
import time
import tracemalloc
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import psutil
import torch
from torch_geometric.data import Data
from evidence.experiment_identity import digest, file_digest
from gog_fraud.production.submission_r3 import (FrozenRelations,apply_policy,calibrate,
    local_predict,load_models,metrics,read_config,save_json,set_seed)
from gog_fraud.streaming.subgraph_store import IncrementalSubgraphStore
from gog_fraud.streaming.embedding_cache import EmbeddingCache


def graph_from_state(state,device):
    nodes=state['nodes'] or [state['contract_id']]; lookup={v:i for i,v in enumerate(nodes)}
    edges=[(lookup[s],lookup[d]) for s,d,_ in state['edges'] if s in lookup and d in lookup]
    indegree=np.zeros(len(nodes),np.float32); outdegree=indegree.copy()
    for s,d in edges: outdegree[s]+=1; indegree[d]+=1
    x=np.stack((np.log1p(indegree),np.log1p(outdegree),np.log1p(indegree+outdegree)),axis=1)
    edge=torch.tensor(edges,dtype=torch.long).t().contiguous() if edges else torch.empty((2,0),dtype=torch.long)
    return Data(x=torch.from_numpy(x),edge_index=edge,graph_id=torch.tensor([0])).to(device)


class ReplayEngine:
    def __init__(self,cfg,selection,local,deep,relations,device,policy):
        self.cfg,self.selection,self.local,self.deep,self.relations,self.device,self.policy=cfg,selection,local,deep,relations,device,policy
        b=cfg['base']['bounded_graph']
        self.store=IncrementalSubgraphStore(temporal_window_seconds=b['temporal_window_seconds'],
            max_nodes_per_contract=b['max_nodes'],max_edges_per_contract=b['max_edges'])
        self.cache=EmbeddingCache(max_entries=cfg['base']['profiling']['cache_entries'],
            max_bytes=64*1024**2,ttl_seconds=b['temporal_window_seconds'])
        self.active=OrderedDict(); self.queue=deque(); self.max_queue=0
        self.expired=0; self.evicted=0; self.processed=0; self.deep_calls=0

    def step(self,record):
        started=time.perf_counter_ns()
        key=str(record['chain_id'])+':'+str(record['contract_id'])
        # Labels and future metadata never enter the live graph store or model.
        payload={k:record[k] for k in ('src','dst','edge_id')}
        event=SimpleNamespace(contract_id=key,event_time=int(record['event_time']),sample_id=record['sample_id'],payload=payload)
        self.queue.append(event); self.max_queue=max(self.max_queue,len(self.queue)); event=self.queue.popleft()
        self.active[key]=event.event_time; self.active.move_to_end(key)
        while len(self.active)>self.cfg['streaming']['max_active_contracts']:
            stale,_=self.active.popitem(last=False)
            self.evicted+=len(self.store._events.get(stale,{}))
            self.store._events.pop(stale,None); self.store._last_seen.pop(stale,None)
        self.store.apply_event(event)
        state=self.store.materialize(key,event.event_time)
        graph=graph_from_state(state,self.device)
        local_start=time.perf_counter_ns()
        fs,emb,var=local_predict(self.local,graph,self.selection['mc_T'])
        local_ms=(time.perf_counter_ns()-local_start)/1e6
        fast=calibrate(fs,self.selection['fast_map'])
        route=self.policy=='full_deep' or (self.policy=='primary' and abs(fast[0]-self.selection['fast_threshold'])<=self.selection['route_cutoff'])
        ds=np.array([np.nan]); deep_ms=0.
        if route:
            self.queue.append(key); self.max_queue=max(self.max_queue,len(self.queue)); self.queue.popleft()
            deep_start=time.perf_counter_ns()
            ds=self.relations.predict(self.deep,emb,fs,self.device)
            deep_ms=(time.perf_counter_ns()-deep_start)/1e6
            self.deep_calls+=1
            self.cache.put(key,emb[0].copy(),now=event.event_time,
                model_version=self.selection['model_id'],feature_version='bounded-log-degree-r3')
        score,routed,pred=apply_policy(fs,ds,self.selection,self.policy)
        assert bool(routed[0])==route
        self.processed+=1
        if self.processed%100==0:
            self.expired+=self.store.expire(event.event_time).edges
            for old in list(self.active):
                if old not in self.store._events: del self.active[old]
        return {'sample_id':record['sample_id'],'contract_key':key,'chain_scope':record['chain_id'],
            'event_time':int(record['event_time']),'label':int(record['label']),
            'raw_fast':float(fs[0]),'raw_deep':float(ds[0]) if route else None,
            'calibrated_fast':float(fast[0]),'final_score':float(score[0]),
            'prediction':int(pred[0]),'deep_executed':bool(route),'mc_variance':float(var[0]),
            'local_ms':local_ms,'deep_ms':deep_ms,'latency_ms':(time.perf_counter_ns()-started)/1e6}

    def snapshot(self):
        return {'store':self.store.snapshot(),'cache':self.cache,'active':self.active,
            'queue':self.queue,'max_queue':self.max_queue,'expired':self.expired,'evicted':self.evicted,
            'processed':self.processed,'deep_calls':self.deep_calls,
            'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state(),'python_rng':random.getstate(),
            'cuda_rng':torch.cuda.get_rng_state_all() if self.device.type=='cuda' else None,
            'policy_config_id':self.selection['policy_config_id']}

    def restore(self,state):
        if state['policy_config_id']!=self.selection['policy_config_id']: raise ValueError('checkpoint policy mismatch')
        self.store.restore(state['store'])
        for name in ('cache','active','queue','max_queue','expired','evicted','processed','deep_calls'): setattr(self,name,state[name])
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None: torch.cuda.set_rng_state_all(state['cuda_rng'])


def memory_row(engine,index,rows,process,preload,arrays):
    pycur,pypeak=tracemalloc.get_traced_memory()
    # Serialization sizes are logical payload estimates, not disjoint RSS partitions.
    store_bytes=len(pickle.dumps(engine.store.snapshot(),protocol=5))
    device=engine.device
    return {'events':index,'rss_bytes':process.memory_info().rss,'python_current_bytes':pycur,
        'python_peak_bytes':pypeak,'cuda_allocated_bytes':torch.cuda.memory_allocated(device) if device.type=='cuda' else 0,
        'cuda_reserved_bytes':torch.cuda.memory_reserved(device) if device.type=='cuda' else 0,
        'store_entries':len(engine.store._events),'store_edges':sum(len(b) for b in engine.store._events.values()),
        'store_serialized_bytes':store_bytes,'cache_entries':len(engine.cache._items),
        'cache_payload_bytes':engine.cache.current_bytes,'queue_depth':len(engine.queue),
        'trace_rows':len(rows),'trace_serialized_bytes':len(pickle.dumps(rows,protocol=5)),
        'source_preload_logical_bytes':preload,'metric_array_bytes':arrays,
        'reference_numpy_bytes':engine.relations.embeddings.nbytes+engine.relations.scores.nbytes,
        'model_parameter_bytes':sum(p.numel()*p.element_size() for m in (engine.local,engine.deep) for p in m.parameters()),
        'cpu_tensor_bytes':sum(p.numel()*p.element_size() for m in (engine.local,engine.deep) for p in m.parameters() if p.device.type=='cpu')}


def run(cfg,policy,lane,repeat,limit):
    root=Path(cfg['output_root']); dest=root/('streaming' if lane=='integrated_streaming' else 'runtime')/f'{policy}_repeat{repeat}'
    if (dest/'summary.json').exists(): return json.loads((dest/'summary.json').read_text())
    dest.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(cfg['raw_events']).iloc[:limit]
    assert len(frame)==limit and frame.sample_id.is_unique and frame.event_time.is_monotonic_increasing
    records=frame.to_dict('records'); seed=cfg['primary_seed']; set_seed(seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local,deep=load_models(cfg,seed,device)
    source=root/'offline'/f'seed{seed}'
    selection=json.loads((source/'selection.json').read_text())
    assert selection['mc_T']==json.loads((root/'primary_policy_freeze.json').read_text())['mc_T']
    reference=np.load(source/'training_reference.npz')
    relations=FrozenRelations(reference['embedding'],reference['score'],cfg['base']['level2']['knn_k'])
    warm=ReplayEngine(cfg,selection,local,deep,relations,device,policy)
    for record in records[:cfg['runtime']['warmup_events']]: warm.step(record)
    del warm
    set_seed(seed)  # Same event-index RNG sequence for each policy; warmup state is discarded.
    engine=ReplayEngine(cfg,selection,local,deep,relations,device,policy)
    process=psutil.Process(); tracemalloc.start(1)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    lat=np.empty(limit); scores=np.empty(limit); predictions=np.empty(limit,np.int8); routes=np.empty(limit,bool)
    array_bytes=sum(x.nbytes for x in (lat,scores,predictions,routes))
    preload=int(frame.memory_usage(deep=True).sum())+len(pickle.dumps(records,protocol=5))
    rows=[]; timeline=[memory_row(engine,0,rows,process,preload,array_bytes)]
    restart_errors=0; checkpoints=0; checkpoint_overhead=0.; max_trace=0
    start=time.perf_counter(); trace=dest/'event_trace.csv'; checkpoint_file=dest/'checkpoint.pt'
    # This path is R3-owned; opening the trace once truncates only an incomplete R3 run.
    with trace.open('w') as handle:
        for i,record in enumerate(records):
            if lane=='integrated_streaming' and i==cfg['streaming']['restart_checkpoint']:
                checkpoint_start=time.perf_counter()
                torch.save(engine.snapshot(),checkpoint_file)
                uninterrupted=engine.step(record)
                fresh=ReplayEngine(cfg,selection,local,deep,relations,device,policy)
                fresh.restore(torch.load(checkpoint_file,map_location='cpu',weights_only=False)); engine=fresh
                row=engine.step(record)
                keys=('raw_fast','raw_deep','final_score','prediction','deep_executed','mc_variance')
                restart_errors+=int(any(row[k]!=uninterrupted[k] for k in keys))
                checkpoints+=1; checkpoint_overhead+=time.perf_counter()-checkpoint_start
                del fresh,uninterrupted
            else: row=engine.step(record)
            rows.append(row); lat[i]=row['latency_ms']; scores[i]=row['final_score']; predictions[i]=row['prediction']; routes[i]=row['deep_executed']
            max_trace=max(max_trace,len(rows))
            if (i+1)%cfg['streaming']['memory_checkpoint']==0 or i+1==limit:
                timeline.append(memory_row(engine,i+1,rows,process,preload,array_bytes))
            if len(rows)>=cfg['streaming']['max_trace_rows'] or i+1==limit:
                pd.DataFrame(rows).to_csv(handle,index=False,header=i+1==len(rows)); handle.flush(); rows.clear()
            if (i+1)%10000==0: print(f'{lane} {policy} {i+1}/{limit} deep={routes[:i+1].mean():.4f} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    elapsed=time.perf_counter()-start; timeline.append(memory_row(engine,limit,rows,process,preload,array_bytes))
    tracemalloc.stop(); memory=pd.DataFrame(timeline); memory.to_csv(dest/'memory_timeline.csv',index=False)
    labels=frame.label.to_numpy(); sums=metrics(labels,scores,predictions)
    summary={'evidence_lane':lane,'prediction_unit':'event_with_inherited_contract_label' if lane=='integrated_streaming' else 'raw_event_runtime_trace',
        'policy_family':policy,'policy_config_id':selection['policy_config_id'],'seed':seed,'mc_T':selection['mc_T'],
        'repeat':repeat,'events_seen':len(frame),'events_processed':engine.processed,'unique_contracts':int(frame[['chain_id','contract_id']].drop_duplicates().shape[0]),
        'positive_support':int(labels.sum()),'event_loss_count':len(frame)-engine.processed,'expired_event_count':engine.expired,
        'capacity_evicted_event_count':engine.evicted,'OOM_count':0,'checkpoint_count':checkpoints,
        'checkpoint_overhead_seconds':checkpoint_overhead,'restart_disagreement_count':restart_errors,
        'deep_route_rate':float(routes.mean()),'direct_exit_rate':float(1-routes.mean()),
        'N_deep':int(routes.sum()),'N_direct':int((~routes).sum()),'deep_model_calls':engine.deep_calls,
        'mean_latency_ms':float(lat.mean()),'median_latency_ms':float(np.median(lat)),
        'P95_latency_ms':float(np.quantile(lat,.95)),'P99_latency_ms':float(np.quantile(lat,.99)),
        'throughput_events_per_second':len(frame)/elapsed,'elapsed_seconds':elapsed,
        'throughput_definition':'wall time includes tracing, memory instrumentation and checkpoint verification',
        'max_queue_depth':engine.max_queue,'max_trace_rows':max_trace,
        'max_store_entries':int(memory.store_entries.max()),'max_store_bytes':int(memory.store_serialized_bytes.max()),
        'max_cache_entries':int(memory.cache_entries.max()),'max_cache_bytes':int(memory.cache_payload_bytes.max()),
        'RSS_start':int(memory.rss_bytes.iloc[0]),'RSS_peak':int(memory.rss_bytes.max()),'RSS_end':int(memory.rss_bytes.iloc[-1]),
        'RSS_fitted_slope_bytes_per_event':float(np.polyfit(memory.events,memory.rss_bytes,1)[0]),
        'VRAM_peak':int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else 0,
        'warmup_events':cfg['runtime']['warmup_events'],'prefix_id':digest(frame.sample_id.tolist()),
        'raw_source_sha256':file_digest(cfg['raw_events']), 'trace_sha256':file_digest(trace),
        'diagnostic_metrics':sums,'diagnostic_status':'event-weighted inherited-label; not offline detection evidence'}
    assert engine.deep_calls==int(routes.sum()) and engine.processed==len(frame)
    save_json(dest/'summary.json',summary); print(json.dumps({k:summary[k] for k in ('policy_family','events_processed','mean_latency_ms','deep_route_rate','restart_disagreement_count')}),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--lane',choices=['runtime','streaming'],required=True)
    args=parser.parse_args(); cfg=read_config()
    if args.lane=='runtime':
        for repeat in range(cfg['runtime']['repeats']):
            for policy in (('primary','full_deep') if repeat%2==0 else ('full_deep','primary')):
                run(cfg,policy,'raw_event_runtime',repeat,cfg['runtime']['events'])
    else:
        for policy in cfg['streaming']['policies']: run(cfg,policy,'integrated_streaming',0,cfg['streaming']['events'])


if __name__=='__main__': main()
