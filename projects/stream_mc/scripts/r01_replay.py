"""Measured runtime populations: batched contracts, short prefix, long events."""
from __future__ import annotations
import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path
from r01_common import ROOT,config,output,write,read,sha256
import numpy as np
import pandas as pd
import psutil
import torch
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.streaming.selective import seed_all,binary_metrics
from gog_fraud.streaming.selective_engine import SelectivePredictor,SelectiveReplayEngine
from gog_fraud.streaming.selective_state import BoundedQueue
from r01_train import load_cache,subset


def events(cfg,count=None):
    frame=pd.read_parquet(ROOT/cfg['raw_events'])
    frame=frame.sort_values(['event_time','block_number','transaction_index','sample_id'],kind='stable').reset_index(drop=True)
    if count is not None:frame=frame.iloc[:count]
    for i,r in enumerate(frame.itertuples(index=False)):
        yield {'sequence_id':i,'event_id':hashlib.sha256(r.sample_id.encode()).hexdigest(),'chain':r.chain_id,'contract_id':r.contract_id,
               'timestamp':int(r.event_time),'source':r.src,'target':r.dst,'label':int(r.label)}


def model_dir(cfg,seed=None):return output()/f"models/pooled_snapshot_GIN_seed{seed or cfg['primary_seed']}"


def engine(cfg,family,seed=None,budget=None,repeat=0):
    predictor=SelectivePredictor(model_dir(cfg,seed),cfg['device'],family,budget if budget is not None else cfg['policy']['primary_deep_budget'],repeat)
    s=cfg['state'];predictor.cache.capacity=s['cache_entries'];predictor.cache.byte_cap=s['cache_bytes']
    return SelectiveReplayEngine(predictor,capacity=s['max_contracts'],max_edges=s['max_edges'],max_nodes=s['max_nodes'],ttl=s['ttl_seconds'])


class TraceWriter:
    def __init__(self,path,rows=1000):self.path=path;self.limit=rows;self.buffer=[];self.handle=None;self.writer=None
    def add(self,row):
        self.buffer.append(row)
        if len(self.buffer)>=self.limit:self.flush()
    def flush(self):
        if not self.buffer:return
        if self.handle is None:
            self.path.parent.mkdir(parents=True,exist_ok=True);self.handle=self.path.open('w',newline='')
            self.writer=csv.DictWriter(self.handle,fieldnames=self.buffer[0].keys());self.writer.writeheader()
        self.writer.writerows(self.buffer);self.buffer=[];self.handle.flush()
    def close(self):
        self.flush()
        if self.handle:self.handle.close()


def replay(cfg,family,population,repeat=0):
    dest=output()/f'runtime/{population}_{family}_repeat{repeat}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return read(dest/'complete.json')
    seed_all(cfg['primary_seed']);eng=engine(cfg,family);process=psutil.Process();cpu_start=process.cpu_times();rss_start=process.memory_info().rss
    torch.cuda.reset_peak_memory_stats();queue=BoundedQueue(cfg['state']['queue_limit']);buffer=TraceWriter(dest/'event_trace.csv',cfg['state']['trace_rows']);memory=TraceWriter(dest/'system_trace.csv',cfg['state']['trace_rows'])
    count=cfg['runtime']['events']+cfg['runtime']['warmup_events'] if population=='prefix' else cfg['runtime']['integrated_events']
    latencies=[];scores=[];labels=[];preds=[];unique=set();fraud_contracts=set();counts={};deep=0;miss=0;peak_rss=rss_start;checkpoint_ms=0.;checkpoint_hash=None
    warm=cfg['runtime']['warmup_events'] if population=='prefix' else 0
    # Source reading/ordering occurs before measurement; declared input preload.
    iterator=events(cfg,count);first=next(iterator);source=[first] if count==1 else None
    import itertools
    wall_start=time.perf_counter();ingest_start=time.time()
    for i,event in enumerate(itertools.chain([first],iterator)):
        if i==warm:wall_start=time.perf_counter()
        ingest_time=time.time()
        unique.add((event['chain'],event['contract_id']));counts[(event['chain'],event['contract_id'])]=counts.get((event['chain'],event['contract_id']),0)+1
        if event['label']:fraud_contracts.add((event['chain'],event['contract_id']))
        if not queue.put(event):raise AssertionError('serial source unexpectedly exceeds queue')
        queued,wait_ms=queue.pop();row=eng.process(queued);measured=i>=warm
        if measured:
            latencies.append(row['end_to_end_ms']);scores.append(row['final_score']);labels.append(event['label']);preds.append(row['final_label']);deep+=int(row['escalated']);miss+=int(not row['accepted'])
        if population=='long' and i+1==cfg['runtime']['checkpoint_event']:
            t=time.perf_counter();checkpoint_hash=eng.checkpoint(dest/'state.checkpoint');checkpoint_ms=(time.perf_counter()-t)*1000
        gpu_alloc=torch.cuda.memory_allocated()/2**20;gpu_res=torch.cuda.memory_reserved()/2**20
        if i%1000==0 or i+1==count:
            rss=process.memory_info().rss;peak_rss=max(peak_rss,rss)
            memory.add({'sequence_id':i,'resident_contracts':len(eng.state.states),'state_payload_budget_bytes':eng.state.payload_bytes(),'cache_entries':len(eng.predictor.cache.values),'cache_payload_bytes':eng.predictor.cache.bytes,
                'rss_mib':rss/2**20,'gpu_allocated_mib':gpu_alloc,'gpu_reserved_mib':gpu_res,**eng.state.counters})
        record={'run_id':cfg['run_id'],'experiment_id':'replay_systems','population':population,'policy':family,'repeat_id':repeat,'sequence_id':i,
            'event_id_hash':hashlib.sha256(event['event_id'].encode()).hexdigest(),'contract_key':hashlib.sha256(f"{event['chain']}:{event['contract_id']}".encode()).hexdigest(),
            'chain':event['chain'],'event_time':event['timestamp'],'ingest_time':ingest_time,'snapshot_cutoff':event['timestamp'],
            'policy_hash':eng.predictor.identity['policy_hash'],'state_version_before':row['version_before'],'state_version_after':row['version_after'],
            'accepted':row['accepted'],'rejection_reason':row['reason'],'label':event['label'],'local_score':row.get('local_score'),'final_score':row['final_score'],'final_label':row['final_label'],
            'route_requested':row.get('route_requested',False),'escalated':row['escalated'],'fallback':row.get('fallback',''),'queue_wait_ms':wait_ms,
            'feature_update_ms':row['feature_update_ms'],'local_ms':row.get('local_ms',0.),'retrieval_ms':row.get('retrieval_ms',0.),'relational_ms':row.get('relational_ms',0.),
            'end_to_end_ms':row['end_to_end_ms'],'resident_contracts':len(eng.state.states),'state_payload_budget_bytes':eng.state.payload_bytes(),'cache_entries':len(eng.predictor.cache.values),
            'cache_hits':eng.predictor.cache.hits,'cache_misses':eng.predictor.cache.misses,'queue_length':len(queue.items),'gpu_allocated_mib':gpu_alloc,'gpu_reserved_mib':gpu_res,
            'selected_reference_max_cutoff':row.get('selected_reference_max_cutoff'),'reference_count':row.get('reference_count',0),'checkpoint_id':checkpoint_hash,'replay_cursor':eng.state.cursor,'measured':measured}
        buffer.add(record)
        if i and i%10000==0:print(population,family,i,'events',flush=True)
    elapsed=time.perf_counter()-wall_start;buffer.close();memory.close();cpu_end=process.cpu_times()
    good=np.array([score is not None for score in scores]);p=np.array([s if s is not None else np.nan for s in scores]);yy=np.array(labels);hh=np.array([h if h is not None else -1 for h in preds])
    result={'run_id':cfg['run_id'],'experiment_id':'replay_systems','population':population,'policy':family,'repeat_id':repeat,'identity':eng.predictor.identity,
        'N_measured':len(latencies),'warmup_events':warm,'N_input':count,'unique_contracts':len(unique),'fraud_contracts':len(fraud_contracts),'max_events_per_contract':max(counts.values()),
        'contract_label_replicated_to_events':True,'event_fraud_ratio':float(yy.mean()),'predictive_scope':'secondary diagnostic, not transaction fraud truth or primary contract performance',
        'deep_fraction':deep/len(latencies),'rejected_events':miss,'abstentions':int((~good).sum()),'wall_seconds':elapsed,'throughput_events_per_second':len(latencies)/elapsed,
        'p50_ms':float(np.quantile(latencies,.5)),'p95_ms':float(np.quantile(latencies,.95)),'p99_ms':float(np.quantile(latencies,.99)),
        'rss_start_mib':rss_start/2**20,'rss_sampled_peak_mib':peak_rss/2**20,'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
        'cpu_utilization_percent_one_core':100*((cpu_end.user-cpu_start.user)+(cpu_end.system-cpu_start.system))/elapsed,
        'gpu_utilization_percent':'not sampled; allocated/reserved bytes and synchronized timing are measured',
        'checkpoint_ms':checkpoint_ms,'checkpoint_hash':checkpoint_hash,'memory_breakdown':eng.predictor.memory,'state_counters':eng.state.counters,
        'clock':'time.perf_counter wall clock; CUDA synchronize; serial queue; source decode/sort excluded, logging included in throughput but not scoring latency',
        'source_preloaded_rows':100000,'queue_high_water':queue.high_water,'trace_buffer_cap':cfg['state']['trace_rows'],
        'metrics':binary_metrics(yy[good],p[good],hh[good]),'trace_sha256':sha256(dest/'event_trace.csv'),'system_trace_sha256':sha256(dest/'system_trace.csv')}
    write(dest/'complete.json',result);print('REPLAY COMPLETE',population,family,result['p99_ms'],flush=True);return result


def offline(cfg):
    dest=output()/'runtime/offline_v2';dest.mkdir(parents=True,exist_ok=True)
    receipt=dest/'budget_runtime_complete.json'
    if receipt.exists():
        saved=read(receipt)
        if saved['config_sha256']!=cfg['config_sha256'] or saved['executed_replay_source_sha256']!=sha256(__file__):raise ValueError('offline v2 resume source/config drift')
        for name,value in saved['artifacts'].items():
            if sha256(dest/name)!=value:raise ValueError('offline v2 resume artifact drift: '+name)
        print('VERIFIED EXISTING OFFLINE V2, no new timing asserted',flush=True);return
    seed_all(cfg['primary_seed']);cache=load_cache(cfg);graphs,metadata=subset(cache,cfg['chains'],'test');rows=[]
    batch_trace=[]
    predictions=dest/'budget_predictions';predictions.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        seed_all(seed)
        fit=read(model_dir(cfg,seed)/'policy_fit.json');reference=pd.read_parquet(model_dir(cfg,seed)/'test_predictions.parquet')
        for policy in fit['policies']:
            for repeat in range(cfg['runtime']['offline_workload_repeats']):
                predictor=SelectivePredictor(model_dir(cfg,seed),cfg['device'],policy['family'],policy['budget'],policy['random_repeat'])
                # Warm only CUDA with one local batch; router RNG and caches remain fresh.
                from gog_fraud.streaming.selective import local_predict
                from torch_geometric.data import Batch
                local_predict(predictor.local,Batch.from_data_list(graphs[:25]).to(predictor.device),1);predictor.synchronize()
                totals={'local_ms':0.,'retrieval_ms':0.,'relational_ms':0.,'wall_ms':0.};pp=[];hh=[];rr=[]
                for start in range(0,len(graphs),128):
                    subset_meta=metadata[start:start+128]
                    p,h,r,times=predictor.predict_batch(graphs[start:start+128],[m['event_end'] for m in subset_meta],[m['sample_id'] for m in subset_meta])
                    pp.extend(p);hh.extend(h);rr.extend(r)
                    batch_trace.append({'run_id':cfg['run_id'],'experiment_id':'pooled_snapshot','seed':seed,'family':policy['family'],'budget':policy['budget'],
                        'random_repeat':policy['random_repeat'],'runtime_repeat':repeat,'batch_start':start,'N_batch':len(p),'N_deep':int(r.sum()),**times})
                    for key,value in times.items():totals[key]+=value
                row={'run_id':cfg['run_id'],'measurement_run_id':cfg['run_id']+'_offline_v2','experiment_id':'pooled_snapshot','seed':seed,'family':policy['family'],'budget':policy['budget'],'random_repeat':policy['random_repeat'],'runtime_repeat':repeat,
                    'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'torch_threads':torch.get_num_threads(),
                    'policy_hash':policy['policy_hash'],'batch_size':128,'N':len(graphs),'deep_fraction':float(np.mean(rr)),'latency_ms_per_contract':totals['wall_ms']/len(graphs),
                    **totals,**binary_metrics(np.array([m['label'] for m in metadata]),np.array(pp),np.array(hh))}
                if policy['family']=='margin' and policy['budget']==.25:
                    row['offline_scalar_batched_score_max_abs_diff']=float(np.max(np.abs(reference.selective_probability-np.array(pp))))
                    row['offline_scalar_batched_prediction_disagreements']=int(np.sum(reference.selective_prediction!=np.array(hh)))
                rows.append(row)
                if repeat==0:
                    filename=f"seed{seed}_{policy['family']}_q{policy['budget']}_random{policy['random_repeat']}.npz"
                    np.savez_compressed(predictions/filename,probability=np.asarray(pp),prediction=np.asarray(hh),escalated=np.asarray(rr),
                        label=np.array([m['label'] for m in metadata]),sample_key=np.array([hashlib.sha256(m['sample_id'].encode()).hexdigest() for m in metadata]))
            print('BUDGET RUNTIME',seed,policy['family'],policy['budget'],flush=True)
        pd.DataFrame(rows).to_csv(dest/'budget_runtime.csv',index=False)
        pd.DataFrame(batch_trace).to_csv(dest/'offline_batch_trace.csv',index=False)
    write(receipt,{'N_cells':len(rows),'measurement_run_id':cfg['run_id']+'_offline_v2','model_run_id':cfg['run_id'],
        'config_sha256':cfg['config_sha256'],'executed_replay_source_sha256':sha256(__file__),
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'torch_threads':torch.get_num_threads(),
        'artifacts':{str(p.relative_to(dest)):sha256(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='budget_runtime_complete.json'},
        'source_sha256':sha256(dest/'budget_runtime.csv'),'cost':'actual batched local + retrieval + only selected relational forward; no deep-fraction proxy','device':cfg['device'],
        'supersedes':'root runtime/budget_* v1 preserved for diagnosis; v1 offline process lacked requested determinism/thread settings'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prefix','long','offline','all'],default='all');parser.add_argument('--policy',choices=['local','margin','full']);parser.add_argument('--repeat',type=int);args=parser.parse_args();cfg=config()
    if args.mode in ('offline','all'):offline(cfg)
    for population in ('prefix','long'):
        if args.mode not in (population,'all'):continue
        for family in [args.policy] if args.policy else cfg['runtime']['policies']:
            for repeat in [args.repeat] if args.repeat is not None else range(cfg['runtime']['repeats'] if population=='prefix' else 1):replay(cfg,family,population,repeat)


if __name__=='__main__':main()
