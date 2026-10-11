"""One shared selective inference path for snapshots and bounded systems replay."""
from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch
from gog_fraud.data.level2.relation_builder import HistoricalReferenceIndex
from .selective import LocalEncoder,RelationalEncoder,local_predict,calibrated,fuse,router_score
from .selective_state import BoundedContractState,BoundedHotCache,rng_state,restore_rng,durable_checkpoint,load_checkpoint


def artifact_sha256(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(2**20),b''):value.update(block)
    return value.hexdigest()


class SelectivePredictor:
    def __init__(self,model_dir,device='cuda:0',family='margin',budget=.25,repeat=0,timeout_seconds=None):
        path=Path(model_dir);self.device=torch.device(device);self.timeout_seconds=timeout_seconds
        self.fit=json.loads((path/'policy_fit.json').read_text());self.mode=family
        actual_model=hashlib.sha256(json.dumps({'local':artifact_sha256(path/'local.pt'),'relational':artifact_sha256(path/'relational.pt')},sort_keys=True,allow_nan=False).encode()).hexdigest()
        if actual_model!=self.fit['model_hash']:raise ValueError('model artifact SHA-256 mismatch')
        if artifact_sha256(path/'reference.npz')!=self.fit['reference_manifest_hash']:raise ValueError('reference artifact SHA-256 mismatch')
        self.policy=next(p for p in self.fit['policies'] if p['family']==family and p['budget']==budget and p['random_repeat']==repeat) if family not in ('local','full','threshold_only') else None
        checkpoint=torch.load(path/'local.pt',map_location=self.device,weights_only=False)
        self.local=LocalEncoder(checkpoint['backbone']).to(self.device);self.local.load_state_dict(checkpoint['state_dict']);self.local.eval()
        self.deep=RelationalEncoder().to(self.device);self.deep.load_state_dict(torch.load(path/'relational.pt',map_location=self.device,weights_only=False)['state_dict']);self.deep.eval()
        with np.load(path/'reference.npz') as refs:
            self.index=HistoricalReferenceIndex(refs['embedding'],refs['score'],refs['ids'],refs['cutoffs'],k=8,metric='euclidean')
        self.reference_contracts={':'.join(i.split(':')[:2]):i for i in self.index.ids}
        self.random=np.random.default_rng((self.policy or {}).get('random_seed',0)+10**6)
        self.cache=BoundedHotCache()
        self.identity={'checkpoint_identity_schema':2,'fit_artifact_hash':artifact_sha256(path/'policy_fit.json'),
            'model_hash':self.fit['model_hash'],'reference_hash':self.fit['reference_manifest_hash'],
            'policy_hash':self.policy['policy_hash'] if self.policy else self.mode,'run_id':self.fit['run_id'],'experiment_id':self.fit['experiment_id']}
        self.memory={'immutable_reference_arrays_bytes':sum(a.nbytes for a in (self.index.embeddings,self.index.scores,self.index.ids,self.index.cutoffs,self.index.search_embeddings)),
                     'search_index_numeric_bytes':self.index.tree.data.nbytes+self.index.tree.indices.nbytes,
                     'model_tensor_bytes':sum(p.numel()*p.element_size() for model in (self.local,self.deep) for p in model.parameters())}

    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)

    def route(self,raw,p,variance,embedding):
        if self.mode in ('local','threshold_only'):return False,None
        if self.mode=='full':return True,None
        policy=self.policy
        if policy['family']=='random':rank=float(self.random.random())
        elif policy['family']=='learned_benefit':
            entropy=-(raw*np.log(max(raw,1e-8))+(1-raw)*np.log(max(1-raw,1e-8)))
            features=np.array([raw,entropy,variance,np.linalg.norm(embedding)])
            rank=float(features@np.asarray(policy['benefit_coefficients'])+policy['benefit_intercept'])
        else:rank=float(router_score(np.array([p]),policy['fast_threshold'],policy['family'])[0])
        decision=rank>=policy['route_cutoff'] if policy['route_cutoff'] is not None else policy['budget']==1
        return bool(decision),rank

    @torch.inference_mode()
    def predict_batch(self,graphs,cutoffs,target_ids):
        """Actual local+retrieval+selected-deep execution; offline batch workload."""
        self.synchronize();start=time.perf_counter();t=start
        raw,embedding,var=local_predict(self.local,Batch.from_data_list(graphs).to(self.device),1);self.synchronize()
        local_ms=(time.perf_counter()-t)*1000;fast=calibrated(raw,self.fit['fast_map'])
        requested=np.array([self.route(float(r),float(p),float(v),e)[0] for r,p,v,e in zip(raw,fast,var,embedding)])
        selected=[];positions=[];t=time.perf_counter()
        for i in np.flatnonzero(requested):
            star,audit=self.index.target_graph(embedding[i],float(raw[i]),int(cutoffs[i]),target_ids[i])
            if audit['reference_count']:selected.append(star);positions.append(i)
        retrieval_ms=(time.perf_counter()-t)*1000;t=time.perf_counter();score=fast.copy();route=np.zeros(len(raw),bool)
        if selected:
            deep_raw=torch.sigmoid(self.deep(Batch.from_data_list(selected).to(self.device))).cpu().numpy();self.synchronize()
            score[positions]=fuse(fast[positions],calibrated(deep_raw,self.fit['deep_map']),self.fit['weight']);route[positions]=True
        relational_ms=(time.perf_counter()-t)*1000
        threshold=self.policy['final_threshold'] if self.policy else self.fit['controls'][self.mode]['threshold']
        return score,score>=threshold,route,{'local_ms':local_ms,'retrieval_ms':retrieval_ms,'relational_ms':relational_ms,'wall_ms':(time.perf_counter()-start)*1000}

    @torch.inference_mode()
    def predict(self,graph,cutoff,target_id=''):
        begin=time.perf_counter();self.synchronize();local_start=time.perf_counter()
        if graph.num_nodes==0:return {'final_score':None,'final_label':None,'fallback':'empty_graph','escalated':False,'end_to_end_ms':0.,'local_ms':0.,'retrieval_ms':0.,'relational_ms':0.}
        raw,emb,var=local_predict(self.local,graph.to(self.device),1);self.synchronize()
        local_ms=(time.perf_counter()-local_start)*1000;raw=float(raw[0]);embedding=emb[0]
        if not np.isfinite(raw) or not np.isfinite(embedding).all():
            return {'final_score':None,'final_label':None,'fallback':'invalid_local_score','escalated':False,'end_to_end_ms':(time.perf_counter()-begin)*1000,'local_ms':local_ms,'retrieval_ms':0.,'relational_ms':0.}
        fast=float(calibrated(np.array([raw]),self.fit['fast_map'])[0]);requested,rank=self.route(raw,fast,float(var[0]),embedding)
        threshold=self.policy['final_threshold'] if self.policy else self.fit['controls'][self.mode]['threshold']
        score=fast;deep_raw=None;fallback='';retrieval_ms=0.;deep_ms=0.;audit={};executed=False
        if requested:
            start=time.perf_counter();star,audit=self.index.target_graph(embedding,raw,cutoff,target_id)
            # Cached immutable features are loaded from the reference arrays on misses.
            indices,_=self.index.query(embedding,cutoff,target_id)
            for i,ix in enumerate(indices):
                value=self.cache.get(int(ix),lambda ix=ix:np.r_[self.index.embeddings[ix],self.index.scores[ix]])
                star.x[i]=torch.from_numpy(value)
            retrieval_ms=(time.perf_counter()-start)*1000
            if not audit['reference_count']:fallback='no_eligible_reference'
            elif self.timeout_seconds is not None and time.perf_counter()-begin>self.timeout_seconds:fallback='soft_timeout_before_relational'
            else:
                start=time.perf_counter();deep_raw=float(torch.sigmoid(self.deep(star.to(self.device))).cpu()[0]);self.synchronize();deep_ms=(time.perf_counter()-start)*1000;executed=True
                if not np.isfinite(deep_raw):fallback='invalid_relational_score'
                elif self.timeout_seconds is not None and time.perf_counter()-begin>self.timeout_seconds:fallback='soft_timeout_after_relational'
                else:score=float(fuse(np.array([fast]),calibrated(np.array([deep_raw]),self.fit['deep_map']),self.fit['weight'])[0])
        # Retrieval's empty fallback must not erase timeout/invalid-deep outcomes.
        audit.pop('fallback',None)
        return {'local_raw_score':raw,'local_score':fast,'final_score':score,'final_label':int(score>=threshold),
            'final_threshold':threshold,'route_score':rank,'route_requested':requested,'escalated':executed,
            'actual_relational_raw_score':deep_raw,'fallback':fallback,'local_ms':local_ms,'retrieval_ms':retrieval_ms,'relational_ms':deep_ms,
            'end_to_end_ms':(time.perf_counter()-begin)*1000,**audit}


class SelectiveReplayEngine:
    def __init__(self,predictor,**state_options):self.predictor=predictor;self.state=BoundedContractState(**state_options)

    def process(self,event):
        start=time.perf_counter();update_start=start;status=self.state.update(event);update_ms=(time.perf_counter()-update_start)*1000
        if not status['accepted']:return {**status,'final_score':None,'final_label':None,'escalated':False,'end_to_end_ms':(time.perf_counter()-start)*1000,'feature_update_ms':update_ms}
        graph=self.state.graph(status['key']);update_ms=(time.perf_counter()-update_start)*1000
        target_contract=f"{event['chain']}:{event['contract_id']}"
        target_id=self.predictor.reference_contracts.get(target_contract,'')
        result=self.predictor.predict(graph,int(event['timestamp']),target_id)
        result.update({**status,'feature_update_ms':update_ms,'end_to_end_ms':(time.perf_counter()-start)*1000,
                       'resident_contracts':len(self.state.states),'state_bytes':self.state.payload_bytes(),'cache_entries':len(self.predictor.cache.values),
                       'cache_hits':self.predictor.cache.hits,'cache_misses':self.predictor.cache.misses})
        result.pop('key',None);return result

    def checkpoint(self,path,fail_at=None):
        payload={'identity':self.predictor.identity,'state':self.state,'cache':self.predictor.cache,
            'cursor':self.state.cursor,'rng':rng_state(),'router_rng':self.predictor.random.bit_generator.state}
        return durable_checkpoint(path,payload,fail_at=fail_at)

    def restore(self,path):
        payload=load_checkpoint(path,self.predictor.identity);self.state=payload['state'];self.predictor.cache=payload['cache']
        restore_rng(payload['rng']);self.predictor.random.bit_generator.state=payload['router_rng'];return payload['cursor']
