"""Bounded replay infrastructure. Not a distributed exactly-once service.

State and input identity are checkpointed atomically on one local filesystem.
Ordering is UTC event-time then input sequence; late records are quarantined.
Duplicate IDs are recognized only in the active contract's bounded edge window.
"""
from __future__ import annotations
import hashlib
import os
import pickle
import random
import time
from collections import OrderedDict,deque
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data


class BoundedContractState:
    def __init__(self,capacity=5000,max_edges=128,max_nodes=128,ttl=7776000):
        if min(capacity,max_edges,max_nodes)<1 or ttl<0:raise ValueError('invalid state limits')
        self.capacity=capacity;self.max_edges=max_edges;self.max_nodes=max_nodes;self.ttl=ttl
        self.states=OrderedDict();self.watermark=None;self.cursor=-1
        self.counters={'capacity_evictions':0,'ttl_evictions':0,'edge_truncations':0,
                       'node_truncations':0,'duplicates':0,'late':0,'accepted':0}
        self.total_edges=0;self.total_nodes=0

    def _remove(self,key,reason):
        state=self.states.pop(key);self.total_edges-=len(state['edges']);self.total_nodes-=len(state['nodes'])
        self.counters[reason]+=1

    def update(self,event):
        ts=int(event['timestamp']);sequence=int(event['sequence_id'])
        if sequence<=self.cursor:return {'accepted':False,'reason':'replayed_sequence','version_before':0,'version_after':0}
        self.cursor=sequence
        if self.watermark is not None and ts<self.watermark:
            self.counters['late']+=1;return {'accepted':False,'reason':'late','version_before':0,'version_after':0}
        self.watermark=ts
        # Event-time LRU order is also last-seen order under monotone input.
        while self.states:
            key,state=next(iter(self.states.items()))
            if ts-state['last_seen']<=self.ttl:break
            self._remove(key,'ttl_evictions')
        key=(str(event['chain']),str(event['contract_id']))
        if len(key[0])>64 or len(key[1])>128:raise ValueError('oversized contract identity')
        state=self.states.get(key)
        if state is None:
            if len(self.states)>=self.capacity:self._remove(next(iter(self.states)),'capacity_evictions')
            state={'edges':deque(),'nodes':[],'last_seen':ts,'version':0};self.states[key]=state
        before=state['version'];eid=str(event['event_id'])
        if len(eid)>128:raise ValueError('oversized event identity')
        if any(edge[2]==eid for edge in state['edges']):
            self.counters['duplicates']+=1
            return {'accepted':False,'reason':'duplicate','version_before':before,'version_after':before}
        src,dst=str(event['source']),str(event['target'])
        if max(len(src),len(dst))>128:raise ValueError('oversized node identity')
        self.total_edges-=len(state['edges']);self.total_nodes-=len(state['nodes'])
        state['edges'].append((src,dst,eid,ts))
        while len(state['edges'])>self.max_edges:state['edges'].popleft();self.counters['edge_truncations']+=1
        nodes=OrderedDict()
        for a,b,_,_ in state['edges']:
            for node in (a,b):nodes[node]=None;nodes.move_to_end(node)
        if len(nodes)>self.max_nodes:
            keep=set(list(nodes)[-self.max_nodes:]);self.counters['node_truncations']+=len(nodes)-self.max_nodes
            state['edges']=deque(e for e in state['edges'] if e[0] in keep and e[1] in keep)
            nodes=OrderedDict((node,None) for node in nodes if node in keep)
        state['nodes']=list(nodes);state['last_seen']=ts;state['version']+=1
        self.total_edges+=len(state['edges']);self.total_nodes+=len(state['nodes'])
        self.states.move_to_end(key);self.counters['accepted']+=1
        return {'accepted':True,'reason':'','version_before':before,'version_after':state['version'],'key':key}

    def graph(self,key):
        state=self.states[key];nodes={node:i for i,node in enumerate(state['nodes'])}
        edges=np.array([(nodes[a],nodes[b]) for a,b,_,_ in state['edges']],dtype=np.int64).reshape(-1,2).T
        n=len(nodes)
        incoming=np.bincount(edges[1],minlength=n);outgoing=np.bincount(edges[0],minlength=n)
        x=np.stack([np.log1p(incoming),np.log1p(outgoing),np.log1p(incoming+outgoing)],axis=1).astype(np.float32)
        return Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edges.copy()),num_nodes=n)

    def payload_bytes(self):
        # Conservative fixed-width logical payload budget, not Python object RSS.
        return len(self.states)*(64+128+32)+self.total_edges*(128*3+8)+self.total_nodes*128

    def invariants(self):
        return (len(self.states)<=self.capacity and self.total_edges==sum(len(s['edges']) for s in self.states.values())
                and self.total_nodes==sum(len(s['nodes']) for s in self.states.values())
                and all(len(s['edges'])<=self.max_edges and len(s['nodes'])<=self.max_nodes for s in self.states.values()))


class BoundedHotCache:
    def __init__(self,capacity=10000,byte_cap=64*2**20):
        if min(capacity,byte_cap)<1:raise ValueError('invalid cache limits')
        self.capacity=capacity;self.byte_cap=byte_cap;self.values=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0;self.evictions=0

    def get(self,key,loader):
        if key in self.values:
            self.hits+=1;self.values.move_to_end(key);return self.values[key]
        self.misses+=1;value=np.asarray(loader(),np.float32).copy()
        if value.nbytes>self.byte_cap:return value
        while self.values and (len(self.values)>=self.capacity or self.bytes+value.nbytes>self.byte_cap):
            _,old=self.values.popitem(last=False);self.bytes-=old.nbytes;self.evictions+=1
        self.values[key]=value;self.bytes+=value.nbytes;return value


class BoundedQueue:
    """Caller retries rejected enqueues; this queue never silently drops input."""
    def __init__(self,capacity=512):
        if capacity<1:raise ValueError('invalid queue capacity')
        self.capacity=capacity;self.items=deque();self.retries=0;self.high_water=0
    def put(self,event):
        if len(self.items)>=self.capacity:self.retries+=1;return False
        self.items.append((time.perf_counter(),event));self.high_water=max(self.high_water,len(self.items));return True
    def pop(self):
        start,event=self.items.popleft();return event,(time.perf_counter()-start)*1000


def durable_checkpoint(path,payload,*,fail_at=None):
    """Crash fixtures use abrupt process exit; not a power-loss guarantee."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.pending')
    if fail_at=='before':os._exit(91)
    encoded=pickle.dumps(payload,protocol=5)
    with tmp.open('wb') as handle:
        half=len(encoded)//2;handle.write(encoded[:half]);handle.flush();os.fsync(handle.fileno())
        if fail_at=='during':os._exit(92)
        handle.write(encoded[half:]);handle.flush();os.fsync(handle.fileno())
    os.replace(tmp,path)
    directory=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(directory)
    finally:os.close(directory)
    if fail_at=='after':os._exit(93)
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint(path,expected_identity):
    # Only our own trusted local checkpoints; pickle is not an untrusted input API.
    with Path(path).open('rb') as handle:payload=pickle.load(handle)
    if payload['identity']!=expected_identity:raise ValueError('checkpoint identity mismatch')
    return payload


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if state['cuda']:torch.cuda.set_rng_state_all(state['cuda'])
