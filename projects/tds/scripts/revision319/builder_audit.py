"""Independent raw-source reconstruction at last-transfer and first+7d cutoffs.

Uses NumPy lexicographic sorting and chunk value-counts, not R317's graph builder.
Frozen graphs are compared on every element/order/dtype, never rewritten.
"""
import argparse
import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import time
import zipfile

import numpy as np
import pandas as pd
import torch

from common import OUT, PRIVATE, PROJECT, RAW, ORIGINAL, HORIZON, sha, write

INVALID={'','nan','none','null'}


def ordered_tail(frame):
    if len(frame)==0:return frame
    # np.lexsort's last key is primary. Original row position breaks all ties.
    order=np.lexsort((frame.row_position.to_numpy(),frame.transaction_hash.to_numpy(),
                      frame.block_number.to_numpy(),frame.timestamp.to_numpy()))
    return frame.iloc[order[-128:]].reset_index(drop=True)


def degree_update(counter, values):
    for key, value in values.value_counts(sort=False).items():
        counter[key]+=int(value)


def tensors(tail, incoming, outgoing):
    nodes=sorted(set(tail['from'])|set(tail['to']))
    if not nodes:raise RuntimeError('No admitted edge at cutoff')
    mapping={name:i for i,name in enumerate(nodes)}
    inc=torch.tensor([incoming[n] for n in nodes],dtype=torch.float32)
    out=torch.tensor([outgoing[n] for n in nodes],dtype=torch.float32)
    x=torch.stack([inc.log1p(),out.log1p(),(inc+out).log1p()],dim=1)
    edges=torch.tensor([[mapping[n] for n in tail['from']],[mapping[n] for n in tail['to']]],dtype=torch.int64)
    return x,edges


def tensor_hash(x,e):
    h=hashlib.sha256()
    for a in [x,e]:
        h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def chain_audit(chain):
    torch.set_num_threads(1)
    protocol_hash=sha(OUT/'protocol_freeze_319.json')
    path=PRIVATE/f'builder_{chain}.json'
    if path.exists():
        prior=json.loads(path.read_text())
        assert prior['protocol_sha256']==protocol_hash
        assert prior['original_graph_sha256']==sha(ORIGINAL) and prior['horizon_graph_sha256']==sha(HORIZON)
        return prior
    start=time.monotonic()
    source=RAW/'transactions'/f'{chain}.zip'
    source_sha=sha(source)
    expected=json.loads((PROJECT/f'results/317/{chain}_source_identity.json').read_text())
    assert source_sha==expected['archive_sha256']
    frame=pd.read_csv(PROJECT/'private/317/sample_time_lineage.csv')
    horizon=pd.read_csv(PROJECT/'results/317/horizon7_population.csv')
    old=torch.load(ORIGINAL,map_location='cpu',weights_only=False)['graphs']
    new=torch.load(HORIZON,map_location='cpu',weights_only=False)['graphs']
    horizon_by_id={int(i):g for i,g in zip(horizon.sample_id,new)}
    nodes_chain=frame[frame.chain==chain]
    audit=[];counts=[];mismatches=[];raw_total=0
    with zipfile.ZipFile(source) as z:
        members={n.rsplit('/',1)[-1][:-4].lower():n for n in z.namelist() if n.endswith('.csv')}
        for j,r in enumerate(nodes_chain.itertuples(),1):
            cutoffs=[int(r.last_valid_tx_time)]
            if r.row_id in horizon_by_id:cutoffs.append(int(r.first_valid_tx_time)+7*86400)
            inc=[collections.defaultdict(int) for _ in cutoffs]
            out=[collections.defaultdict(int) for _ in cutoffs]
            tails=[None for _ in cutoffs];numbers=[0 for _ in cutoffs];max_t=[-1 for _ in cutoffs]
            raw_n=0
            with z.open(members[r.contract]) as f:
                for chunk in pd.read_csv(f,usecols=['timestamp','block_number','transaction_hash','from','to'],chunksize=100000):
                    chunk['row_position']=np.arange(raw_n,raw_n+len(chunk),dtype=np.int64);raw_n+=len(chunk)
                    for field in ['from','to']:chunk[field]=chunk[field].astype(str).str.strip().str.lower()
                    chunk=chunk[(~chunk['from'].isin(INVALID))&(~chunk['to'].isin(INVALID))]
                    chunk['timestamp']=pd.to_numeric(chunk.timestamp,errors='raise').astype('int64')
                    chunk['block_number']=pd.to_numeric(chunk.block_number,errors='raise').astype('int64')
                    assert chunk.transaction_hash.notna().all(),'Ambiguous missing transaction hash requires explicit ordering audit'
                    for k,cutoff in enumerate(cutoffs):
                        selected=chunk[chunk.timestamp<=cutoff]
                        if len(selected)==0:continue
                        degree_update(inc[k],selected['to']);degree_update(out[k],selected['from'])
                        numbers[k]+=len(selected);max_t[k]=max(max_t[k],int(selected.timestamp.max()))
                        tails[k]=ordered_tail(pd.concat([tails[k],selected],ignore_index=True) if tails[k] is not None else selected)
            raw_total+=raw_n
            assert raw_n==int(r.raw_rows)
            for k,cutoff in enumerate(cutoffs):
                cohort='original' if k==0 else 'horizon7'
                actual_x,actual_e=tensors(tails[k],inc[k],out[k])
                expected_g=old[r.row_id] if k==0 else horizon_by_id[r.row_id]
                expected_x=expected_g['x'] if k==0 else expected_g.x
                expected_e=expected_g['edge_index'] if k==0 else expected_g.edge_index
                expected_label=int(expected_g['label']) if k==0 else int(expected_g.y)
                expected_chain=int(expected_g['chain_index']) if k==0 else int(expected_g.chain_id)
                expected_time=int(expected_g['timestamp']) if k==0 else int(expected_g.timestamp)
                x_equal=actual_x.dtype==expected_x.dtype and torch.equal(actual_x,expected_x)
                e_equal=actual_e.dtype==expected_e.dtype and torch.equal(actual_e,expected_e)
                meta_equal=(int(r.label),int(r.chain_index) if hasattr(r,'chain_index') else {'ethereum':0,'bsc':1,'polygon':2}[chain],cutoff)==(expected_label,expected_chain,expected_time)
                # Actual coding is recovered from original tensors, independently of sample ordering.
                assert expected_chain==int(old[r.row_id]['chain_index'])
                result=dict(sample_id=int(r.row_id),cohort=cohort,chain=chain,x_exact=x_equal,edges_exact=e_equal,
                            metadata_exact=meta_equal,tensor_sha256=tensor_hash(actual_x,actual_e),
                            reference_sha256=tensor_hash(expected_x,expected_e),cutoff=cutoff,max_edge_time=max_t[k])
                audit.append(result)
                if not(x_equal and e_equal and meta_equal):mismatches.append(result)
                assert max_t[k]<=cutoff
                counts.append(dict(sample_id=int(r.row_id),cohort=cohort,chain=chain,label=int(r.label),
                    cutoff=cutoff,max_input_timestamp=max_t[k],retained_nodes=actual_x.shape[0],retained_edges=actual_e.shape[1],
                    full_history_nodes=len(set(inc[k])|set(out[k])),full_history_edges=numbers[k]))
            if j%250==0:print('source319',chain,j,'/',len(nodes_chain),'rows',raw_total,'seconds',round(time.monotonic()-start),flush=True)
    pd.DataFrame(audit).to_csv(OUT/f'builder_{chain}_comparisons.csv',index=False)
    pd.DataFrame(counts).to_csv(OUT/f'cutoff_counts_{chain}.csv',index=False)
    result=dict(chain=chain,status='PASS' if not mismatches else 'FAIL',protocol_sha256=protocol_hash,
                original_graph_sha256=sha(ORIGINAL),horizon_graph_sha256=sha(HORIZON),source_archive_sha256=source_sha,
                contracts=len(nodes_chain),comparisons=len(audit),mismatches=mismatches,raw_rows_scanned=raw_total,
                elapsed_seconds=time.monotonic()-start,source_zip_members_crc_verified=True)
    write(path,result);print('source319 DONE',chain,result['status'],raw_total,flush=True)
    return result


def main():
    assert (OUT/'protocol_freeze_319.json').exists()
    torch.set_num_threads(1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(chain_audit,c) for c in ['ethereum','bsc','polygon']]
        results=[f.result() for f in concurrent.futures.as_completed(jobs)]
    comparisons=pd.concat([pd.read_csv(OUT/f'builder_{r["chain"]}_comparisons.csv') for r in results],ignore_index=True)
    counts=pd.concat([pd.read_csv(OUT/f'cutoff_counts_{r["chain"]}.csv') for r in results],ignore_index=True)
    counts.sort_values(['cohort','sample_id']).to_csv(OUT/'cutoff_admissible_counts.csv',index=False)
    mismatch=sum(len(r['mismatches']) for r in results)
    state=dict(status='PASS' if mismatch==0 else 'FAIL',mismatches=mismatch,
               original_graph_sha256=sha(ORIGINAL),horizon_graph_sha256=sha(HORIZON),
               counts_sha256=sha(OUT/'cutoff_admissible_counts.csv'),
               original_compared=int((comparisons.cohort=='original').sum()),
               horizon_compared=int((comparisons.cohort=='horizon7').sum()),
               raw_transfer_rows_scanned=sum(r['raw_rows_scanned'] for r in results),chains=results,
               all_max_edge_time_admissible=bool((comparisons.max_edge_time<=comparisons.cutoff).all()),
               source='independent NumPy lexsort and chunked degree counts; R317 builder not imported',
               training_permitted=mismatch==0)
    assert state['original_compared']==24316 and state['horizon_compared']==24179
    write(OUT/'seven_day_builder_equivalence.json',state)
    assert mismatch==0,'STOP: builder mismatch; no additional model fits authorized until correction'
    print('ALL SOURCE-TO-TENSOR CHECKS PASS',flush=True)


if __name__=='__main__':main()
