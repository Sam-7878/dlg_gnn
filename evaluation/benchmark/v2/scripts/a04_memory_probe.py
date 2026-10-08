#!/usr/bin/env python3
"""Fresh-process selected full/cap VRAM measurement, never a modeled matrix."""
import argparse,hashlib,json,sys,time,traceback,uuid
from pathlib import Path
import numpy as np
import psutil,torch,yaml
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.pipelines.run_sci_round4c import _datasets,_models,_instantiate
from gog_fraud.models.pygod.shared_reconstruction import SharedDLGFull
from evaluation.benchmark.v2.scripts.a03_run_crypto_production import load_gog_graph
OUT=ROOT/'evaluation/benchmark/v2/paper_ready_final/publication_evidence_a04/memory_raw'
OUT.mkdir(parents=True,exist_ok=True)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def tensor_sha(tensor):
    array=tensor.detach().cpu().contiguous().numpy()
    digest=hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['Cora-Syn','DGraphFin','Ethereum','Reddit-Syn']);
    ap.add_argument('--model',required=True,choices=['DOMINANT','DLG-Aug','DLG-Base']);ap.add_argument('--envelope',required=True,choices=['full','cap8g']);args=ap.parse_args()
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('cuda:0 must be RTX 3090')
    physical=torch.cuda.get_device_properties(0).total_memory
    cap=8*2**30 if args.envelope=='cap8g' else None
    if cap:torch.cuda.set_per_process_memory_fraction(cap/physical,0)
    cfg=yaml.safe_load((ROOT/'configs/benchmark/sci_round5_final.yaml').read_text())
    cfg['data']['root']='/mnt/d/_Work/_data/DLG'
    seed_everything(42,deterministic=True)
    ds=args.dataset
    if ds=='Ethereum':
        data=load_gog_graph('ethereum');
        if args.model!='DLG-Aug':raise RuntimeError('Ethereum selected only for DLG-Aug')
        det=SharedDLGFull(epoch=30,l1_epochs=20,gpu=0,verbose=0,batch_size=0,
                          message_backend='sparse_fused',reconstruction_backend='exact_sparse',
                          gradient_checkpointing=False,score_chunk_size=8192)
    else:
        internal=ds.removesuffix('-Syn');data=_datasets(cfg)[internal]()
        det=_instantiate(cfg,args.model,_models(cfg)[args.model],0)
    manifest={row['dataset_id']:row for row in json.loads((ROOT/'evaluation/benchmark/v2/paper_ready_final/dataset_manifest_canonical.json').read_text())}[ds]
    input_hashes={'feature_hash':tensor_sha(data.x),'edge_hash':tensor_sha(data.edge_index),'label_hash':tensor_sha(data.y)}
    if any(input_hashes[key]!=manifest[key] for key in input_hashes):
        raise RuntimeError(f'memory probe input differs from canonical manifest: {ds}')
    record={'run_id':str(uuid.uuid4()),'dataset':ds,'model':args.model,'seed':42,
            'envelope':args.envelope,'device':torch.cuda.get_device_name(0),
            'physical_vram_bytes':physical,'allocator_cap_bytes':cap,
            'allocator_fraction':cap/physical if cap else None,
            'graph_nodes':data.num_nodes,'graph_edges':data.edge_index.size(1),'feature_dimension':data.x.size(1),
            'dataset_tensor_sha256':manifest['constructed_tensor_sha256'],'input_hashes':input_hashes,
            'environment_lock_sha256':sha(ROOT/'environment/locks/benchmark-a04-cuda.lock.txt'),
            'runner_source_sha256':sha(Path(__file__)),'status':None}
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    proc=psutil.Process();rss_before=proc.memory_info().rss
    start=time.perf_counter()
    try:
        det.fit(data)
        record['train_time_sec']=time.perf_counter()-start
        start_score=time.perf_counter();scores=det.decision_function(data)
        scores=scores.detach().cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        torch.cuda.synchronize(0)
        record['score_time_sec']=time.perf_counter()-start_score
        record['raw_score_sha256']=hashlib.sha256(np.asarray(scores).tobytes()).hexdigest()
        record['status']='SUPPORTED_EXACT'
    except Exception as exc:
        record['status']='UNSUPPORTED_RESOURCE_OOM' if isinstance(exc,torch.cuda.OutOfMemoryError) else 'UNSUPPORTED_EXECUTION_ERROR'
        record['error_type']=type(exc).__name__;record['error_message']=str(exc);record['traceback']=traceback.format_exc(limit=8)
    record['total_wall_sec']=time.perf_counter()-start
    record['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
    record['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
    record['rss_before_bytes']=rss_before;record['rss_after_bytes']=proc.memory_info().rss
    path=OUT/f'{ds}__{args.model}__{args.envelope}__seed42.json'
    path.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'path':str(path.relative_to(ROOT)),'status':record['status'],'peak_allocated_bytes':record['peak_allocated_bytes'],'wall_sec':record['total_wall_sec']}))
    if record['status']!='SUPPORTED_EXACT':sys.exit(2)

if __name__=='__main__':main()
