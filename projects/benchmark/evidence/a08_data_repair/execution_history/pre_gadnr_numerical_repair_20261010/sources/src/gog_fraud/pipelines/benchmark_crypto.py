"""Versioned A08 training and raw-score evidence, using existing detectors."""
from __future__ import annotations
import hashlib, inspect, json, os, platform, random, re, resource, shutil, subprocess, time, types
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from pygod.detector import CoLA, OCGNN
from gog_fraud.models.pygod.gadnr import GADNR
from gog_fraud.models.pygod.shared_reconstruction import SharedDOMINANT, SharedAnomalyDAE, SharedDLGBase, SharedDLGFull
from gog_fraud.data.crypto_raw import array_hash, sha256
from gog_fraud.data.benchmark_node_adapter import load_frozen_node_graph
from gog_fraud.data.benchmark_metrics import evaluate_scores, validate_score
from gog_fraud.data.benchmark_lineage import expected_run_identity, execution_sources

FACTORIES={'DOMINANT':SharedDOMINANT,'AnomalyDAE':SharedAnomalyDAE,'DLG-Base':SharedDLGBase,'DLG-Aug':SharedDLGFull,'CoLA':CoLA,'OCGNN':OCGNN,'GADNR':GADNR}


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def factory(model, n, epochs, local_epochs=20, gpu=0):
    cfg={'epoch':epochs,'gpu':gpu,'verbose':0,'batch_size':64 if n>5000 else 32}
    if model in ('DOMINANT','AnomalyDAE','DLG-Base','DLG-Aug'):cfg['batch_size']=0
    if model=='DOMINANT':cfg['weight']=.5
    if model=='AnomalyDAE':cfg['score_chunk_size']=256
    if model=='DLG-Aug':cfg['l1_epochs']=local_epochs
    det=FACTORIES[model](**cfg)
    # Record inherited constructor defaults actually present, not just caller kwargs.
    scalar={k:v for k,v in vars(det).items() if isinstance(v,(str,int,float,bool)) or v is None}
    callables={k:v.__module__+'.'+v.__qualname__ for k,v in vars(det).items()
               if (inspect.isfunction(v) or inspect.isclass(v)) and hasattr(v,'__qualname__')}
    signature=re.sub(r' at 0x[0-9a-fA-F]+','',str(inspect.signature(FACTORIES[model])))
    resolved={'caller':cfg,'scalar_attributes':scalar,'callable_attributes':callables,'constructor_signature':signature}
    return det,resolved


def train_run(root, config, chain, model, seed, *, smoke=False, smoke_n=0):
    root=Path(root);local=root/'local_storage/benchmark/a08_data_repair';evidence=root/'projects/benchmark/evidence/a08_data_repair'
    if not torch.cuda.is_available() or '3090' not in torch.cuda.get_device_name(0):raise RuntimeError('qualified RTX3090 CUDA device required')
    torch.set_num_threads(4)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed);random.seed(seed)
    dataset_id=chain+'_contract_clean_v1';manifest=local/'frozen'/chain/'input_manifest.json'
    data=load_frozen_node_graph(manifest,dataset_id,seed)
    if smoke_n:
        # Real input subset, explicit smoke identity; preserve labels/ID ordering.
        n=min(smoke_n,data.num_nodes);allowed=(data.edge_index<n).all(dim=0)
        data=Data(x=data.x[:n].clone(),edge_index=data.edge_index[:,allowed].clone(),y=data.y[:n].clone(),num_nodes=n,
                  node_ids=data.node_ids[:n],val_mask=data.val_mask[:n],test_mask=data.test_mask[:n],
                  input_manifest_hash=data.input_manifest_hash,split_hash=data.split_hash)
    epochs=1 if smoke else config['epochs'][chain]
    local_epochs=1 if smoke else config['local_epochs']
    det,resolved=factory(model,data.num_nodes,epochs,local_epochs)
    config_hash=hashlib.sha256(json.dumps(resolved,sort_keys=True).encode()).hexdigest()
    identity=expected_run_identity(root,config,chain,model,seed,smoke=smoke)
    if smoke_n:raise ValueError('qualified campaign uses the whole actual chain, not a subset')
    if resolved!=identity['resolved_model_config'] or config_hash!=identity['model_config_hash']:
        raise ValueError('actual detector settings differ from frozen settings')
    if data.input_manifest_hash!=identity['input_manifest_hash'] or data.split_hash!=identity['split_hash']:
        raise ValueError('actual model input differs from frozen identity')
    stage='smoke' if smoke else 'runs';run_id=f'{dataset_id}__{model}__seed{seed}'
    out=local/stage/run_id;record_path=out/'run_manifest.json'
    if record_path.exists():raise FileExistsError('immutable run record already exists: '+str(record_path))
    if out.exists() and any(out.iterdir()):
        previous=local/'attempt_history'/stage/run_id/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        previous.parent.mkdir(parents=True,exist_ok=True);shutil.move(out,previous)
        dump(previous/'interruption_disposition.json',{'status':'INCOMPLETE_INTERRUPTED_ATTEMPT','no_final_record':True,
             'rerun_policy':'fresh initialization under unchanged frozen settings; no checkpoint or score reuse',
             'counted_as_resource_failure':False})
    out.mkdir(parents=True,exist_ok=True)
    training=Data(x=data.x.clone(),edge_index=data.edge_index.clone(),num_nodes=data.num_nodes)
    # Neither labels nor masks are supplied to unsupervised detector.fit.
    code_hashes=execution_sources(root)
    record={'schema_version':1,'run_id':run_id,'project_id':'benchmark','protocol_id':config['protocol_id'],'campaign_id':config['campaign_id'],
            'dataset_id':dataset_id,'model':model,'seed':seed,'prediction_unit':'contract_node','status':'RUNNING','scope':'qualification' if smoke else 'primary',
            'input_manifest_hash':data.input_manifest_hash,'split_hash':data.split_hash,'model_config_hash':config_hash,'resolved_model_config':resolved,
            'execution_manifest_hash':identity['execution_manifest_hash'],
            'planned_global_epochs':epochs,'planned_local_epochs':local_epochs if model=='DLG-Aug' else 0,'resume_segments':[],
            'code_hashes':code_hashes,'scientific_source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
            'dirty_patch_hash':hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=root)).hexdigest(),
            'interpreter':str(Path(os.sys.executable).resolve()),'environment_lock_hash':sha256(root/'environment/locks/benchmark-a03-cuda.lock.txt'),
            'device':torch.cuda.get_device_name(0),'cuda_runtime':torch.version.cuda,'torch':torch.__version__,'os':platform.platform(),
            'node_id_hash':array_hash(np.array(data.node_ids)),'score_orientation':'higher_is_anomalous','start_utc':datetime.now(timezone.utc).isoformat(),
            'original_model_input':{'x':array_hash(training.x.numpy()),'edge_index':array_hash(training.edge_index.numpy())},
            'training_target_access':'none; training Data has no y, masks or node IDs'}
    dump(out/'identity.json',record)
    samples=[];phase=['training']
    if hasattr(det,'forward_model'):
        original=det.forward_model
        def observe(this,*args,**kwargs):
            result=original(*args,**kwargs)
            loss=result[0]
            if not torch.isfinite(loss).all():raise FloatingPointError('nonfinite detector loss')
            samples.append({'phase':phase[0],'loss':float(loss.detach().cpu())})
            return result
        det.forward_model=types.MethodType(observe,det)
    if hasattr(det,'_backward_exact'):
        original_back=det._backward_exact
        def observe_back(this,*args,**kwargs):
            scores=original_back(*args,**kwargs)
            if not torch.isfinite(scores).all():raise FloatingPointError('nonfinite training scores')
            samples.append({'phase':'training','loss':float(scores.mean())})
            return scores
        det._backward_exact=types.MethodType(observe_back,det)
    try:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        det.fit(training)
        phase[0]='inference';scores=det.decision_function(training)
        if torch.is_tensor(scores):scores=scores.detach().cpu().numpy()
        scores=validate_score(scores,data.num_nodes)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        metrics=evaluate_scores(data.y.numpy(),scores,np.array(data.node_ids),data.val_mask.numpy(),data.test_mask.numpy())
        np.savez_compressed(out/'scores.npz',scores=scores,node_ids=np.array(data.node_ids),labels=data.y.numpy(),val_mask=data.val_mask.numpy(),test_mask=data.test_mask.numpy())
        torch.save(det.model.state_dict(),out/'final_model_state.pt')
        if model=='DLG-Aug':
            np.savez_compressed(out/'augmented_input.npz',x=training.x.detach().cpu().numpy(),local_score=training.dlg_l1_score.numpy())
            record['actual_augmented_input_hash']=array_hash(training.x.detach().cpu().numpy())
        dump(out/'metrics.json',metrics);dump(out/'loss_curve.json',{'granularity':'epoch' if hasattr(det,'_backward_exact') else 'forward_model call (minibatch)', 'samples':samples})
        record.update(status='SUPPORTED_EXACT',completed_epochs=getattr(det,'actual_epochs_',epochs),wall_time_seconds=elapsed,
                      raw_scores_path=str(out/'scores.npz'),raw_scores_sha256=sha256(out/'scores.npz'),score_array_hash=array_hash(scores),
                      checkpoint_path=str(out/'final_model_state.pt'),checkpoint_sha256=sha256(out/'final_model_state.pt'),
                      metrics_path=str(out/'metrics.json'),metrics_sha256=sha256(out/'metrics.json'),
                      loss_curve_sha256=sha256(out/'loss_curve.json'),score_min=float(scores.min()),score_max=float(scores.max()),score_std=float(scores.std()),
                      peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except torch.cuda.OutOfMemoryError as e:record.update(status='UNSUPPORTED_RESOURCE_OOM',failure_reason=str(e))
    except FloatingPointError as e:record.update(status='FAILED_NUMERICAL_VALIDATION',failure_reason=str(e))
    except Exception as e:record.update(status='FAILED_RUNTIME_ERROR',failure_reason=repr(e))
    record['end_utc']=datetime.now(timezone.utc).isoformat()
    dump(record_path,record)
    # Public record has hashed identities and metric numbers, no raw addresses.
    public={k:v for k,v in record.items() if k not in ('raw_scores_path','checkpoint_path','metrics_path')}
    if record['status']=='SUPPORTED_EXACT':
        public['metrics']={k:v for k,v in metrics.items() if k!='alert_budgets'}
        public['metrics']['alert_budgets']={q:{k:v for k,v in b.items() if k!='selected_node_ids'} for q,b in metrics['alert_budgets'].items()}
    dump(evidence/('qualification_runs' if smoke else 'run_manifests')/(run_id+'.json'),public)
    print(json.dumps({'run_id':run_id,'status':record['status'],'scope':record['scope'],'epochs':record.get('completed_epochs',0),'seconds':record.get('wall_time_seconds',0)}),flush=True)
    return record['status']
