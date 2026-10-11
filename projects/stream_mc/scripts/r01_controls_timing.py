"""Matched held-out input inference cost for light and local-GATv2 controls."""
from __future__ import annotations
import time
from r01_common import ROOT,config,output,read,write,sha256
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from sklearn.preprocessing import StandardScaler
from r01_train import load_cache,subset,degree_features
from gog_fraud.streaming.selective import DenseClassifier,local_predict,calibrated,binary_metrics,seed_all
from gog_fraud.streaming.selective_engine import SelectivePredictor


def main():
    cfg=config();out=output();cache=load_cache(cfg);graphs,meta=subset(cache,cfg['chains'],'test');records=[];traces=[]
    y=np.array([m['label'] for m in meta]);device=torch.device(cfg['device']);torch.set_num_threads(1)
    for seed in cfg['seeds']:
        dest=out/f'models/pooled_snapshot_GIN_seed{seed}';controls=torch.load(dest/'controls.pt',map_location='cpu',weights_only=False)
        scaler=StandardScaler();scaler.mean_=np.asarray(controls['degree_feature_scaler_mean']);scaler.scale_=np.asarray(controls['degree_feature_scaler_scale']);scaler.var_=scaler.scale_**2;scaler.n_features_in_=11;scaler.n_samples_seen_=17020
        models={}
        for name,key,width in [('degree_MLP','degree_mlp',11),('relational_MLP','relational_mlp',130)]:
            model=DenseClassifier(width).to(device);model.load_state_dict(controls[key]);model.eval();models[name]=model
        for name in ['degree_logistic','degree_MLP','relational_MLP','knn_mean','GATv2_local','GATv2_margin']:
            for repeat in range(3):
                seed_all(seed);model_path=out/f"models/pooled_snapshot_{'GATv2' if name.startswith('GATv2') else 'GIN'}_seed{seed}"
                predictor=SelectivePredictor(model_path,cfg['device'],'margin' if name=='GATv2_margin' else 'local',.25)
                local_predict(predictor.local,Batch.from_data_list(graphs[:25]).to(device),1);predictor.synchronize()
                scores=[];labels=[];actual_deep=0;stages={'feature_ms':0.,'local_ms':0.,'retrieval_ms':0.,'head_ms':0.,'relational_ms':0.};total=0.
                fit=read(dest/f'{name}_fit.json') if not name.startswith('GATv2') else predictor.fit
                for start in range(0,len(graphs),128):
                    block=graphs[start:start+128];metadata=meta[start:start+128];predictor.synchronize();begin=time.perf_counter();t=begin;parts={k:0. for k in stages}
                    with torch.inference_mode():
                        if name.startswith('GATv2'):
                            p,h,route,times=predictor.predict_batch(block,[m['event_end'] for m in metadata],[m['sample_id'] for m in metadata]);actual_deep+=int(route.sum())
                            parts.update({k:times[k] for k in ('local_ms','retrieval_ms','relational_ms')})
                        elif name in ('degree_logistic','degree_MLP'):
                            xx=scaler.transform(np.stack([degree_features(g) for g in block])).astype(np.float32)
                            parts['feature_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
                            if name=='degree_logistic':
                                logits=xx@controls['logistic_coef'][0]+controls['logistic_intercept'][0];raw=1/(1+np.exp(-logits))
                            else:raw=torch.sigmoid(models[name](torch.from_numpy(xx).to(device))).cpu().numpy();predictor.synchronize()
                            p=calibrated(raw,fit['calibration']);h=p>=fit['threshold'];parts['head_ms']=(time.perf_counter()-t)*1000
                        else:
                            raw,emb,_=local_predict(predictor.local,Batch.from_data_list(block).to(device),1);predictor.synchronize();parts['local_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter();features=[];means=[]
                            for i,m in enumerate(metadata):
                                star,audit=predictor.index.target_graph(emb[i],float(raw[i]),m['event_end'],m['sample_id'])
                                mean=star.x[:-1].mean(0).numpy() if len(star.x)>1 else np.zeros(65)
                                features.append(np.r_[star.x[-1].numpy(),mean]);means.append(float(mean[-1]) if len(star.x)>1 else float(raw[i]))
                            parts['retrieval_ms']=(time.perf_counter()-t)*1000;t=time.perf_counter()
                            if name=='knn_mean':values=np.array(means)
                            else:values=torch.sigmoid(models[name](torch.from_numpy(np.asarray(features,np.float32)).to(device))).cpu().numpy();predictor.synchronize()
                            p=calibrated(values,fit['calibration']);h=p>=fit['threshold'];parts['head_ms']=(time.perf_counter()-t)*1000
                    wall=(time.perf_counter()-begin)*1000;total+=wall;scores.extend(p);labels.extend(h)
                    for k,v in parts.items():stages[k]+=v
                    traces.append({'seed':seed,'method':name,'repeat':repeat,'batch_start':start,'N':len(block),'wall_ms':wall,**parts})
                baseline=read(model_path/'complete.json');head_params=sum(v.numel() for v in controls['degree_mlp' if name=='degree_MLP' else 'relational_mlp'].values()) if name in models else (controls['logistic_coef'].size+controls['logistic_intercept'].size if name=='degree_logistic' else 0)
                params=baseline['local_training']['parameter_count'] if name.startswith('GATv2') or name in ('knn_mean','relational_MLP') else 0
                if name=='GATv2_margin':params+=baseline['relational_training']['parameter_count']
                if name.startswith('GATv2'):
                    reference=pd.read_parquet(model_path/'test_predictions.parquet');prefix='selective' if name=='GATv2_margin' else 'local'
                else:reference=pd.read_parquet(dest/f'{name}_predictions.parquet');prefix=''
                reference_p=reference[prefix+'_probability' if prefix else 'probability'].to_numpy();reference_h=reference[prefix+'_prediction' if prefix else 'prediction'].to_numpy()
                score_diff=float(np.max(np.abs(np.array(scores)-reference_p)));disagreements=int(np.sum(np.array(labels)!=reference_h))
                if score_diff>1e-6 or disagreements:raise ValueError(f'control pipeline parity failed {name} {score_diff} {disagreements}')
                records.append({'run_id':cfg['run_id'],'experiment_id':'pooled_snapshot','seed':seed,'method':name,'repeat':repeat,'batch_size':128,'N':len(y),'latency_ms_per_contract':total/len(y),
                    'head_parameters':int(head_params),'pipeline_parameters':int(params+head_params),'local_training_seconds':baseline['local_training']['training_seconds'] if params else None,
                    'head_training_seconds':None,'head_training_time_note':'light head training not separately recorded; do not interpret NA as zero',
                    'raw_score_max_abs_diff':score_diff,'prediction_disagreements':disagreements,'deep_fraction':actual_deep/len(y),**stages,**binary_metrics(y,np.array(scores),np.array(labels))})
            print('CONTROL TIMING',seed,name,flush=True)
    pd.DataFrame(records).to_csv(out/'runtime/control_runtime.csv',index=False);pd.DataFrame(traces).to_csv(out/'runtime/control_batch_trace.csv',index=False)
    write(out/'audits/control_parity.json',{'N_runs':len(records),'max_score_difference':max(r['raw_score_max_abs_diff'] for r in records),'prediction_disagreements':sum(r['prediction_disagreements'] for r in records),'raw_trace_sha256':sha256(out/'runtime/control_batch_trace.csv'),'scope':'actual matched batch128 inference; light-control training wall time unavailable'})


if __name__=='__main__':main()
