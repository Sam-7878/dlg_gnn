"""Diagnose selected/full batching drift before setting an explicit score guard."""
from r01_common import config, output, read, write
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from r01_train import apply_fitted, load_cache, subset
from gog_fraud.streaming.selective import calibrated, fuse, seed_all, local_predict
from gog_fraud.streaming.selective_engine import SelectivePredictor


def main():
    out=output();cfg=config();rows=[];worst=None;initial_threads=torch.get_num_threads()
    for directory in sorted((out/'models').glob('pooled_snapshot_GIN_seed*')):
        fit=read(directory/'policy_fit.json');frame=pd.read_parquet(directory/'test_predictions.parquet')
        with np.load(directory/'test_features.npz',allow_pickle=False) as arrays:
            benefit=arrays['benefit'];embedding=arrays['embedding']
        for policy in fit['policies']:
            seed=int(directory.name.split('seed')[1]);name=f"seed{seed}_{policy['family']}_q{policy['budget']}_random{policy['random_repeat']}.npz"
            with np.load(out/'runtime/budget_predictions'/name,allow_pickle=False) as actual:
                score,label,route=apply_fitted(frame.raw_local_probability.to_numpy(),frame.counterfactual_full_raw_probability.to_numpy(),benefit,fit,policy=policy,eligible=frame.reference_count.to_numpy()>0)
                errors=np.abs(score-actual['probability']);ix=int(errors.argmax())
                rows.append({'cell':name,'max_abs_score_difference':float(errors[ix]),'N_over_1e6':int((errors>1e-6).sum()),'prediction_disagreements':int((label!=actual['prediction']).sum()),'route_disagreements':int((route!=actual['escalated']).sum())})
                if worst is None or errors[ix]>worst['difference']:
                    worst={'difference':float(errors[ix]),'ix':ix,'directory':directory,'fit':fit,'frame':frame,'embedding':embedding.copy(),'policy':policy,'route':route,'actual_probability':float(actual['probability'][ix]),'expected_probability':float(score[ix])}
    assert len(rows)==180
    pd.DataFrame(rows).to_csv(out/'audits/floating_batch_comparison.csv',index=False)
    w=worst;ix=w['ix'];frame=w['frame'];seed_all(int(w['directory'].name.split('seed')[1]))
    p=SelectivePredictor(w['directory'],cfg['device'],w['policy']['family'],w['policy']['budget'],w['policy']['random_repeat'])
    full_ix=list(range(ix//256*256,min(ix//256*256+256,len(frame))))
    selected_ix=[i for i in range(ix//128*128,min(ix//128*128+128,len(frame))) if w['route'][i]]
    def compute(indices):
        stars=[p.index.target_graph(w['embedding'][i],float(frame.raw_local_probability.iloc[i]),int(frame.snapshot_cutoff.iloc[i]),frame.sample_id.iloc[i])[0] for i in indices]
        with torch.inference_mode():return float(torch.sigmoid(p.deep(Batch.from_data_list(stars).to(p.device))).cpu().numpy()[indices.index(ix)])
    full=compute(full_ix);selected=compute(selected_ix);saved=float(frame.counterfactual_full_raw_probability.iloc[ix]);local=float(frame.local_probability.iloc[ix])
    composed=float(fuse(np.array([local]),calibrated(np.array([selected]),w['fit']['deep_map']),w['fit']['weight'])[0])
    cache=load_cache(cfg);graphs,metadata=subset(cache,cfg['chains'],'test');start=ix//128*128;end=min(start+128,len(frame))
    trials=[]
    for deterministic in (False,True):
        torch.use_deterministic_algorithms(deterministic);torch.set_num_threads(1 if deterministic else initial_threads)
        for repeat in range(3):
            p.random=np.random.default_rng(w['policy']['random_seed']+10**6)
            if w['policy']['family']=='random':p.random.random(start)
            values,labels,routes,times=p.predict_batch(graphs[start:end],[m['event_end'] for m in metadata[start:end]],[m['sample_id'] for m in metadata[start:end]])
            with torch.inference_mode():raw,embedding,_=local_predict(p.local,Batch.from_data_list(graphs[start:end]).to(p.device),1)
            trials.append({'deterministic_algorithms':deterministic,'torch_threads':torch.get_num_threads(),'repeat':repeat,
                'score':float(values[ix-start]),'difference_from_expected':abs(float(values[ix-start])-w['expected_probability']),
                'local_raw_block_difference':float(np.max(np.abs(raw-frame.raw_local_probability.iloc[start:end].to_numpy()))),
                'local_embedding_block_difference':float(np.max(np.abs(embedding-w['embedding'][start:end]))),'target_route':bool(routes[ix-start])})
    report={'N_cells':180,'global_max_score_difference':max(r['max_abs_score_difference'] for r in rows),'N_cells_over_initial_1e6_guard':sum(r['max_abs_score_difference']>1e-6 for r in rows),
        'prediction_disagreements':sum(r['prediction_disagreements'] for r in rows),'route_disagreements':sum(r['route_disagreements'] for r in rows),
        'worst_case':{'cell':next(r['cell'] for r in rows if r['max_abs_score_difference']==w['difference']),'row_index':ix,'full_forward_batch_size':len(full_ix),'selected_forward_batch_size':len(selected_ix),
            'saved_full_raw_probability':saved,'reexecuted_full_raw_probability':full,'reexecuted_selected_raw_probability':selected,'raw_batch_difference':abs(full-selected),
            'full_to_saved_raw_difference':abs(full-saved),'recomposed_selected_to_actual_score_difference':abs(composed-w['actual_probability']),
            'expected_score':w['expected_probability'],'actual_selected_score':w['actual_probability'],'final_threshold':w['policy']['final_threshold']},
        'fresh_execution_trials':trials,
        'source_audit':'v1 offline() did not call seed_all(), unlike train_one() and replay(); deterministic flags/thread setting were not enforced in that process',
        'status':'diagnostic observations; initial guard failure preserved, not an automatic PASS or performance equivalence assertion',
        'scope':'same mathematical model/fit; audit local embedding/retrieval sensitivity to runtime determinism, not assumed just full/selected batch arithmetic'}
    write(out/'audits/floating_batch_audit.json',report);print(report,flush=True)


if __name__=='__main__':main()
