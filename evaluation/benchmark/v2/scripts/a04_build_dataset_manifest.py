#!/usr/bin/env python3
"""Inspect exact available model inputs; mark unverified historical inputs explicitly."""
from __future__ import annotations
import csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/'evaluation/benchmark/v2'
OUT=V2/'paper_ready_final'
OUT.mkdir(parents=True,exist_ok=True)
DATA=Path('/mnt/d/_Work/_data/DLG')
GOG=Path('/mnt/d/_Work/_data/GoG')
PRIMARY=['Elliptic','DGraphFin','BitcoinOTC','Ethereum','BSC','Polygon','Yelp-Syn','Amazon-Syn','Reddit-Syn','Flickr-Syn','Cora-Syn','CiteSeer-Syn','PubMed-Syn','LANL-RedTeam']
R5=json.loads((ROOT/'outputs/benchmark/sci_round5_final/manifests/data_freeze.json').read_text())
FREEZE={x['dataset']+'-Syn' if x['dataset'] in {'Yelp','Amazon','Reddit','Flickr','Cora','CiteSeer','PubMed'} else x['dataset']:x for x in R5['datasets']}
A03={x['dataset_name']:x for x in csv.DictReader((V2/'manifests/datasets/dataset_manifests_a03.csv').open())}


def filehash(path):
    if not path or not path.is_file(): return None
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def tensorhash(t):
    a=t.detach().cpu().contiguous().numpy()
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()


def construct_hash(row):
    h=hashlib.sha256()
    for key in ('feature_hash','edge_hash','label_hash'):
        h.update(key.encode());h.update(bytes.fromhex(row[key]))
    return h.hexdigest()


def observe(row,data,source,raw_nodes=None,prediction_unit='node',split_policy='UNKNOWN'):
    y=data.y.view(-1).long()
    eval_mask=(y==0)|(y==1)
    row.update({'raw_source_path':str(source),'raw_source_sha256':filehash(source),
        'graph_nodes':int(data.num_nodes),'graph_edges':int(data.edge_index.shape[1]),
        'feature_dimension':int(data.x.shape[1]),'evaluation_nodes':int(eval_mask.sum()),
        'positive_evaluation_nodes':int((y[eval_mask]==1).sum()),
        'prediction_unit':prediction_unit,'directed':'true','feature_hash':tensorhash(data.x),
        'edge_hash':tensorhash(data.edge_index),'label_hash':tensorhash(y),
        'source_graph_nodes':raw_nodes,'split_policy':split_policy,'verification_status':'LOADER_OBSERVED'})
    row['evaluation_positive_rate']=row['positive_evaluation_nodes']/row['evaluation_nodes']
    row['constructed_tensor_sha256']=construct_hash(row)
    if split_policy in {'round5_seeded_node_transductive','see_round5_per_run'}:
        eligible=torch.nonzero(eval_mask,as_tuple=False).flatten().numpy()
        labels=y.numpy()[eligible]
        hashes={key:{} for key in ('train_mask_hash','val_mask_hash','test_mask_hash')}
        for seed in range(42,47):
            val_local,test_local=train_test_split(np.arange(len(eligible)),test_size=0.5,random_state=seed,stratify=labels)
            train=np.ones(data.num_nodes,dtype=np.bool_)
            val=np.zeros(data.num_nodes,dtype=np.bool_);val[eligible[val_local]]=True
            test=np.zeros(data.num_nodes,dtype=np.bool_);test[eligible[test_local]]=True
            for key,mask in [('train_mask_hash',train),('val_mask_hash',val),('test_mask_hash',test)]:
                hashes[key][str(seed)]=tensorhash(torch.from_numpy(mask))
        for key,mapping in hashes.items():row[key]=json.dumps(mapping,sort_keys=True)
    return row


def main():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    builder_hash=filehash(Path(__file__))
    rows=[]
    for name in PRIMARY:
        row={'dataset_id':name,'raw_source_path':None,'raw_source_sha256':None,
            'constructed_artifact_path':None,'constructed_artifact_sha256':None,
            'constructed_tensor_sha256':None,'graph_nodes':None,'graph_edges':None,
            'feature_dimension':None,'evaluation_nodes':None,'positive_evaluation_nodes':None,
            'evaluation_positive_rate':None,'prediction_unit':None,'directed':None,
            'feature_hash':None,'edge_hash':None,'label_hash':None,
            'train_mask_hash':None,'val_mask_hash':None,'test_mask_hash':None,
            'builder_commit':commit,'builder_source_sha256':builder_hash,
            'source_graph_nodes':None,'split_policy':None,
            'label_nature':('synthetic_node_injection_on_real_trust_graph' if name=='BitcoinOTC' else
                            'synthetic_node_injection' if name.endswith('-Syn') else
                            'real_external_label'),
            'verification_status':'NOT_LOADER_VERIFIED','source_evidence':None}
        if name in {'Ethereum','BSC','Polygon'}:
            chain=name.lower();source=GOG/chain/f'{chain}_hybrid_graph.pt'
            raw=torch.load(source,map_location='cpu',weights_only=False)
            from torch_geometric.data import Data
            data=Data(x=torch.from_numpy(raw['embeddings']).float(),
                edge_index=torch.unique(raw['edge_index'].long(),dim=1).contiguous(),
                y=raw['labels'].long().view(-1),num_nodes=int(raw['num_nodes']))
            observe(row,data,source,prediction_unit='smart_contract_node',split_policy='seeded_random_60_20_20')
            # Exactly mirrors a03_run_crypto_production.py; retain every seed's mask identity.
            splits={}
            n=data.num_nodes
            for seed in range(42,47):
                perm=np.random.RandomState(seed).permutation(n)
                tr=np.zeros(n,dtype=np.bool_);va=tr.copy();te=tr.copy()
                a=int(.6*n);b=int(.8*n);tr[perm[:a]]=True;va[perm[a:b]]=True;te[perm[b:]]=True
                splits[str(seed)]={'train_mask_hash':hashlib.sha256(tr.tobytes()).hexdigest(),
                    'val_mask_hash':hashlib.sha256(va.tobytes()).hexdigest(),
                    'test_mask_hash':hashlib.sha256(te.tobytes()).hexdigest(),
                    'test_nodes':int(te.sum()),'test_positive_nodes':int(((data.y.numpy()==1)&te).sum())}
            for mask_name in ('train_mask_hash','val_mask_hash','test_mask_hash'):
                row[mask_name]=json.dumps({seed:s[mask_name] for seed,s in splits.items()},sort_keys=True)
            row['source_evidence']='a03_run_crypto_production.load_gog_graph + seed split'
        elif name=='DGraphFin':
            sys.path.insert(0,str(ROOT/'src'))
            from gog_fraud.data.dgraphfin_aligned import load_dgraphfin_aligned
            source=DATA/'DGraphFin/dgraphfin.npz'
            data=load_dgraphfin_aligned(source)
            observe(row,data,source,raw_nodes=3700550,prediction_unit='loan_account_node',split_policy='official_random_70_15_15')
            for key,mask in [('train_mask_hash',data.train_mask),('val_mask_hash',data.val_mask),('test_mask_hash',data.test_mask)]:
                row[key]=tensorhash(mask)
            row['evaluation_nodes']=int(data.eval_mask.sum())
            row['positive_evaluation_nodes']=int((data.y[data.eval_mask]==1).sum())
            row['evaluation_positive_rate']=row['positive_evaluation_nodes']/row['evaluation_nodes']
            row['source_evidence']='dgraphfin_aligned.load_dgraphfin_aligned; official val/test masks'
        elif name=='Elliptic':
            sys.path.insert(0,str(ROOT))
            from scripts import benchmark_8x10_pipeline as legacy
            legacy.DATA_ROOT=str(DATA)
            data=legacy.load_elliptic()
            source=DATA/'Elliptic/raw/elliptic_txs_classes.csv'
            observe(row,data,source,raw_nodes=203769,prediction_unit='labeled_transaction_node',split_policy='see_round5_per_run')
            row['source_evidence']='benchmark_8x10_pipeline.load_elliptic (unknown labels removed)'
        elif name=='LANL-RedTeam':
            source=ROOT/'outputs/benchmark/sci_defense_extension_real/graphs/lanl_graph.pt'
            data=torch.load(source,map_location='cpu',weights_only=False)
            observe(row,data,source,prediction_unit='destination_computer_node',split_policy='stratified_node_transductive')
            row['constructed_artifact_sha256']=filehash(source)
            # The defense runner's 20/20 stratified node split; test support is
            # checked against the preserved per-run LANL JSON records.
            labels=data.y.numpy()
            masks={key:{} for key in ('train_mask_hash','val_mask_hash','test_mask_hash')}
            for seed in range(42,47):
                rng=np.random.RandomState(seed)
                positives=np.flatnonzero(labels==1);negatives=np.flatnonzero(labels==0)
                rng.shuffle(positives);rng.shuffle(negatives)
                p=int(len(positives)*.2);q=int(len(negatives)*.2)
                val=np.concatenate((positives[:p],negatives[:q]))
                test=np.concatenate((positives[p:2*p],negatives[q:2*q]))
                train=np.concatenate((positives[2*p:],negatives[2*q:]))
                for key,indices in (('train_mask_hash',train),('val_mask_hash',val),('test_mask_hash',test)):
                    mask=np.zeros(data.num_nodes,dtype=np.bool_);mask[indices]=True
                    masks[key][str(seed)]=tensorhash(torch.from_numpy(mask))
                sample=ROOT/f'outputs/benchmark/sci_defense_extension_real/benchmark/raw/LANL-RedTeam__DLG-Aug__seed{seed}.json'
                original=json.loads(sample.read_text())
                if len(test)!=original['n_test_samples'] or int(labels[test].sum())!=original['n_test_positives']:
                    raise RuntimeError(f'LANL split support differs from raw run: seed {seed}')
            for key,mapping in masks.items():row[key]=json.dumps(mapping,sort_keys=True)
            row['source_evidence']='lanl_graph.pt + lanl_ground_truth_freeze.json; 749 red-team events map to 301 nodes'
        else:
            # Rebuild seven deterministic injection graphs from the actual round5 loader.
            sys.path.insert(0,str(ROOT))
            from scripts import benchmark_8x10_pipeline as legacy
            legacy.DATA_ROOT=str(DATA);legacy.DATASET_SEED=42
            short=name.removesuffix('-Syn')
            loaders={'Yelp':legacy.load_yelp,'Amazon':legacy.load_amazon,
                     'Reddit':legacy.load_reddit,'Flickr':legacy.load_flickr,
                     'Cora':lambda:legacy.load_planetoid('Cora'),
                     'CiteSeer':lambda:legacy.load_planetoid('CiteSeer'),
                     'PubMed':lambda:legacy.load_planetoid('PubMed'),
                     'BitcoinOTC':legacy.load_bitcoin_otc}
            paths={'Yelp':DATA/'Yelp/processed/data.pt','Amazon':DATA/'Amazon/Computers/processed/data.pt',
                   'Reddit':DATA/'Reddit/processed/data.pt','Flickr':DATA/'Flickr/processed/data.pt',
                   'Cora':DATA/'Cora/Cora/processed/data.pt','CiteSeer':DATA/'CiteSeer/CiteSeer/processed/data.pt',
                   'PubMed':DATA/'PubMed/PubMed/processed/data.pt',
                   'BitcoinOTC':DATA/'BitcoinOTC/processed/data.pt'}
            try:
                from gog_fraud.evaluation.reproducibility import seed_everything
                seed_everything(42,deterministic=True)
                data=loaders[short]()
                observe(row,data,paths[short],prediction_unit='node',split_policy='round5_seeded_node_transductive')
                row['source_evidence']='benchmark_8x10_pipeline.load_'+short.lower()+' + fixed injection seed 42'
                old=FREEZE.get(name)
                if old and any(row[key]!=old[key] for key in ('feature_hash','edge_hash','label_hash')):
                    row['verification_status']='LOADER_MISMATCH_ROUND5'
                    row['source_evidence']+='; differs from archived round5 data_freeze.json'
                elif old:
                    row['source_evidence']+='; all 3 tensor hashes match round5 data_freeze.json'
            except Exception as exc:
                row['verification_status']='LOADER_ERROR'
                row['source_evidence']=f'{type(exc).__name__}: {exc}'

        if row['verification_status']!='LOADER_ERROR':
            if name=='LANL-RedTeam':
                artifact=source
            else:
                artifact=ROOT/'outputs/benchmark/a04_constructed_graphs'/f'{name}.pt'
                artifact.parent.mkdir(parents=True,exist_ok=True)
                payload={'dataset_id':name,'x':data.x.detach().cpu(),
                    'edge_index':data.edge_index.detach().cpu(),
                    'y':data.y.detach().cpu(),'num_nodes':int(data.num_nodes)}
                for mask in ('train_mask','val_mask','test_mask','eval_mask'):
                    value=getattr(data,mask,None)
                    if isinstance(value,torch.Tensor):payload[mask]=value.detach().cpu()
                torch.save(payload,artifact)
            row['constructed_artifact_path']=str(artifact.relative_to(ROOT))
            row['constructed_artifact_sha256']=filehash(artifact)
        rows.append(row)
    path=OUT/'dataset_manifest_canonical.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'dataset_manifest_canonical.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps([{'dataset':r['dataset_id'],'nodes':r['graph_nodes'],'features':r['feature_dimension'],
        'positive':r['positive_evaluation_nodes'],'status':r['verification_status']} for r in rows],indent=2))

if __name__=='__main__':main()
