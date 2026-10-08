#!/usr/bin/env python3
"""Read known torch ZIP storage and pickle opcodes without executing pickle.
The decoder deliberately supports only the exact layout of the three frozen
hybrid tensors. Unknown formats fail rather than invoking torch.load/pickle.
"""
import argparse,csv,hashlib,json,pickletools,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
PUB='evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/'


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def decode(path,n,feature=True):
    with zipfile.ZipFile(path) as z:
        meta=z.read(next(t for t in z.namelist() if t.endswith('data.pkl')))
        ops=list(pickletools.genops(meta))
        # Known tensor: persistent storage, offset 0, shape (2,E), stride (1,2).
        start=next(i for i,(o,a,p) in enumerate(ops) if o.name=='BINPERSID')
        ints=[a for o,a,p in ops[start+1:start+11] if o.name in ('BININT','BININT1','BININT2')]
        if len(ints)!=5 or ints[0]!=0 or ints[1]!=2 or ints[3:]!=[1,2]:
            raise ValueError('unsupported edge storage layout')
        edge=np.frombuffer(z.read(next(t for t in z.namelist() if t.endswith('/data/0'))),dtype='<i8').reshape(-1,2).T
        if edge.shape!=(2,ints[2]) or not np.all((edge>=0)&(edge<n)):raise ValueError('edge bounds/layout')
        result={'edges':edge,'codes':np.unique(edge[0]*n+edge[1])}
        if feature:
            values=[a for o,a,p in ops if o.name=='BINUNICODE' and isinstance(a,str) and len(a)==n*8*8]
            if len(values)!=1:raise ValueError('unsupported NumPy serialization')
            result['x']=np.frombuffer(values[0].encode('latin1'),dtype='<f8').reshape(n,8)
            result['y']=np.frombuffer(z.read(next(t for t in z.namelist() if t.endswith('/data/1'))),dtype='<i8')
            if result['y'].shape!=(n,) or not np.isin(result['y'],[0,1]).all():raise ValueError('label encoding')
            marker=next(i for i,(o,a,p) in enumerate(ops) if isinstance(a,str) and a=='contract_to_idx')
            end=next(i for i,(o,a,p) in enumerate(ops) if isinstance(a,str) and a=='idx_to_contract')
            result['json_basenames']=[Path(a).name for o,a,p in ops[marker:end] if o.name=='BINUNICODE' and isinstance(a,str) and '/graphs/' in a]
            if len(result['json_basenames'])!=n:raise ValueError('node map count')
        return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'projects/benchmark/evidence/astra_revision/hybrid_tensor_audit.json')
    args=ap.parse_args()
    with zipfile.ZipFile(ROOT/'projects/benchmark/evidence/public_numeric_evidence.zip') as z:
        manifest=json.loads(z.read(PUB+'dataset_manifest_canonical.json'))
    results=[]
    for dataset in ('Ethereum','BSC','Polygon'):
        row=next(r for r in manifest if r['dataset_id']==dataset);c=dataset.lower();base=args.data_root/c
        p=base/(c+'_hybrid_graph.pt');sha=digest(p)
        if sha!=row['raw_source_sha256']:raise ValueError('not the frozen artifact: '+dataset)
        n=row['graph_nodes'];hybrid=decode(p,n);knn=decode(base/(c+'_knn_graph.pt'),n,feature=False)
        x=hybrid['x'];y=hybrid['y'];codes=hybrid['codes']
        if len(codes)!=row['graph_edges'] or int(y.sum())!=row['positive_evaluation_nodes']:raise ValueError('manifest mismatch')
        extra=np.setdiff1d(codes,knn['codes']);removed=np.setdiff1d(knn['codes'],codes)
        csv_path=args.data_root/'features'/(c+'_basic_metrics_processed.csv')
        csv_rows=list(csv.DictReader(csv_path.open()))
        mapped=np.array([int(csv_rows[int(Path(name).stem)]['label']) for name in hybrid['json_basenames']])
        label_mismatches=int(np.count_nonzero(mapped!=y))
        if label_mismatches:raise ValueError('upstream CSV label alignment mismatch')
        addresses=[csv_rows[int(Path(name).stem)]['Contract'].lower() for name in hybrid['json_basenames']]
        source_labels=list(csv.DictReader((args.data_root/'labels.csv').open()))
        chain_labels={r['Contract'].lower():r['Category'] for r in source_labels if r['Chain'].lower()==c}
        missing_labels=[a for a in addresses if a not in chain_labels]
        category_mismatch=sum(int(y[i]) != int(chain_labels[a]=='0') for i,a in enumerate(addresses) if a in chain_labels)
        directory=args.data_root/'global_graph'
        mapping_path=directory/(c+'_contract_to_number_mapping.json')
        mapping=json.loads(mapping_path.read_text())
        address_to_node={a:i for i,a in enumerate(addresses)}
        upstream_to_node={int(v):address_to_node.get(a.lower()) for a,v in mapping.items()}
        global_path=directory/(c+'_graph_more_than_1_ratio.csv')
        global_codes=[];outside=0;global_rows=0
        with global_path.open() as stream:
            for r in csv.DictReader(stream):
                global_rows+=1
                a=upstream_to_node.get(int(r['Contract1']));b=upstream_to_node.get(int(r['Contract2']))
                if a is None or b is None:outside+=1;continue
                global_codes.extend((a*n+b,b*n+a))
        global_codes=np.unique(np.asarray(global_codes,dtype=np.int64))
        global_audit={'csv_sha256':digest(global_path),'mapping_sha256':digest(mapping_path),'source_rows':global_rows,'rows_with_unretained_endpoint':outside,'retained_directed_edges_with_both_orientations':len(global_codes),'hybrid_overlap':len(np.intersect1d(codes,global_codes)),'added_hybrid_overlap':len(np.intersect1d(extra,global_codes)),'same_label_fraction':float(np.mean(y[global_codes//n]==y[global_codes%n])),'comparison':'Address-aligned, unweighted undirected union of retained CSV edges; no claim about builder causality.'}
        sample=[]
        for i in np.linspace(0,n-1,12,dtype=int):
            path=base/'graphs'/hybrid['json_basenames'][i]
            j=json.loads(path.read_text())
            sample.append(dict(node=int(i),json_basename=path.name,json_sha256=digest(path),tensor_label=int(y[i]),json_label=j.get('label'),feature_column_3=float(x[i,3]),json_feature_rows=len(j.get('features',[]))))
        results.append(dict(dataset=dataset,hybrid_file_sha256=sha,knn_file_sha256=digest(base/(c+'_knn_graph.pt')),n_nodes=n,n_hybrid_edges=len(codes),n_knn_edges=len(knn['codes']),n_added_edges=len(extra),n_removed_knn_edges=len(removed),added_same_label_edges=int(np.count_nonzero(y[extra//n]==y[extra%n])),added_edge_same_label_fraction=float(np.mean(y[extra//n]==y[extra%n])),constant_zero_feature_columns=np.flatnonzero((x==0).all(0)).tolist(),feature_column_min=x.min(0).tolist(),feature_column_max=x.max(0).tolist(),features_equal_to_label_columns=[i for i in range(8) if np.array_equal(x[:,i],y)],upstream_label_alignment={'csv_basename':csv_path.name,'csv_sha256':digest(csv_path),'n_checked':n,'n_mismatches':label_mismatches,'mapping':'contract_to_idx graph JSON basename -> processed CSV row'},labels_csv_alignment={'labels_csv_sha256':digest(args.data_root/'labels.csv'),'category_zero_positive':True,'n_checked':n-len(missing_labels),'missing_contracts':len(missing_labels),'n_mismatches':category_mismatch},global_graph_comparison=global_audit,json_sample=sample,construction_causality='UNVERIFIED: same-label edges observed; original builder required to determine label use'))
    report=dict(status='PROVENANCE_REVIEW_REQUIRED',method='non-executing pickle opcode inspection and typed ZIP storage parsing; known layout only',source_sha256=digest(Path(__file__)),results=results,provided_pipeline_trace={'pipeline':'src/run_evaluation_pipeline.py','input':'existing polygon_hybrid_graph.pt','output':'data/benchmark/gog_microrag_stream_v1/polygon_hybrid_graph.pt','builder':'src/benchmark/semi_synthetic_builder.py','conclusion':'copies existing tensors or generates a 5000x32 simulated fallback; does not produce the three frozen raw 8-feature tensors'},inference_limit='No label-leakage causal conclusion is asserted without the original producer. Native financial-fraud generalization from these artifacts is not cleared.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'datasets':[{k:r[k] for k in ('dataset','n_added_edges','added_edge_same_label_fraction','constant_zero_feature_columns')} for r in results]},indent=2))
if __name__=='__main__':main()
