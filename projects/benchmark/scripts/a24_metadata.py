"""Generate A24 graph metadata from reconstruction identities, never old hybrid rows."""
from a24_common import *
import numpy as np
proof=json.loads((PUB/'a24_repo_manuscript_consistency.json').read_text());by={r['dataset']:r for r in proof['datasets']}
rr=rows(ROOT/'projects/benchmark/publication/a23/dataset_graph_identity_canonical.csv')
for r in rr:
 n=r['dataset_id']
 if n not in ['Ethereum','BSC','Polygon']:continue
 i=by[n]
 with np.load(ROOT/i['source_path'],allow_pickle=False) as z: positives=int(z['y'].sum()) if 'y' in z else int(z['labels'].sum())
 for k,v in [('raw_upstream_object_count',i['raw_members_count']),('raw_upstream_unit','raw contract transaction histories'),('constructed_graph_N',i['N']),('model_input_N',i['N']),('constructed_graph_E',i['E']),('model_input_E',i['E']),('labeled_evaluation_population_N',i['N']),('feature_hash',i['feature_hash']),('edge_hash',i['edge_hash']),('label_hash',i['label_hash'])]:r[k]=v
 r['node_count_semantics']=i['relation'];r['label_source_and_prediction_unit']=f"source/native contract label / smart_contract_node ({positives} positives across {i['N']} contracts)"
 r['directed_out_degree_E_over_N']=i['E']/i['N'];r['undirected_degree_equiv_2E_over_N']='not applicable: E already stores both directions'
 r['degree_convention']='symmetric edge_index; E/N is undirected mean degree'
 r['builder_commit']='see A24 source hash manifest and A08 input manifest hash '+i['input_manifest_sha256']
 r['split_hash']=';'.join(f"{s}:{m['test_mask']}" for s,m in i['split_hashes'].items())
writecsv(PUB/'dataset_graph_identity_a24.csv',rr)
print('Corrected complete crypto metadata from score-bound input identities')
