"""Verify retained labels against the provider, without assuming cache lineage."""
import csv
import sys
import zipfile
from collections import Counter
from pathlib import Path
from r01_common import ROOT,config,output,write,sha256
import pandas as pd
import torch


def main():
    cfg=config();out=output();source=Path('/mnt/d/_Work/_data/gog_round7_upstream/Token Data')
    labels=pd.read_csv(source/'labels.csv');mapping={(r.Chain.lower(),r.Contract.lower()):int(r.Category==0) for r in labels.itertuples()}
    cache=torch.load(ROOT/cfg['graph_cache'],weights_only=False,map_location='cpu');rows=[];unknown=[];mismatch=[];overlap=[]
    identities={}
    for split,metadata in cache['metadata'].items():
        for m,g in zip(metadata,cache['graphs'][split]):
            key=(m['chain'].lower(),m['contract_id'].lower());native=mapping.get(key)
            identity=':'.join(key)
            if identity in identities:overlap.append({'contract_hash':__import__('hashlib').sha256(identity.encode()).hexdigest(),'first_split':identities[identity],'other_split':split})
            identities[identity]=split
            if native is None:unknown.append(m['sample_id'])
            elif native!=m['label']:mismatch.append(m['sample_id'])
            rows.append({'split':split,'chain':m['chain'],'label':m['label'],'cutoff':m['event_end'],'nodes':g.num_nodes,'edges':g.num_edges})
    frame=pd.DataFrame(rows);support=frame.groupby(['split','chain']).agg(N=('label','size'),N_positive=('label','sum'),min_cutoff=('cutoff','min'),max_cutoff=('cutoff','max'),max_nodes=('nodes','max'),max_edges=('edges','max')).reset_index()
    support.to_csv(out/'split_support.csv',index=False)
    raw=pd.read_parquet(ROOT/cfg['raw_events']);write(out/'raw_event_schema.json',{'columns':list(raw.columns),'N':len(raw),'first_record':raw.iloc[0].to_dict()})
    upstream={}
    for chain in cfg['chains']:
        path=source/'transactions'/f'{chain}.zip'
        with zipfile.ZipFile(path) as archive:
            members=[m for m in archive.namelist() if m.endswith('.csv')];first=members[0]
            with archive.open(first) as handle:header=handle.readline().decode(errors='replace').strip()
        upstream[chain]={'path':str(path),'sha256':sha256(path),'csv_members':len(members),'first_header':header}
    report={'scope':'retained retrospective bounded contract snapshots; not a new source reconstruction',
        'cache_sha256':sha256(ROOT/cfg['graph_cache']),'provider_labels_sha256':sha256(source/'labels.csv'),
        'provider_mapping':'Category 0 -> fraud-positive 1; other provider categories -> negative under provider convention, not proven benign',
        'provider_label_missing':len(unknown),'provider_label_mismatch':len(mismatch),'duplicate_contracts_between_splits':len(overlap),
        'edge_timestamp_available_in_cache':False,'edge_lineage_exact_reconstruction_verified':False,'label_available_time':'unknown',
        'strict_online_eligible':False,'raw_hybrid_graph_consumed':False,'upstream':upstream}
    write(out/'data_audit.json',report);write(out/'split_overlap.json',overlap)
    print(report,flush=True);print('RAW EVENT SCHEMA',list(raw.columns),raw.iloc[0].to_dict(),flush=True)
    if mismatch or unknown or overlap:raise SystemExit('Data identity/label audit failed; do not proceed with this cache')


if __name__=='__main__':main()
