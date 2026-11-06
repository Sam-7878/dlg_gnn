"""Rebuild one-, three- and fourteen-day cohorts in one source pass."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'revision319'))
from common import PROJECT, RAW, sha
from builder_audit import INVALID, degree_update, ordered_tail, tensors

DAYS=(1,3,14)
END=1709251199
DEST=PROJECT/'private/322/windows'


def scan(chain):
    torch.set_num_threads(1)
    path=RAW/'transactions'/f'{chain}.zip'
    identity=json.loads((PROJECT/f'results/317/{chain}_source_identity.json').read_text())
    assert sha(path)==identity['archive_sha256']
    frame=pd.read_csv(PROJECT/'private/317/sample_time_lineage.csv').query('chain==@chain')
    code=int(pd.read_csv(PROJECT/'results/317/original_population.csv').query('chain==@chain').chain_index.iloc[0])
    result={day:[] for day in DAYS}
    with zipfile.ZipFile(path) as archive:
        index={Path(name).stem.lower():name for name in archive.namelist() if name.endswith('.csv')}
        for number,row in enumerate(frame.itertuples(index=False),1):
            cutoffs={day:int(row.first_valid_tx_time+day*86400) for day in DAYS if row.first_valid_tx_time+day*86400<=END}
            if not cutoffs:
                continue
            buffers={day:[Counter(),Counter(),None] for day in cutoffs}
            offset=0
            with archive.open(index[row.contract]) as stream:
                for chunk in pd.read_csv(stream,usecols=['timestamp','block_number','transaction_hash','from','to'],chunksize=100000):
                    chunk['row_position']=np.arange(offset,offset+len(chunk))
                    offset+=len(chunk)
                    chunk=chunk[chunk.timestamp<=max(cutoffs.values())].copy()
                    for field in ('from','to'):
                        chunk[field]=chunk[field].astype(str).str.strip().str.lower()
                    chunk=chunk[~chunk['from'].isin(INVALID)&~chunk['to'].isin(INVALID)]
                    for day,cutoff in cutoffs.items():
                        selected=chunk[chunk.timestamp<=cutoff]
                        if selected.empty:
                            continue
                        incoming,outgoing,tail=buffers[day]
                        degree_update(incoming,selected['to'])
                        degree_update(outgoing,selected['from'])
                        buffers[day][2]=ordered_tail(pd.concat([tail,selected],ignore_index=True) if tail is not None else selected)
            assert offset==row.raw_rows
            for day,cutoff in cutoffs.items():
                incoming,outgoing,tail=buffers[day]
                assert tail is not None and tail.timestamp.max()<=cutoff
                x,e=tensors(tail,incoming,outgoing)
                graph=Data(x=x,edge_index=e,y=torch.tensor(float(row.label)),chain_id=torch.tensor(code))
                result[day].append((row.row_id,row.event_id,cutoff,int(row.label),code,graph))
            if number%1000==0:
                print(chain,number,flush=True)
    for day,records in result.items():
        torch.save(records,DEST/f'{day}_{chain}.pt')
    return chain


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    chains=('ethereum','bsc','polygon')
    with ProcessPoolExecutor(max_workers=3) as pool:
        for chain in pool.map(scan,chains):
            print('source complete',chain,flush=True)
    for day in DAYS:
        records=[row for chain in chains for row in torch.load(DEST/f'{day}_{chain}.pt',weights_only=False)]
        records.sort(key=lambda row:(row[2],row[1]))
        population=pd.DataFrame([row[:5] for row in records],columns=['sample_id','event_id','timestamp','label','chain_index'])
        assert population.sample_id.is_unique
        n=len(population)
        a,b=int(.7*n),int(.85*n)
        population['split']=np.repeat(['train','validation','test'],[a,b-a,n-b])
        population.drop(columns='event_id').to_csv(DEST/f'{day}_population.csv',index=False)
        torch.save([row[-1] for row in records],DEST/f'{day}_graphs.pt')
        print('ready',day,n,flush=True)


if __name__=='__main__':
    main()
