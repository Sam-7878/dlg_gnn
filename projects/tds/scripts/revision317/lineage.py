"""Stream upstream CSVs; reconstruct lifetime before any model fitting."""
import concurrent.futures
import json
import time
import zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from common import PROJECT,OUT,REPORTS,PRIVATE,RAW,packed,frame_from,sha,write

INVALID={'','nan','none','null'}
BOUNDARY=1709251199


def scan_chain(chain,rows):
    cache=PRIVATE/f'lineage_{chain}.csv'
    completed=pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    done=set(completed.row_id) if len(completed) else set()
    entries=completed.to_dict('records');start=time.monotonic()
    with zipfile.ZipFile(RAW/'transactions'/f'{chain}.zip') as z:
        index={n.rsplit('/',1)[-1].removesuffix('.csv').lower():n for n in z.namelist() if n.endswith('.csv')}
        for n,row in enumerate(rows.itertuples(),1):
            if row.row_id in done:continue
            first=None;last=None;count=0;raw_count=0;equal=0;at7=0;at30=0;times=[]
            # Only3 columns; chunking bounds memory independently of contract size.
            with z.open(index[row.contract]) as f:
                for part in pd.read_csv(f,usecols=['from','to','timestamp'],dtype=str,keep_default_na=False,chunksize=200000):
                    raw_count+=len(part)
                    valid=~part['from'].str.strip().str.lower().isin(INVALID)&~part['to'].str.strip().str.lower().isin(INVALID)
                    t=pd.to_numeric(part.loc[valid,'timestamp'],errors='raise').astype('int64').to_numpy()
                    if not len(t):continue
                    first=int(t.min()) if first is None else min(first,int(t.min()))
                    last=int(t.max()) if last is None else max(last,int(t.max()))
                    count+=len(t);equal+=int((t==BOUNDARY).sum())
            if first is None or last!=int(row.timestamp):raise RuntimeError(('time lineage mismatch',row.event_id,first,last,row.timestamp))
            entries.append(dict(row_id=row.row_id,event_id=row.event_id,chain=chain,contract=row.contract,label=row.label,
                first_valid_tx_time=first,last_valid_tx_time=last,original_sample_cutoff=row.timestamp,
                collection_boundary_time=None,reference_archive_endpoint=BOUNDARY,
                documented_coverage_end_month='2024-02',exact_collection_policy_verified=False,
                tx_count=count,raw_rows=raw_count,observed_lifetime_seconds=last-first,
                original_split=row.original_split,first_time_missing=False,last_time_missing=False,
                label_source='upstream labels.csv; Category0 positive',label_unit='token/contract',
                label_asof_time_if_available=None,label_time_unknown=True,rows_at_reference_endpoint=equal))
            if n%100==0:
                pd.DataFrame(entries).to_csv(cache,index=False)
                print(chain,n,'/',len(rows),'elapsed_s',round(time.monotonic()-start,1),flush=True)
    pd.DataFrame(entries).to_csv(cache,index=False)
    manifest=json.loads((PROJECT/'results/main_final_v2/upstream_reacquisition_manifest.json').read_text())
    expected=next(e['sha256'] for e in manifest['official_distribution']['entries'] if e['relative_path']==f'transactions/{chain}.zip')
    actual=sha(RAW/'transactions'/f'{chain}.zip');assert actual==expected
    write(OUT/f'{chain}_source_identity.json',dict(archive_sha256=actual,all_members_read_with_crc=True,contracts=len(entries)))
    return entries


def support(frame,order):
    f=frame.sort_values([order,'event_id'],kind='stable').copy()
    f['alternative_split']=np.repeat(['train','validation','test'],[17021,3647,3648])
    rows=[]
    for split,sub in f.groupby('alternative_split',sort=False):
        for chain,g in [('all',sub),*list(sub.groupby('chain'))]:
            rows.append(dict(ordering=order,split=split,chain=chain,n=len(g),positive=int(g.label.sum()),prevalence=float(g.label.mean()),
                first_min=int(g.first_valid_tx_time.min()),first_median=float(g.first_valid_tx_time.median()),first_max=int(g.first_valid_tx_time.max()),
                last_min=int(g.last_valid_tx_time.min()),last_median=float(g.last_valid_tx_time.median()),last_max=int(g.last_valid_tx_time.max()),
                lifetime_median_days=float(g.observed_lifetime_seconds.median()/86400),tx_count_median=float(g.tx_count.median()),
                within1d_reference_endpoint_fraction=float((BOUNDARY-g.last_valid_tx_time<=86400).mean()),
                exactly_at_reference_endpoint=int((g.last_valid_tx_time==BOUNDARY).sum())))
    return rows,f


def summarize(frame):
    assert len(frame)==24316 and frame.row_id.is_unique and frame.event_id.is_unique
    labels=pd.read_csv(RAW/'labels.csv');labels.columns=[x.lower() for x in labels.columns]
    labels['chain']=labels.chain.str.lower().str.strip();labels['contract']=labels.contract.str.lower().str.strip()
    merged=frame.merge(labels[['chain','contract','category']],on=['chain','contract'],validate='one_to_one')
    assert len(merged)==len(frame) and ((merged.category==0).astype(int)==merged.label).all()
    frame=merged.sort_values('row_id');frame.to_csv(PRIVATE/'sample_time_lineage.csv',index=False)
    public=frame.drop(columns=['event_id','contract']).rename(columns={'row_id':'sample_id'})
    public.to_csv(OUT/'sample_time_lineage.csv',index=False)
    lifetime=[]
    for (chain,label),g in frame.groupby(['chain','label']):
        lifetime.append(dict(chain=chain,label=int(label),n=len(g),
            lifetime_q25_days=float(g.observed_lifetime_seconds.quantile(.25)/86400),
            lifetime_median_days=float(g.observed_lifetime_seconds.median()/86400),
            lifetime_q75_days=float(g.observed_lifetime_seconds.quantile(.75)/86400),
            lifetime_q95_days=float(g.observed_lifetime_seconds.quantile(.95)/86400)))
    pd.DataFrame(lifetime).to_csv(OUT/'lifespan_by_class.csv',index=False)
    a,old=support(frame,'last_valid_tx_time');b,new=support(frame,'first_valid_tx_time')
    pd.DataFrame(a+b).to_csv(OUT/'first_last_ordering_diagnostics.csv',index=False)
    pd.DataFrame(a).to_csv(OUT/'split_chain_entity_composition.csv',index=False)
    horizons=[]
    for days in [7,30]:
        cutoff=frame.first_valid_tx_time+days*86400
        eligible=cutoff<=BOUNDARY
        for (chain,label),g in frame.groupby(['chain','label']):
            mask=eligible.loc[g.index]
            horizons.append(dict(horizon_days=days,chain=chain,label=int(label),n=len(g),eligible=int(mask.sum()),
                censored_at_reference_boundary=int((~mask).sum()),
                observed_activity_reaches_horizon=int((g.last_valid_tx_time>=cutoff.loc[g.index]).sum()),
                interpretation='eligibility by observed archive endpoint, not verified prospective label availability'))
    pd.DataFrame(horizons).to_csv(OUT/'fixed_horizon_eligibility.csv',index=False)
    old_p=[x['prevalence'] for x in a if x['chain']=='all'];new_p=[x['prevalence'] for x in b if x['chain']=='all']
    decision='SELECTION_EFFECT_SUPPORTED' if max(abs(x-y) for x,y in zip(old_p,new_p))>=.05 else 'INCONCLUSIVE'
    write(OUT/'selection_decision.json',dict(decision=decision,decision_rule='pre-registered descriptive threshold: >=5 percentage-point change under first-vs-last sorting',
       original_prevalence=old_p,first_time_prevalence=new_p,
       observed_archive_max_timestamp=int(frame.last_valid_tx_time.max()),reference_endpoint=BOUNDARY,
       documented_month='2024-02',exact_collection_cutoff_policy_verified=False,
       right_censoring_causal_mechanism_established=False,label_ascertainment_time_available=False,
       scientific_claim_scope='archive/sampling-associated composition, not identified behavioral drift or prospective early warning',
       full_lineage_reconstructed=True,raw_rows=int(frame.raw_rows.sum()),valid_rows=int(frame.tx_count.sum())))
    # ECDFs are explicit data, not a fitted survival model under known censoring.
    ecdf=[]
    for label,g in frame.groupby('label'):
        values=np.sort(g.observed_lifetime_seconds.to_numpy()/86400)
        for q in np.linspace(0,1,101):ecdf.append(dict(label=int(label),quantile=float(q),observed_lifetime_days=float(np.quantile(values,q))))
    pd.DataFrame(ecdf).to_csv(OUT/'lifespan_ecdf.csv',index=False)
    print('G1',decision,'last',old_p,'first',new_p,flush=True)


def main():
    frame=frame_from(packed())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(scan_chain,c,g) for c,g in frame.groupby('chain')]
        values=[]
        for f in concurrent.futures.as_completed(futures):values.extend(f.result())
    summarize(pd.DataFrame(values))


if __name__=='__main__':main()
