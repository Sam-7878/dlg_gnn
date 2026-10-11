"""Dependency-free numeric verification of the curated R01 archive, not training."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import zipfile
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT.parents[1]
EVIDENCE=PROJECT/'evidence/r01_numeric_evidence.zip'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()
def rows(archive,name):return list(csv.DictReader(io.StringIO(archive.read(name).decode())))
def flag(value):return str(value).lower() in ('true','1','1.0')


def metrics(records,prefix):
    y=[int(r['label']) for r in records];h=[flag(r[prefix+'_prediction']) for r in records]
    p=[float(r[prefix+'_probability']) for r in records]
    tp=sum(a==1 and b for a,b in zip(y,h));tn=sum(a==0 and not b for a,b in zip(y,h))
    fp=sum(a==0 and b for a,b in zip(y,h));fn=sum(a==1 and not b for a,b in zip(y,h))
    positive=sum(y);ap=None
    if positive and positive<len(y):
        ordered=sorted(zip(p,y),reverse=True);i=0;seen=0;seenpos=0;ap=0.
        while i<len(ordered):
            j=i+1
            while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
            group_positive=sum(v for _,v in ordered[i:j]);seen+=j-i;seenpos+=group_positive
            ap+=group_positive/positive*seenpos/seen;i=j
    den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return {'N':len(y),'N_positive':positive,'ap':ap,'f1':2*tp/(2*tp+fp+fn) if positive else None,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/positive if positive else None,
            'mcc':((tp*tn-fp*fn)/math.sqrt(den) if den else 0.) if positive and positive<len(y) else None,'tp':tp,'tn':tn,'fp':fp,'fn':fn}


def exact_p(b,c):
    n=b+c
    return min(1.,2*sum(math.comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.


def holm(values):
    adjusted=[0.]*len(values);running=0.
    for rank,ix in enumerate(sorted(range(len(values)),key=values.__getitem__)):
        running=max(running,(len(values)-rank)*values[ix]);adjusted[ix]=min(1.,running)
    return adjusted


def verified_archive(path=EVIDENCE,expected_path=None):
    expected_path=expected_path or PROJECT/'evidence/r01_release.json'
    expected=json.loads(Path(expected_path).read_text());require(path.is_file(),'missing R01 evidence ZIP')
    require(sha(path.read_bytes())==expected['archive_sha256'],'archive SHA-256 mismatch')
    archive=zipfile.ZipFile(path);names=archive.namelist()
    require(len(names)==len(set(names)),'duplicate ZIP member')
    require(all(not Path(n).is_absolute() and '..' not in Path(n).parts and '\\' not in n for n in names),'unsafe ZIP member')
    manifest=json.loads(archive.read('release_manifest.json'))
    require(manifest['run_id']=='selectivestream_r01_20261011','mixed release run identity')
    require(set(manifest['files'])==set(names)-{'release_manifest.json'},'manifest membership mismatch')
    for name,identity in manifest['files'].items():
        data=archive.read(name)
        require(len(data)==identity['bytes'] and sha(data)==identity['sha256'],'payload drift: '+name)
    return archive,manifest


def numeric(archive):
    registry=[json.loads(s) for s in archive.read('analysis/experiment_registry.jsonl').decode().splitlines()]
    require(len(registry)==25,'expected 25 independent model identities')
    expected=rows(archive,'analysis/per_seed_metrics.csv');output=[];stats=[]
    for r in registry:
        name=Path(r['model_dir']).name;records=rows(archive,'analysis/'+name+'_paired_predictions.csv')
        require(len(records)==len({x['sample_key'] for x in records}),'duplicate contract key')
        require(all(x['run_id']==r['run_id'] and x['experiment_id']==r['experiment_id'] and x['model_hash']==r['model_hash'] for x in records),'mixed prediction identity')
        if r['experiment_id'].startswith('loco_'):
            require(r['experiment_id'][5:] not in r['train_chains'],'target entered source-only fitting')
        for chain in ['pooled']+r['test_chains']:
            subset=records if chain=='pooled' else [x for x in records if x['chain_scope']==chain]
            if not subset:continue
            for prefix,method in [('local','local'),('full','full'),('threshold_only','threshold_only'),('selective','margin')]:
                values=metrics(subset,prefix)
                target=[x for x in expected if x['experiment_id']==r['experiment_id'] and x['backbone']==r['backbone'] and int(x['seed'])==r['seed'] and x['chain_scope']==chain and x['method']==method and x['split']=='test' and (method!='margin' or float(x['budget'])==.25)]
                require(len(target)==1,'ambiguous numeric source identity')
                for key,value in values.items():
                    cell=target[0][key]
                    require((value is None and cell=='') or (value is not None and cell!='' and abs(value-float(cell))<1e-10),'raw/metric mismatch: '+name+'/'+chain+'/'+method+'/'+key)
                output.append({'experiment_id':r['experiment_id'],'backbone':r['backbone'],'seed':r['seed'],'chain_scope':chain,'method':method,**values})
        if r['experiment_id']=='pooled_snapshot' and r['backbone']=='GIN':
            b=sum(flag(x['local_prediction'])==int(x['label']) and flag(x['selective_prediction'])!=int(x['label']) for x in records)
            c=sum(flag(x['local_prediction'])!=int(x['label']) and flag(x['selective_prediction'])==int(x['label']) for x in records)
            stats.append({'seed':r['seed'],'b':b,'c':c,'exact_p':exact_p(b,c)})
    for r,p in zip(stats,holm([x['exact_p'] for x in stats])):r['holm_p']=p
    targets=rows(archive,'analysis/paired_statistics.csv')
    for r in stats:
        t=next(x for x in targets if int(x['seed'])==r['seed'])
        require(r['b']==int(t['n_local_correct_selective_wrong']) and r['c']==int(t['n_local_wrong_selective_correct']) and abs(r['exact_p']-float(t['exact_mcnemar_p']))<1e-12 and abs(r['holm_p']-float(t['holm_p']))<1e-12,'paired exact/Holm mismatch')
    return output,stats


def historical_numeric(archive):
    result=[]
    for seed in (11,22,33,44,55):
        records=rows(archive,f'historical_r4/seed{seed}_predictions.csv')
        require(len(records)==len({r['sample_id'] for r in records}),'duplicate historical contract key')
        b=sum(flag(r['direct_only_prediction'])==int(r['label']) and flag(r['primary_prediction'])!=int(r['label']) for r in records)
        c=sum(flag(r['direct_only_prediction'])!=int(r['label']) and flag(r['primary_prediction'])==int(r['label']) for r in records)
        result.append({'seed':seed,'b':b,'c':c,'exact_p':exact_p(b,c)})
    targets=rows(archive,'analysis/historical_r4_statistics_audit.csv')
    for record,p in zip(result,holm([r['exact_p'] for r in result])):
        record['holm_p']=p;target=next(r for r in targets if int(r['seed'])==record['seed'])
        require(record['b']==int(target['n_local_correct_selective_wrong']) and record['c']==int(target['n_local_wrong_selective_correct'])
                and abs(record['exact_p']-float(target['exact_mcnemar_p']))<1e-12 and abs(p-float(target['holm_p']))<1e-12,'historical exact/Holm mismatch')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['verify','tables'],default='verify');parser.add_argument('--archive',type=Path,default=EVIDENCE);parser.add_argument('--expected',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    archive,manifest=verified_archive(args.archive,args.expected)
    with archive:
        for name,expected_hash in manifest['current_source_hashes'].items():
            require((ROOT/name).is_file() and sha((ROOT/name).read_bytes())==expected_hash,'curated source drift: '+name)
        results,stats=numeric(archive);historical=historical_numeric(archive)
        # Frozen execution-critical source bytes are checked separately from current sidecars.
        frozen=json.loads(archive.read('training_source_manifest.json'))
        for name,expected in frozen['dirty_revision_source_hashes'].items():
            require(sha((ROOT/name).read_bytes())==expected,'training source drift: '+name)
        if args.mode=='tables':
            dest=args.output or PROJECT/'results/public_recomputed_r01';dest.mkdir(parents=True,exist_ok=True)
            for name,data in [('primary_metrics.csv',results),('exact_statistics.csv',stats),('historical_exact_statistics.csv',historical)]:
                with (dest/name).open('w',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=data[0]);writer.writeheader();writer.writerows(data)
        print(json.dumps({'status':'PASS','model_identities':25,'raw_metric_rows':len(results),'exact_paired_tests':len(stats),'separate_historical_exact_paired_tests':len(historical),'scope':'archive/source integrity and aligned raw AP/confusion/paired exact statistics; not new training, visual review, or independent scientific approval'},indent=2))


if __name__=='__main__':main()
