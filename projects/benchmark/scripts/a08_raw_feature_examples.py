#!/usr/bin/env python3
"""Independent csv/set arithmetic on real provider examples, not builder calls."""
import csv,hashlib,io,json,math,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
E=ROOT/'projects/benchmark/evidence/a08_data_repair';L=ROOT/'local_storage/benchmark/a08_data_repair'


def main():
    config=json.loads((ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml').read_text())
    checks=[];private=[]
    for chain in ('polygon','bsc','ethereum'):
        with np.load(L/'frozen'/chain/'contract_graph.npz',allow_pickle=False) as z:
            raw=z['raw_features'];ids=z['node_ids'].tolist();x=z['x']
            mean=np.array([math.fsum(float(v) for v in raw[:,j])/len(raw) for j in range(8)])
            std=np.array([math.sqrt(math.fsum((float(v)-mean[j])**2 for v in raw[:,j])/len(raw)) for j in range(8)])
            scale=np.where(std>0,std,1.)
            if not np.allclose(z['preprocessing_mean'],mean,rtol=0,atol=1e-10) or not np.allclose(z['preprocessing_scale'],scale,rtol=0,atol=1e-10):
                raise ValueError('full-population preprocessing mismatch '+chain)
            expected_x=((raw-mean)/scale).astype(np.float32)
            if not np.allclose(x,expected_x,rtol=1e-6,atol=1e-6):raise ValueError('standardized feature mismatch '+chain)
            checks.append({'chain':chain,'check':'all-node standardization via independent math.fsum','population':len(raw),'status':'PASS','float64_atol':1e-10,'float32_atol':1e-6,'float32_rtol':1e-6})
        members=json.loads((L/'build_a'/chain/'raw_members.json').read_text());lookup={m['member']:m for m in members}
        with zipfile.ZipFile(Path(config['raw_root'])/'transactions'/f'{chain}.zip') as archive:
            selected=0
            for member in sorted(archive.infolist(),key=lambda a:a.filename):
                if member.is_dir() or not member.filename.endswith('.csv') or member.file_size>15000:continue
                payload=archive.read(member)
                pairs=[(r['from'].strip().lower(),r['to'].strip().lower()) for r in csv.DictReader(io.StringIO(payload.decode('utf-8-sig')))]
                if not pairs:continue
                if hashlib.sha256(payload).hexdigest()!=lookup[member.filename]['sha256']:raise ValueError('selected raw member changed')
                senders={a for a,b in pairs};receivers={b for a,b in pairs};unique=set(pairs);nonself={(a,b) for a,b in unique if a!=b}
                expected=np.array([math.log1p(len(pairs)),math.log1p(len(senders)),math.log1p(len(receivers)),math.log1p(len(senders|receivers)),math.log1p(len(unique)),
                                   sum(a==b for a,b in pairs)/len(pairs),sum((b,a) in unique for a,b in nonself)/max(1,len(nonself)),1-len(unique)/len(pairs)])
                identifier=chain+':'+Path(member.filename).stem.lower();position=ids.index(identifier)
                error=float(np.max(np.abs(raw[position]-expected)))
                if error>1e-12:raise ValueError('independent real CSV feature mismatch '+chain)
                checks.append({'chain':chain,'check':'all8 raw features via csv/set arithmetic','stable_id_sha256':hashlib.sha256(identifier.encode()).hexdigest(),
                               'raw_member_sha256':hashlib.sha256(payload).hexdigest(),'max_abs_error':error,'atol':1e-12,'status':'PASS'})
                private.append({'chain':chain,'node_id':identifier,'raw_member':member.filename,'expected_feature_values':expected.tolist(),'builder_feature_values':raw[position].tolist()})
                selected+=1
                if selected==2:break
            if selected!=2:raise ValueError('insufficient independent raw examples')
    (L/'raw_feature_example_values.json').write_text(json.dumps(private,indent=2)+'\n')
    (E/'audit/raw_feature_examples.json').write_text(json.dumps({'status':'PASS','scope':'6 real raw CSV examples independently recomputed;3 whole-population standardization checks; no label access or builder feature function calls',
        'selection':'first2 nonempty CSVs no larger than15KB in lexicographic archive member order; not chosen using targets or scores','checks':checks,
        'private_values_sha256':hashlib.sha256((L/'raw_feature_example_values.json').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('PASS:6 independent real CSV feature examples and3 full-population standardization checks')
if __name__=='__main__':main()
