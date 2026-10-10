#!/usr/bin/env python3
"""Durable sequential qualification + campaign supervisor, bound to G0–G4."""
import csv,hashlib,json,os,shutil,subprocess,sys,time,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.benchmark_lineage import verify_run
E=ROOT/'projects/benchmark/evidence/a08_data_repair'
L=ROOT/'local_storage/benchmark/a08_data_repair'
CFG=ROOT/'configs/benchmark/a08_crypto_clean_v1.yaml'
CFG_VALUE=json.loads(CFG.read_text())
SMI=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'


def h(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,indent=2)+'\n')


def command(args,log,guard=86400):
    stop=threading.Event()
    def monitor():
        telemetry=L/'telemetry'/'active_gpu_samples.csv'
        telemetry.parent.mkdir(parents=True,exist_ok=True)
        while not stop.is_set():
            row=subprocess.run([SMI,'--query-gpu=timestamp,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.mem','--format=csv,noheader'],capture_output=True,text=True)
            with telemetry.open('a') as t:t.write(row.stdout)
            stop.wait(30)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    with Path(log).open('a') as f:
        f.write('\nCOMMAND '+json.dumps(args)+'\n');f.flush()
        try:return subprocess.run(args,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=CFG_VALUE['gpu_uuid']),stdout=f,stderr=f,timeout=guard).returncode
        except subprocess.TimeoutExpired:return 124
        finally:stop.set();worker.join(timeout=5)


def check_previous(gates):
    for gate in gates:
        p=E/'audit'/f'{gate}.json'
        if not p.exists() or json.loads(p.read_text()).get('status')!='PASS':raise RuntimeError(f'{gate} not passed')


def recover_public_record(path,run_id,smoke=False):
    if path.exists():return
    out=L/('smoke' if smoke else 'runs')/run_id
    local=out/'run_manifest.json'
    if not local.exists():return
    record=json.loads(local.read_text())
    verify_run(ROOT,CFG_VALUE,record,smoke=smoke)
    public={k:v for k,v in record.items() if k not in ('raw_scores_path','checkpoint_path','metrics_path')}
    metrics=json.loads((out/'metrics.json').read_text())
    public['metrics']={k:v for k,v in metrics.items() if k!='alert_budgets'}
    public['metrics']['alert_budgets']={q:{k:v for k,v in b.items() if k!='selected_node_ids'} for q,b in metrics['alert_budgets'].items()}
    write(path,public)


def main():
    check_previous(['G0','G1','G2','G3','G4'])
    qual=E/'audit/cuda_exactness.json'
    if not qual.exists() or json.loads(qual.read_text())['status']!='PASS':raise RuntimeError('CUDA numerical qualification missing')
    smoke=[]
    for model in CFG_VALUE['models']:
        rid=f'polygon_contract_clean_v1__{model}__seed42'
        p=E/'qualification_runs'/(rid+'.json')
        recover_public_record(p,rid,smoke=True)
        if not p.exists():
            code=command([sys.executable,'-u',str(ROOT/'projects/benchmark/scripts/a08_run_crypto.py'),'--chain','polygon','--model',model,'--seed','42','--smoke'],L/'qualification.log')
            if code!=0:raise RuntimeError('actual Polygon qualification failed '+model)
        record=json.loads(p.read_text())
        if record['status']!='SUPPORTED_EXACT' or record['scope']!='qualification':raise RuntimeError('wrong smoke evidence')
        verify_run(ROOT,CFG_VALUE,record,smoke=True)
        smoke.append({'path':str(p.relative_to(ROOT)),'sha256':h(p)})
    numerical=E/'audit/gadnr_numerical_qualification.json'
    if not numerical.is_file():raise RuntimeError('revision2 GADNR exact-objective numerical qualification missing')
    numerical_record=json.loads(numerical.read_text())
    if numerical_record['status']!='PASS' or numerical_record['source_sha256']!=h(ROOT/'src/gog_fraud/models/a08_gadnr_numerics.py'):
        raise RuntimeError('GADNR numerical qualification source changed or failed')
    rid='bsc_contract_clean_v1__GADNR__seed42';p=E/'qualification_runs'/(rid+'.json')
    recover_public_record(p,rid,smoke=True)
    if not p.exists():
        code=command([sys.executable,'-u',str(ROOT/'projects/benchmark/scripts/a08_run_crypto.py'),'--chain','bsc','--model','GADNR','--seed','42','--smoke'],L/'qualification.log')
        if code!=0:raise RuntimeError('failed-chain BSC GADNR qualification did not pass')
    verify_run(ROOT,CFG_VALUE,json.loads(p.read_text()),smoke=True)
    write(E/'audit/G5.json',{'status':'PASS','cuda_exactness_sha256':h(qual),'actual_chain_integration':smoke,
         'gadnr_numerical_qualification_sha256':h(numerical),'failed_chain_integration':{'path':str(p.relative_to(ROOT)),'sha256':h(p)},
         'scope':'seven actual Polygon and failed-chain BSC one-epoch integrations; exact-objective GADNR arithmetic qualified; primary not yet completed'})
    # Process-independent seeds permit this operational ordering; it tests the
    # previously failed chain before spending time on remaining populations.
    for chain in ['bsc','polygon','ethereum']:
        for model in CFG_VALUE['models']:
            for seed in CFG_VALUE['seeds']:
                rid=f'{chain}_contract_clean_v1__{model}__seed{seed}';p=E/'run_manifests'/(rid+'.json')
                recover_public_record(p,rid)
                if p.exists():
                    r=json.loads(p.read_text())
                    if r['status']=='SUPPORTED_EXACT':
                        verify_run(ROOT,CFG_VALUE,r)
                        continue
                    raise RuntimeError('existing unresolved run '+rid+' '+r['status'])
                # Hardware state is captured for each fresh invocation.
                telemetry=subprocess.check_output([SMI,'--query-gpu=name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.mem','--format=csv'],text=True)
                t=L/'telemetry'/(rid+'.txt');t.parent.mkdir(parents=True,exist_ok=True);t.write_text(telemetry)
                code=command([sys.executable,'-u',str(ROOT/'projects/benchmark/scripts/a08_run_crypto.py'),'--chain',chain,'--model',model,'--seed',str(seed)],L/'campaign.log',CFG_VALUE['guard_seconds'])
                if code==124:
                    write(p,{'run_id':rid,'dataset_id':chain+'_contract_clean_v1','model':model,'seed':seed,'status':'UNSUPPORTED_OPERATIONAL_TIMEOUT_OBSERVED','guard_seconds':86400,'guard_scope':'per_model_dataset_seed','actual_timeout':True})
                    raise RuntimeError('observed timeout; evidence review and support decision required '+rid)
                if code!=0:raise RuntimeError('run needs investigation or clean OOM confirmation '+rid)
    manifests=list((E/'run_manifests').glob('*.json'))
    if len(manifests)!=105 or not all(json.loads(p.read_text())['status']=='SUPPORTED_EXACT' for p in manifests):raise RuntimeError('campaign coverage incomplete')
    write(E/'audit/G6.json',{'status':'PASS','actual_successes':105,'records':[{'path':str(p.relative_to(ROOT)),'sha256':h(p)} for p in sorted(manifests)],'scope':'all planned fresh primary runs completed; later score recomputation/aggregation still required'})
    print('CAMPAIGN_COMPLETE; G6-gated postprocess service will aggregate/build PDFs; final review still required',flush=True)
if __name__=='__main__':main()
