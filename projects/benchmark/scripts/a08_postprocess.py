#!/usr/bin/env python3
"""Continue authorized tables/private PDF work once the GPU campaign closes."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gog_fraud.data.crypto_raw import sha256
E=ROOT/'projects/benchmark/evidence/a08_data_repair';L=ROOT/'local_storage/benchmark/a08_data_repair'


def main():
    L.mkdir(parents=True,exist_ok=True)
    lock=(L/'postprocess.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (E/'audit/G6.json').exists():
        state=subprocess.run(['systemctl','--user','is-active','dlg-gnn-a08-campaign.service'],capture_output=True,text=True).stdout.strip()
        if state not in ('active','activating'):
            raise RuntimeError('campaign stopped before G6; inspect actual failure, never build old-paper substitute')
        time.sleep(20)
    if json.loads((E/'audit/G6.json').read_text())['status']!='PASS':raise ValueError('G6 not passed')
    aggregate=ROOT/'projects/benchmark/scripts/a08_aggregate.py'
    with (L/'postprocessing.log').open('a') as log:
        g7=E/'audit/G7.json'
        if g7.exists():
            gate=json.loads(g7.read_text())
            if gate['status']!='PASS' or gate['approved_registry_sha256']!=sha256(E/'approved_registry.json') or gate['aggregate_source_sha256']!=sha256(aggregate):
                raise ValueError('completed aggregate binding changed; explicit repair/review required')
        else:subprocess.run([sys.executable,'-u',str(aggregate)],cwd=ROOT,stdout=log,stderr=log,check=True)
        writer=ROOT/'projects/benchmark/scripts/a08_build_manuscript.py'
        if not writer.is_file():raise RuntimeError('author-local A08 paper writer absent')
        if not (E/'audit/G8.json').exists():
            subprocess.run([sys.executable,'-u',str(writer)],cwd=ROOT,stdout=log,stderr=log,check=True)
        build=L/'paper_build_manifest.json';paper=json.loads(build.read_text())
        if paper['status']!='BUILT_NOT_REVIEWED' or len(paper['artifacts'])!=2:raise RuntimeError('new neutral/MDPI build incomplete')
        if paper['private_writer_sha256']!=sha256(writer) or paper['approved_registry_sha256']!=sha256(E/'approved_registry.json'):
            raise ValueError('paper source/registry changed after build')
        for item in paper['artifacts']:
            if sha256(ROOT/item['path'])!=item['sha256']:raise ValueError('paper PDF bytes changed')
        gate={'status':'PASS','paper_build_manifest_sha256':sha256(build),'artifacts':paper['artifacts'],
              'supplement_disposition':'F1 and full pairwise panels integrated into both A08 appendices; A07 supplement preserved historical',
              'scope':'actual neutral/MDPI compilation only; G9 every-page numeric/semantic/layout review and G10 package audit remain required'}
        (E/'audit/G8.json').write_text(json.dumps(gate,indent=2)+'\n')
    print('G7_AND_G8_COMPLETE; FINAL_PASS still requires every-page review and public/local reproduction audit',flush=True)
if __name__=='__main__':main()
