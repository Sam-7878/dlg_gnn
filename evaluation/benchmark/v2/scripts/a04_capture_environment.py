#!/usr/bin/env python3
"""Capture the executed environment and hash the frozen offline wheelhouse."""
import csv,hashlib,json,platform,subprocess,sys
from importlib import metadata
from pathlib import Path
from packaging.utils import canonicalize_name, parse_wheel_filename
import torch,torch_geometric,pygod
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'evaluation/benchmark/v2/paper_ready_final/publication_evidence_a04/environment'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'environment/locks/benchmark-a03-cuda.lock.txt'
LOCK=ROOT/'environment/locks/benchmark-a04-cuda.lock.txt'
LOCK.write_bytes(SOURCE.read_bytes())
(OUT/LOCK.name).write_bytes(LOCK.read_bytes())
WHEELHOUSE=ROOT/'environment/wheelhouse-a04'
wheel_map={}
for wheel in sorted(WHEELHOUSE.glob('*.whl')):
    name,version,_,_=parse_wheel_filename(wheel.name)
    key=canonicalize_name(name)
    if key in wheel_map:
        raise RuntimeError(f'Duplicate wheel for {key}: {wheel_map[key].name}, {wheel.name}')
    wheel_map[key]=(wheel,str(version))
probe=ROOT/'evaluation/benchmark/v2/environment/journal_cuda/environment_probe.json'
info={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
      'torch':torch.__version__,'torch_cuda':torch.version.cuda,'pyg':torch_geometric.__version__,
      'pygod':metadata.version('pygod'),'cuda_available':torch.cuda.is_available(),
      'source_probe_path':str(probe.relative_to(ROOT)),'source_probe_sha256':hashlib.sha256(probe.read_bytes()).hexdigest(),
      'a04_lock_sha256':hashlib.sha256(LOCK.read_bytes()).hexdigest(),
      'wheelhouse_path':str(WHEELHOUSE.relative_to(ROOT)),
      'qualification_note':'Empirically qualified; PyG 2.8 published compatibility does not list PyTorch 2.14.'}
try: info['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
except Exception:info['git_commit']=None
(OUT/'toolchain-a04.json').write_text(json.dumps(info,indent=2)+'\n')
(ROOT/'environment/toolchains/benchmark-a04-cuda.json').write_text(json.dumps(info,indent=2)+'\n')
rows=[];hashed=[];seen=set()
for line in SOURCE.read_text().splitlines():
    if not line or line.startswith('#') or '==' not in line:continue
    package,version=line.split('==',1)
    key=canonicalize_name(package)
    seen.add(key)
    wheel,wheel_version=wheel_map.get(key,(None,None))
    expected_version='2.14.1+cu130' if key=='torch' else version
    if wheel is None or wheel_version!=expected_version:
        rows.append({'package':package,'installed_version':version,'wheel_version':wheel_version or 'MISSING',
                     'wheel_filename':wheel.name if wheel else 'MISSING','wheel_sha256':'MISSING',
                     'status':'MISSING_OR_VERSION_MISMATCH'})
        continue
    digest=hashlib.sha256(wheel.read_bytes()).hexdigest()
    rows.append({'package':package,'installed_version':version,'wheel_version':wheel_version,
                 'wheel_filename':wheel.name,'wheel_sha256':digest,'status':'HASH_VERIFIED'})
    hashed.append(f'{package}=={expected_version} --hash=sha256:{digest}')
extra=set(wheel_map)-seen
if extra:raise RuntimeError(f'Wheelhouse contains unpinned packages: {sorted(extra)}')
with (OUT/'wheel_artifacts_a04.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(OUT/'wheel_artifacts_a04.sha256').write_text(''.join(f'{r["wheel_sha256"]}  {r["wheel_filename"]}\n' for r in rows if r['status']=='HASH_VERIFIED'))
HASHED=ROOT/'environment/locks/benchmark-a04-cuda.hashed.txt'
HASHED.write_text('# Exact wheel builds; install with --no-index --no-deps --require-hashes and --find-links environment/wheelhouse-a04\n'+'\n'.join(hashed)+'\n')
(OUT/HASHED.name).write_bytes(HASHED.read_bytes())
info['hashed_lock_sha256']=hashlib.sha256(HASHED.read_bytes()).hexdigest()
(OUT/'toolchain-a04.json').write_text(json.dumps(info,indent=2)+'\n')
(ROOT/'environment/toolchains/benchmark-a04-cuda.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps({'lock_sha256':info['a04_lock_sha256'],'hashed_lock_sha256':info['hashed_lock_sha256'],
                  'packages_without_wheel_artifact':sum(r['status']!='HASH_VERIFIED' for r in rows),'toolchain':info},indent=2))
