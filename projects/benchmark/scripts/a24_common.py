"""A24 supplementary helpers; never mutate frozen A23 inputs."""
from pathlib import Path
import csv,json,hashlib,subprocess
ROOT=Path(__file__).resolve().parents[3];PUB=ROOT/'projects/benchmark/publication/a24';SUP=ROOT/'projects/benchmark/supplementary/a24';LOCAL=ROOT/'local_storage/benchmark/a24';PAPER=ROOT/'projects/benchmark/paper/a24';REG=ROOT/'projects/benchmark/publication/a16/canonical_selected_run_registry.csv'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def rows(p):return list(csv.DictReader(Path(p).open()))
def writecsv(p,rs):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def located(p):
 p=Path(p)
 if p.exists():return p
 s=str(p)
 if s.startswith('outputs/benchmark/'):return ROOT/'archive/benchmark/outputs'/s[len('outputs/benchmark/'):]
 if s.startswith(str(ROOT)+'/outputs/benchmark/'):return ROOT/'archive/benchmark/outputs'/s.split('/outputs/benchmark/',1)[1]
 return ROOT/p
