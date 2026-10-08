#!/usr/bin/env python3
"""Expose constructor defaults and archived caller overrides, with provenance limits."""
import csv, hashlib, inspect, json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from pygod.detector import DOMINANT,AnomalyDAE,CoLA,OCGNN
from gog_fraud.models.pygod.dlg import DLG
from gog_fraud.models.pygod.dlg_full import DLGFull
from gog_fraud.models.pygod.gadnr import GADNR
OUT=ROOT/'projects/benchmark/evidence/astra_revision'
OUT.mkdir(parents=True,exist_ok=True)
FIELDS=['hid_dim','num_layers','lr','weight_decay','dropout','weight','batch_size','num_neigh','contamination','l1_hid_dim','l1_hops','l1_epochs','emb_dim','alpha','eta','theta']
classes={'DOMINANT':DOMINANT,'AnomalyDAE':AnomalyDAE,'CoLA':CoLA,'OCGNN':OCGNN,'GADNR':GADNR,'DLG-Base':DLG,'DLG-Aug':DLGFull}
rows=[]; signatures={}
for name,cls in classes.items():
    sig=inspect.signature(cls.__init__);source=Path(inspect.getsourcefile(cls))
    row={'model':name,**{field:(str(sig.parameters[field].default) if field in sig.parameters else 'NOT_EXPOSED') for field in FIELDS},'optimizer':'Adam (runner/model implementation)', 'setting_basis':'current pinned constructor defaults; historical exact per-run defaults not uniformly bound','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    rows.append(row); signatures[name]={k:str(p.default) for k,p in sig.parameters.items() if k!='self'}
with (OUT/'astra_model_defaults.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(OUT/'astra_model_signatures.json').write_text(json.dumps(signatures,indent=2)+'\n')
# Frozen caller policy controls what overrides the defaults; source IDs are
# preserved separately. No constructor-default table is called a tuning sweep.
campaigns=[{'campaign':'Round5','datasets':'ten original graphs (BitcoinOTC superseded)','epochs':50,'local_epochs':20,'batch_policy':'full graph; CoLA/OCGNN/GADNR num_neigh=-1','threshold':'validation-max F1; shared threshold protocol','provenance':'original run config/source hashes; incomplete per-run locks'},
{'campaign':'A03 crypto comparators','datasets':'Ethereum; BSC; Polygon','epochs':'30;40;40','local_epochs':'constructor default','batch_policy':'CoLA/OCGNN/GADNR: 64 if N>5000 else32; reconstruction: full graph','threshold':'validation prevalence percentile','provenance':'script reconstruction; not run-bound'},
{'campaign':'A03 repaired crypto Aug','datasets':'Ethereum; BSC; Polygon','epochs':'30;40;40','local_epochs':20,'batch_policy':'full graph; fresh clone per run','threshold':'restricted validation-max F1; percentiles80..99.5;40 candidates','provenance':'script reconstruction; raw scores/thresholds missing'},
{'campaign':'A05 BitcoinOTC','datasets':'canonical reconstructed tensor; seven detectors','epochs':50,'local_epochs':20,'batch_policy':'see archived per-run model_config; exact paths','threshold':'shared validation-max F1 protocol','provenance':'new run-bound source/score/config evidence'},
{'campaign':'A05 AnomalyDAE','datasets':'Ethereum; DGraphFin; failed preflights on large graphs','epochs':50,'local_epochs':'NA','batch_policy':'full graph; row block256','threshold':'shared validation protocol','provenance':'new run-bound source versions; restarted timing caveat'}]
with (OUT/'astra_campaign_settings.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=campaigns[0]);w.writeheader();w.writerows(campaigns)
print(json.dumps({'models':len(rows),'campaigns':len(campaigns),'status':'documented; historical defaults remain provenance-limited'},indent=2))
