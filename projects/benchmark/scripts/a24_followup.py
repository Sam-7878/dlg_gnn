"""Preflight-qualified full resource measurements then bounded score recovery."""
from a24_run_studies import *
# All four resource cells were selected before their AP was observed; proceed by runtime only.
rs=[]
for d,m in [('Cora-Syn','DOMINANT'),('DGraphFin','DOMINANT'),('Ethereum','DLG-Aug'),('Reddit-Syn','DLG-Base')]:
 pilot=json.loads((LOCAL/'runs'/f'A24_RESOURCE_ONE_EPOCH_PILOT__{d}__{m}__seed42'/'result.json').read_text());epochs=30 if d=='Ethereum' else 50
 assert pilot['mean_global_epoch_seconds']*epochs<86400
 r=run_gnn(d,m,42,epochs,'A24_RESOURCE_FULL_DECLARED');rs.append(r);writecsv(SUP/'measured_full_resource_cases.csv',rs)
# Reuse the10existing Elliptic Base/Aug control vectors; never rerun them.
alerts=[json.loads(p.read_text()) for p in sorted((LOCAL/'runs').glob('A24_CANONICAL_CONTROL__Elliptic__*__seed*/result.json')) if json.loads(p.read_text())['model'] in ['DLG-Base','DLG-Aug']]
for d in ['LANL-RedTeam','DGraphFin']:
 for s in range(42,47):
  for m in ['DLG-Base','DLG-Aug']:
   r=run_gnn(d,m,s,50,'A24_ALERT_REPRODUCTION');alerts.append(r)
   keys=list(dict.fromkeys(k for r in alerts for k in r));writecsv(SUP/'real_domain_alert_budget_per_seed.csv',[{k:r.get(k,'') for k in keys} for r in alerts])
print('Bounded followup complete;4full resource cells;30Base/Aug real-domain vectors (10reused,20new).',flush=True)
