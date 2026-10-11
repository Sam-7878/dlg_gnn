"""Inventory and preserve the input before revision; no historical source rewrite."""
import csv
import json
import platform
import subprocess
import sys
import zipfile
from pathlib import Path
from r01_common import ROOT,PROJECT,CONFIG,config,output,read,write,sha256,git_sha,source_identity


def main():
    cfg=config();out=output();out.mkdir(parents=True,exist_ok=True)
    source=PROJECT/'paper/current';snapshot=out/'input_snapshot.zip'
    if not snapshot.exists():
        with zipfile.ZipFile(snapshot,'w',zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(source.rglob('*')):
                if path.is_file():archive.write(path,str(path.relative_to(source)))
    inventory=[]
    targets={'graph_cache':ROOT/cfg['graph_cache'],'raw_events':ROOT/cfg['raw_events'],
        'historical_registry':PROJECT/'results/canonical_r4/canonical/experiment_registry.csv',
        'original_snapshot_source':Path('/mnt/d/_Work/_data/GoG_sci_v2'),
        'provider_transactions':Path('/mnt/d/_Work/_data/gog_round7_upstream/Token Data')}
    for key,path in targets.items():
        inventory.append({'artifact_id':key,'path':str(path),'status':'present' if path.exists() else 'unavailable',
            'sha256':sha256(path) if path.is_file() else '',
            'public_access':'local_provider_data' if key in ('provider_transactions','original_snapshot_source') else 'local_artifact',
            'reused_as_new_run_evidence':key in ('graph_cache','raw_events')})
    for path in sorted((PROJECT/'results/canonical_r4').rglob('*')):
        if path.is_file() and (path.name in ('selection.json','summary.json','event_trace.csv','completed.json') or path.name.endswith('_predictions.csv')):
            inventory.append({'artifact_id':'historical_r4','path':str(path.relative_to(ROOT)),'status':'present','sha256':sha256(path),'public_access':'repository_numeric_evidence','reused_as_new_run_evidence':False})
    with (out/'evidence_inventory.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=inventory[0].keys());writer.writeheader();writer.writerows(inventory)
    baseline=out/'execution_plan.json'
    if baseline.exists():
        if read(baseline)['config_sha256']!=cfg['config_sha256']:raise RuntimeError('Frozen R01 configuration changed; create a new run ID')
    else:
        write(baseline,{'run_id':cfg['run_id'],'config_sha256':cfg['config_sha256'],'scientific_baseline_commit':git_sha(),
            'analysis_status':'revision plan recorded after historical test review; not prospective preregistration',
            'scope':'retrospective contract snapshots and systems replay; label availability and raw edge time remain unknown',
            'input_snapshot_sha256':sha256(snapshot),'review_input_archive_claimed_sha256':'6a0e13355016faa95a710d6512b2305817afae84a7ca32c20127733bc0b0cd14',
            'review_zip_available':False,'review_zip_note':'Directory snapshot has its own hash; do not claim byte identity with reviewed ZIP.',
            'inputs':{key:sha256(path) for key,path in targets.items() if path.is_file()},'source_hashes':source_identity()})
    import torch,torch_geometric,pygod,numpy,scipy,sklearn,pandas
    environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'os_release':Path('/etc/os-release').read_text(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'pyg':torch_geometric.__version__,'pygod':pygod.__version__,'numpy':numpy.__version__,
        'scipy':scipy.__version__,'sklearn':sklearn.__version__,'pandas':pandas.__version__,
        'gpus':[{'index':i,'name':torch.cuda.get_device_name(i),'bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]}
    write(out/'environment.json',environment)
    (out/'environment.lock').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    cache=torch.load(ROOT/cfg['graph_cache'],map_location='cpu',weights_only=False)
    overview={split:{'contracts':len(cache['graphs'][split]),'metadata_example':cache['metadata'][split][0],
        'graph_keys':list(cache['graphs'][split][0].keys())} for split in cache['graphs']}
    write(out/'cache_inventory.json',overview)
    print(json.dumps({'run_id':cfg['run_id'],'gpus':environment['gpus'],'snapshot':sha256(snapshot),'cache':overview},indent=2),flush=True)


if __name__=='__main__':main()
