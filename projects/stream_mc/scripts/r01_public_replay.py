"""Opt-in systems replay from the alias inputs and disclosed frozen tensors."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
from r01_common import PROJECT,config,write,sha256
from r01_public import verified_archive


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--confirm-full',action='store_true');parser.add_argument('--population',choices=['prefix','long'],default='prefix');parser.add_argument('--policy',choices=['local','margin','full'],default='margin');args=parser.parse_args()
    if not args.confirm_full:raise SystemExit('Actual GPU replay requires --confirm-full; no inference started.')
    import r01_replay as replay
    cfg=config();original_run=cfg['run_id'];cfg['run_id']='selectivestream_r01_public_alias_replay';cfg['runtime']['repeats']=1
    dest=PROJECT/'results/public_alias_replay';model_name='pooled_snapshot_GIN_seed11';model=dest/'models'/model_name;model.mkdir(parents=True,exist_ok=True)
    with verified_archive()[0] as archive:
        for name in ('local.pt','relational.pt','policy_fit.json'):(model/name).write_bytes(archive.read('models/'+model_name+'/'+name))
        (model/'reference.npz').write_bytes(archive.read('models/'+model_name+'/reference_alias.npz'))
        import json
        fit=json.loads((model/'policy_fit.json').read_text());fit['reference_manifest_hash']=sha256(model/'reference.npz');fit['run_id']=cfg['run_id'];write(model/'policy_fit.json',fit)
        event_bytes=archive.read('inputs/events.csv')
    replay.output=lambda:dest;replay.model_dir=lambda cfg,seed=None:model
    def inputs(cfg,count=None):
        records=list(csv.DictReader(io.StringIO(event_bytes.decode())))
        for r in records[:count]:
            yield {**r,'sequence_id':int(r['sequence_id']),'timestamp':int(r['timestamp']),'label':int(r['label'])}
    replay.events=inputs
    result=replay.replay(cfg,args.policy,args.population,0)
    write(dest/'alias_input_identity.json',{'source_run':original_run,'run_id':cfg['run_id'],'archive_sha256':sha256(PROJECT/'evidence/r01_numeric_evidence.zip'),
        'event_input_sha256':hashlib.sha256(event_bytes).hexdigest(),'original_model_tensors':'unchanged bytes; reference identity and fit file are explicit derivative aliases',
        'scope':'same retained event order/equality and numeric model/reference data; new timing run, not original event/source identity or live detection'})
    print('PUBLIC ALIAS REPLAY COMPLETE',result['N_measured'],flush=True)


if __name__=='__main__':main()
