"""New explicit deterministic batch-measurement identity; preserve all v1 data."""
import tarfile
from r01_common import ROOT,config,output,write,sha256
from r01_replay import offline


def main():
    out=output();cfg=config();names=['src/gog_fraud/streaming/selective_engine.py','src/gog_fraud/streaming/selective_state.py',
        'src/gog_fraud/streaming/selective.py','src/gog_fraud/data/level2/relation_builder.py',
        'projects/stream_mc/scripts/r01_replay.py','projects/stream_mc/scripts/r01_offline_v2.py','projects/stream_mc/configs/r01.json']
    hashes={name:sha256(ROOT/name) for name in names};archive=out/'runtime_sources_v2.tar'
    if not archive.exists():
        with tarfile.open(archive,'w') as bundle:
            for name in names:bundle.add(ROOT/name,arcname=name)
    offline(cfg)
    assert hashes=={name:sha256(ROOT/name) for name in names},'runtime source changed during measurement'
    write(out/'audits/runtime_v2_source_identity.json',{'measurement_run_id':cfg['run_id']+'_offline_v2','model_run_id':cfg['run_id'],
        'executed_source_hashes':hashes,'source_archive_sha256':sha256(archive),
        'scope':'540 new actual measurements with one CPU thread and requested deterministic algorithms; old v1 batch outputs retained separately, not overwritten; models/calibrators/routers unchanged'})
    print('DETERMINISTIC OFFLINE V2 COMPLETE',flush=True)


if __name__=='__main__':main()
