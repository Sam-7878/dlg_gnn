"""Freeze exact training sources; refuse to pretend a changed file was executed."""
import zipfile
from r01_common import ROOT,CONFIG,config,output,read,write,sha256


def main():
    out=output();records=[read(p) for p in sorted((out/'models').glob('*/complete.json'))]
    if len(records)!=25:raise ValueError('source freeze requires all 25 completed model identities')
    required=['projects/stream_mc/scripts/r01_train.py','projects/stream_mc/scripts/r01_common.py',
        'src/gog_fraud/streaming/selective.py','src/gog_fraud/data/level2/relation_builder.py']
    frozen={}
    for path in required:
        hashes={r['source_hashes_at_process_start'][path] for r in records}
        if len(hashes)!=1 or sha256(ROOT/path)!=next(iter(hashes)):raise ValueError('training-source drift: '+path)
        frozen[path]=next(iter(hashes))
    archive=out/'training_sources.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for path in required:bundle.write(ROOT/path,path)
        bundle.write(CONFIG,'projects/stream_mc/configs/r01.json')
    write(out/'training_source_manifest.json',{'run_id':config()['run_id'],'scope':'four execution-critical project/core training sources verified against all 25 process-start records',
        'baseline_commit':read(out/'execution_plan.json')['scientific_baseline_commit'],'dirty_revision_source_hashes':frozen,'archive_sha256':sha256(archive),
        'not_claimed_clean_commit':True,'development_sidecar_hashes_in_model_records':'workspace snapshot at process start; not all were imported by training'})
    print('TRAINING SOURCE FREEZE PASS',sha256(archive))


if __name__=='__main__':main()
