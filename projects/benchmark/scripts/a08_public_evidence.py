#!/usr/bin/env python3
"""Curate and verify current numeric evidence without loading private tensors.

This checks public artifact integrity and run identity, not private raw-score
recomputation, scientific acceptance, author approval, or publication.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[3]
BASE = 'projects/benchmark/evidence/a08_data_repair/'
E = ROOT / BASE
BUNDLE = ROOT / 'projects/benchmark/evidence/a08_public_numeric_evidence.zip'
OUTER = ROOT / 'projects/benchmark/evidence/a08_public_release.json'
PRIVATE_SUFFIXES = {'.tex', '.bib', '.cls', '.sty', '.pdf', '.pt', '.pkl', '.pickle', '.npz', '.npy', '.whl'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_name(name):
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and '\\' not in name,
            'unsafe public payload path: ' + name)
    require(p.suffix.lower() not in PRIVATE_SUFFIXES, 'private/binary payload: ' + name)
    require(not any(part in {'local_storage', 'paper', 'manuscript_base_a06'} for part in p.parts)
            and 'build_manuscript' not in name and 'tracked-working-tree.patch' not in name,
            'private manuscript or working diff payload: ' + name)
    return p


def verify_numeric_identity(read):
    """Validate public identities using bytes from disk or the curated ZIP."""
    obj = lambda path: json.loads(read(BASE + path))
    gate = obj('audit/G7.json')
    require(gate.get('status') == 'PASS' and not gate.get('is_template'), 'G7 not approved')
    for field, path in [('approved_registry_sha256', 'approved_registry.json'),
                        ('number_registry_sha256', 'number_registry.json'),
                        ('recompute_audit_sha256', 'audit/metric_recompute_audit.json'),
                        ('statistics_sha256', 'statistics/statistics_s1_s4.json'),
                        ('historical_reuse_sha256', 'audit/historical_reuse.json')]:
        require(digest(read(BASE + path)) == gate[field], 'G7 binding drift: ' + path)
    require(digest(read('projects/benchmark/scripts/a08_aggregate.py')) == gate['aggregate_source_sha256'],
            'aggregate source differs from approved generation')
    registry = obj('approved_registry.json')
    records = registry['records']
    current = [r for r in records if r.get('evidence_tier') == 'a08_raw_score_recomputable']
    historic = [r for r in records if r.get('evidence_tier') == 'historical_preserved']
    require(len(current) == registry['current_new_records'] == 105, 'current run coverage incomplete')
    require(not any(r['dataset'] in {'Ethereum', 'BSC', 'Polygon'} for r in historic),
            'historical crypto mixed with corrected primary records')
    planned = list(csv.DictReader(io.StringIO(read(BASE + 'planned_cells.csv').decode())))
    expected = {(r['dataset_id'], r['model'], int(r['seed'])) for r in planned}
    require(len(expected) == len(planned) == 105, 'empty/duplicate planned cells')
    seen = set()
    execution_bytes = read(BASE + 'execution_manifest.json')
    execution = json.loads(execution_bytes)
    settings = obj('model_settings_frozen.json')
    config = json.loads(read('configs/benchmark/a08_crypto_clean_v1.yaml'))
    object_hash = lambda value: digest(json.dumps(value, sort_keys=True).encode())
    require(execution['config_hash'] == object_hash(config)
            and execution['model_settings_hash'] == object_hash(settings), 'frozen settings/config drift')
    inputs = obj('input_manifest.json')['datasets']
    require(len(inputs) == 3, 'missing corrected chain input')
    for record in current:
        key = record['dataset_version'], record['model'], int(record['seed'])
        require(key in expected and key not in seen, 'unknown or duplicate current identity')
        seen.add(key)
        run = obj('run_manifests/' + record['run_id'] + '.json')
        chain = key[0].split('_')[0]
        input_doc = next(m for m in inputs if m['dataset_id'] == key[0])
        construction=read(BASE+'construction_manifests/'+chain+'/observables_manifest.json')
        require(digest(construction)==input_doc['source_manifest_hash'],'raw construction-manifest binding drift')
        construction_doc=json.loads(construction)
        require(construction_doc['source_zip_sha256']==input_doc['raw_zip_sha256'] and construction_doc['labels_accessed'] is False,
                'raw source or label-inaccessible construction identity drift')
        # Consolidated manifests preserve the same exact fields as local per-chain JSON.
        manifest_hash = digest((json.dumps(input_doc, indent=2) + '\n').encode())
        require(run['input_manifest_hash'] == record['input_manifest_hash'] == manifest_hash,
                'new metadata bound to old input: ' + record['run_id'])
        masks = {name: input_doc['arrays'][f'{name}_{key[2]}']
                 for name in ('train_mask', 'val_mask', 'test_mask')}
        require(run['split_hash'] == record['split_hash'] == object_hash(masks), 'seed split drift')
        require(run['model_config_hash'] == record['model_config_hash'] == object_hash(settings[chain][key[1]]),
                'model configuration drift')
        require(run['execution_manifest_hash'] == digest(execution_bytes)
                and run['code_hashes'] == execution['source_hashes'], 'execution/source identity drift')
        require(run['scope'] == 'primary' and run['status'] == 'SUPPORTED_EXACT'
                and run['completed_epochs'] == run['planned_global_epochs'] == config['epochs'][chain],
                'qualification or partial run mixed into primary')
        require(run['raw_scores_sha256'] == record['raw_score_hash']
                and run['metrics_sha256'] == record['metric_json_hash'], 'score/metric hash drift')
        for field, source in [('pr_auc', 'ap'), ('roc_auc', 'roc_auc')]:
            require(record[field] == run['metrics'][source], 'scalar metric drift')
        require(record['validation_f1'] == run['metrics']['thresholded']['f1'], 'threshold metric drift')
    require(seen == expected, 'current planned coverage missing')
    return {'scope': 'current numeric integrity and identities; restricted raw scores not recomputed',
            'current_runs': len(current), 'historical_records': len(historic),
            'historical_successes': sum(r['status'] == 'success' for r in historic)}


def build():
    require((E / 'audit/G7.json').is_file(), 'actual G7 required before public packaging')
    for input_doc in json.loads((E/'input_manifest.json').read_text())['datasets']:
        chain=input_doc['dataset_id'].split('_')[0]
        data=(ROOT/'local_storage/benchmark/a08_data_repair/build_a'/chain/'observables_manifest.json').read_bytes()
        require(digest(data)==input_doc['source_manifest_hash'],'frozen construction metadata differs')
        copy=E/'construction_manifests'/chain/'observables_manifest.json'
        require(not copy.exists() or copy.read_bytes()==data,'existing construction manifest snapshot differs')
        copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(data)
    schema={'source_manifest_hash':'per-chain exact raw observable construction manifest SHA256, resolved under construction_manifests/<chain>/observables_manifest.json; not the consolidated source_manifest.json',
            'consolidated_source_manifest':'separately bound by G4.source_manifest_sha256',
            'privacy':'array hashes, dimensions, source archive/code/member hashes only; raw member addresses and scores remain local'}
    (E/'input_schema_notes.json').write_text(json.dumps(schema,indent=2)+'\n')
    verify_numeric_identity(lambda name: (ROOT / name).read_bytes())
    historical_note=ROOT/'evaluation/benchmark/v2/a05/dgraphfin_hardware_runtime_note.json'
    historical_bytes=historical_note.read_bytes()
    historical_copy=E/'historical_hardware_runtime_note.json'
    require(not historical_copy.exists() or historical_copy.read_bytes()==historical_bytes,'historical hardware snapshot drift')
    historical_copy.write_bytes(historical_bytes)
    runs=[json.loads(p.read_text()) for p in sorted((E/'run_manifests').glob('*.json'))]
    hardware={'scope':'current run-bound actual device/framework fields; historical clock-offset report separately dated',
              'current_device_names':sorted({r['device'] for r in runs}),
              'current_torch_versions':sorted({r['torch'] for r in runs}),
              'current_cuda_runtimes':sorted({r['cuda_runtime'] for r in runs}),
              'current_run_count':len(runs),'historical_note_sha256':digest(historical_bytes),
              'reported_prior_memory_clock_offset_mhz':json.loads(historical_bytes)['reported_memory_clock_adjustment_mhz'],
              'current_offset_independently_verified':False,
              'timing_scope':'current fresh-run fit-plus-inference segments cover complete declared budgets; old resumed DGraphFin segment times are not total training times',
              'telemetry_policy':'author-local per-run and 30-second active NVIDIA telemetry; actual allocation/reservation peaks recorded per run'}
    (E/'hardware_runtime_context.json').write_text(json.dumps(hardware,indent=2)+'\n')
    # Deliberate allowlist: no recursively gathered author workspace or raw inputs.
    paths = [E / name for name in (
        'input_manifest.json', 'split_manifest.json', 'planned_cells.csv', 'model_settings_frozen.json',
        'execution_manifest.json', 'feature_spec.yaml', 'relation_spec.yaml', 'source_manifest.json',
        'old_input_denylist.json', 'approved_registry.json', 'number_registry.json',
        'threshold_audit.csv', 'confusion_matrices.csv','historical_hardware_runtime_note.json','hardware_runtime_context.json','input_schema_notes.json')]
    audits = ('G0.json', 'G1.json', 'G2.json', 'G3.json', 'G4.json', 'G5.json', 'G6.json', 'G7.json',
              'regression_tests.json', 'counterfactual_and_rebuild.json', 'raw_full_counterfactual.json',
              'cuda_exactness.json', 'historical_reuse.json', 'metric_recompute_audit.json',
              'raw_feature_examples.json', 'family_impact.json', 'preprediction_settings_metadata_amendment.json')
    paths += [E / 'audit' / name for name in audits]
    paths += [E/'audit'/name for name in ('gadnr_numerical_qualification.json','gadnr_numerical_repair_disposition.json')]
    for directory in ('run_manifests', 'qualification_runs', 'tables', 'statistics', 'source_versions','construction_manifests'):
        paths += sorted((E / directory).rglob('*'))
    paths += [ROOT / 'configs/benchmark/a08_crypto_clean_v1.yaml',
              ROOT / 'projects/benchmark/protocols/A08_DATA_REPAIR_AMENDMENT.md',
              ROOT / 'projects/benchmark/protocols/A08_NUMERICAL_REPAIR_AMENDMENT_2026-10-10.md',
              ROOT / 'projects/benchmark/scripts/a08_aggregate.py']
    files = {}
    for p in paths:
        if p.is_dir():
            continue
        name = p.relative_to(ROOT).as_posix()
        safe_name(name)
        files[name] = p.read_bytes()
    # Preserve exact scientific execution sources, including installed PyGOD bytes.
    # Import only when building in the author environment; verify has stdlib only.
    import pygod
    package = Path(pygod.__file__).parent
    license_path = package.parent / 'pygod-1.1.0.dist-info/LICENSE'
    require(license_path.is_file(), 'installed PyGOD license missing from source snapshot')
    files[BASE + 'execution_sources/PyGOD_LICENSE.txt'] = license_path.read_bytes()
    source_map = {}
    for name, sha in json.loads(files[BASE + 'execution_manifest.json'])['source_hashes'].items():
        p = package / name.removeprefix('installed_pygod/') if name.startswith('installed_pygod/') else ROOT / name
        data = p.read_bytes()
        require(digest(data) == sha, 'scientific source changed: ' + name)
        snapshot = BASE + 'execution_sources/' + sha + '/' + name
        safe_name(snapshot)
        files[snapshot] = data
        source_map[name] = {'path': snapshot, 'sha256': sha}
    files[BASE + 'execution_sources/manifest.json'] = (json.dumps(source_map, indent=2) + '\n').encode()
    for name, data in files.items():
        if name.startswith(BASE + 'execution_sources/'):
            path = ROOT / name
            require(not path.exists() or path.read_bytes() == data, 'existing execution snapshot differs')
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(data)
    manifest = {'schema_version': 1, 'revision': 'a08', 'scope': 'curated numeric evidence and exact scientific sources',
                'private_scores_checkpoints_node_ids': 'not distributed; author-local raw recomputation required',
                'files': [{'path': name, 'bytes': len(data), 'sha256': digest(data)} for name, data in sorted(files.items())]}
    manifest_data = (json.dumps(manifest, indent=2) + '\n').encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        info = zipfile.ZipInfo('release_manifest.json', date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, manifest_data)
    bundle_data = buffer.getvalue()
    require(not BUNDLE.exists() or BUNDLE.read_bytes() == bundle_data,
            'existing A08 candidate differs; preserve it and declare a new release revision')
    if not BUNDLE.exists():
        BUNDLE.write_bytes(bundle_data)
    outer = {'schema_version': 1, 'revision': 'a08', 'publication_status': 'LOCAL_CANDIDATE_NOT_PUBLISHED',
             'bundle_path': BUNDLE.relative_to(ROOT).as_posix(), 'bundle_sha256': digest(BUNDLE.read_bytes()),
             'manifest_sha256': digest(manifest_data), 'payload_count': len(files)}
    OUTER.write_text(json.dumps(outer, indent=2) + '\n')
    return verify()


def verify():
    outer = json.loads(OUTER.read_text())
    require(digest(BUNDLE.read_bytes()) == outer['bundle_sha256'], 'current ZIP drift')
    with zipfile.ZipFile(BUNDLE) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'duplicate ZIP payload')
        for name in names:
            safe_name(name)
        manifest_bytes = archive.read('release_manifest.json')
        require(digest(manifest_bytes) == outer['manifest_sha256'], 'manifest drift')
        manifest = json.loads(manifest_bytes)
        require(len(manifest['files']) == outer['payload_count'] > 0, 'empty/incorrect public expected list')
        listed = {f['path'] for f in manifest['files']}
        require(len(listed) == len(manifest['files']) and listed == set(names) - {'release_manifest.json'},
                'manifest/payload membership drift')
        for entry in manifest['files']:
            data = archive.read(entry['path'])
            require(len(data) == entry['bytes'] and digest(data) == entry['sha256'], 'payload drift')
            require((ROOT / entry['path']).is_file()
                    and digest((ROOT / entry['path']).read_bytes()) == entry['sha256'],
                    'working evidence differs from bundled identity: ' + entry['path'])
        sources = json.loads(archive.read(BASE + 'execution_sources/manifest.json'))
        execution = json.loads(archive.read(BASE + 'execution_manifest.json'))
        require(set(sources) == set(execution['source_hashes']), 'missing execution source snapshot')
        for name, item in sources.items():
            require(digest(archive.read(item['path'])) == item['sha256'] == execution['source_hashes'][name],
                    'execution source snapshot drift')
            if not name.startswith('installed_pygod/'):
                require(digest((ROOT / name).read_bytes()) == item['sha256'], 'current scientific source drift')
        result = verify_numeric_identity(archive.read)
    return {'status': 'PASS', 'payload_count': outer['payload_count'], 'bundle_sha256': outer['bundle_sha256'], **result}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', action='store_true')
    args = ap.parse_args()
    print(json.dumps(build() if args.build else verify(), indent=2))


if __name__ == '__main__':
    main()
