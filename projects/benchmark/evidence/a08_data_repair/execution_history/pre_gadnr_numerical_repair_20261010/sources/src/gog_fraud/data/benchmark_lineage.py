"""Bind fresh and resumed A08 runs to immutable inputs and execution sources."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from .crypto_raw import sha256, array_hash


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def execution_sources(root):
    root = Path(root)
    import pygod
    package = Path(pygod.__file__).parent
    files = {}
    for directory in ('src/gog_fraud/models/pygod', 'src/gog_fraud/data'):
        for path in sorted((root / directory).glob('*.py')):
            files[str(path.relative_to(root))] = sha256(path)
    path = root / 'src/gog_fraud/pipelines/benchmark_crypto.py'
    files[str(path.relative_to(root))] = sha256(path)
    for directory in ('detector', 'nn', 'utils'):
        for path in sorted((package / directory).rglob('*.py')):
            files['installed_pygod/' + str(path.relative_to(package))] = sha256(path)
    files['environment/locks/benchmark-a03-cuda.lock.txt'] = sha256(root / 'environment/locks/benchmark-a03-cuda.lock.txt')
    return files


def bind_execution(root, config, model_settings):
    """Only before the first prediction; an existing binding must match exactly."""
    root = Path(root)
    evidence = root / 'projects/benchmark/evidence/a08_data_repair'
    value = {'schema_version': 1, 'config_hash': object_hash(config),
             'model_settings_hash': object_hash(model_settings),
             'source_hashes': execution_sources(root)}
    path = evidence / 'execution_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('execution source/config changed after binding; explicit new revision required')
    else:
        if list((evidence / 'run_manifests').glob('*.json')) or list((evidence / 'qualification_runs').glob('*.json')):
            raise ValueError('cannot introduce execution binding after predictions')
        path.write_text(json.dumps(value, indent=2) + '\n')
    return value


def expected_run_identity(root, config, chain, model, seed, *, smoke=False):
    root = Path(root)
    evidence = root / 'projects/benchmark/evidence/a08_data_repair'
    execution = json.loads((evidence / 'execution_manifest.json').read_text())
    if execution['config_hash'] != object_hash(config) or execution['source_hashes'] != execution_sources(root):
        raise ValueError('current execution differs from frozen execution manifest')
    settings = json.loads((evidence / 'model_settings_frozen.json').read_text())
    if execution['model_settings_hash'] != object_hash(settings):
        raise ValueError('frozen model settings changed')
    manifest_path = root / 'local_storage/benchmark/a08_data_repair/frozen' / chain / 'input_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    masks = {name: manifest['arrays'][f'{name}_{seed}'] for name in ('train_mask', 'val_mask', 'test_mask')}
    # Qualification deliberately uses one epoch, with every other resolved setting fixed.
    resolved = json.loads(json.dumps(settings[chain][model]))
    if smoke:
        resolved['caller']['epoch'] = 1
        resolved['scalar_attributes']['epoch'] = 1
        if model == 'DLG-Aug':
            resolved['caller']['l1_epochs'] = 1
            resolved['scalar_attributes']['l1_epochs'] = 1
    return {'input_manifest_hash': sha256(manifest_path), 'split_hash': object_hash(masks),
            'model_config_hash': object_hash(resolved), 'execution_manifest_hash': sha256(evidence / 'execution_manifest.json'),
            'scope': 'qualification' if smoke else 'primary', 'resolved_model_config': resolved}


def verify_run(root, config, record, *, smoke=False, local_artifacts=True):
    chain = record['dataset_id'].split('_')[0]
    expected = expected_run_identity(root, config, chain, record['model'], int(record['seed']), smoke=smoke)
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError('run identity mismatch: ' + key)
    if record.get('status') != 'SUPPORTED_EXACT':
        raise ValueError('run has unresolved failure/support disposition')
    epochs = 1 if smoke else config['epochs'][chain]
    if record.get('completed_epochs') != epochs or record.get('planned_global_epochs') != epochs:
        raise ValueError('incomplete epoch coverage')
    if not local_artifacts:
        return expected
    output = Path(root) / 'local_storage/benchmark/a08_data_repair' / ('smoke' if smoke else 'runs') / record['run_id']
    for name, field in [('scores.npz', 'raw_scores_sha256'), ('final_model_state.pt', 'checkpoint_sha256'),
                        ('metrics.json', 'metrics_sha256'), ('loss_curve.json', 'loss_curve_sha256')]:
        if sha256(output / name) != record[field]:
            raise ValueError('run artifact hash mismatch: ' + name)
    frozen = Path(root) / 'local_storage/benchmark/a08_data_repair/frozen' / chain / 'contract_graph.npz'
    with np.load(output / 'scores.npz', allow_pickle=False) as scores, np.load(frozen, allow_pickle=False) as inputs:
        for name in ('node_ids', 'labels'):
            if not np.array_equal(scores[name], inputs[name]):
                raise ValueError('score population/target differs from frozen input')
        for name in ('val_mask', 'test_mask'):
            if not np.array_equal(scores[name], inputs[f'{name}_{record["seed"]}']):
                raise ValueError('score split differs from frozen input')
        if array_hash(scores['scores']) != record['score_array_hash']:
            raise ValueError('score array identity mismatch')
    return expected
