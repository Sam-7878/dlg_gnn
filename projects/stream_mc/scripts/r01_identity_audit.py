"""Verify the post-timing checkpoint identity amendment without new score claims."""
from __future__ import annotations
import json
import shutil
import tempfile
from pathlib import Path
from r01_common import ROOT, config, output, read, write, sha256


def main():
    import numpy as np
    from gog_fraud.streaming.selective_engine import SelectivePredictor, SelectiveReplayEngine
    out = output(); source = out/'models/pooled_snapshot_GIN_seed11'
    history = out/'stress/restart_v1'; history.mkdir(exist_ok=True)
    for name in ('restart_audit.json', 'checkpoint_diff.csv'):
        if not (history/name).exists():
            shutil.copy2(out/'stress'/name, history/name)
    observations = []
    with tempfile.TemporaryDirectory(prefix='streammc-identity-') as folder:
        model = Path(folder)/'model'; model.mkdir()
        for name in ('policy_fit.json', 'local.pt', 'relational.pt', 'reference.npz'):
            shutil.copy2(source/name, model/name)
        predictor = SelectivePredictor(model, device='cpu', family='full')
        engine = SelectiveReplayEngine(predictor); checkpoint = Path(folder)/'state.checkpoint'
        engine.checkpoint(checkpoint)
        assert engine.restore(checkpoint) == engine.state.cursor
        fit = read(model/'policy_fit.json')
        fit['fast_map']['b'] += .01
        write(model/'policy_fit.json', fit)
        amended = SelectiveReplayEngine(SelectivePredictor(model, device='cpu', family='full'))
        try:
            amended.restore(checkpoint)
        except ValueError as exc:
            observations.append({'case':'full-lane calibration-only change', 'rejected':True, 'error':str(exc)})
        else:
            raise AssertionError('calibration-only checkpoint mismatch accepted')
        shutil.copy2(source/'policy_fit.json', model/'policy_fit.json')
        for name in ('local.pt', 'reference.npz'):
            with (model/name).open('r+b') as handle:
                first = handle.read(1); handle.seek(0); handle.write(bytes([first[0]^1]))
            try:
                SelectivePredictor(model, device='cpu', family='full')
            except ValueError as exc:
                observations.append({'case':name+' bytes changed', 'rejected':True, 'error':str(exc)})
            else:
                raise AssertionError('input artifact mismatch accepted: '+name)
            shutil.copy2(source/name, model/name)
        p = predictor
        shared = bool(np.shares_memory(p.index.search_embeddings, p.index.tree.data))
        assert shared
        write(out/'audits/memory_accounting_scope.json', {
            'search_embeddings_and_tree_data_share_memory':shared,
            'reported_reference_and_index_component_bytes_are_not_additive':True,
            'shared_bytes':p.index.tree.data.nbytes,
            'logical_state_payload':'fixed-width ASCII field accounting for tested source IDs; not exact Python object/RSS bytes, and not a Unicode allocation bound',
            'identifier_limits':'characters, not encoded UTF-8 bytes',
            'RSS':'sampled process high-water measurement; immutable input preload and instrumentation outside bounded core',
            'GPU':'actual allocated/reserved measurements; no GPU-utilization percentage inferred'})
    write(out/'audits/checkpoint_identity_amendment.json', {
        'run_id':config()['run_id'], 'checkpoint_identity_schema':2,
        'timed_engine_v1_sha256':'8e4e8ce7c41a750298bf2032c3c3cc12691379cccc19742ca3c56197c8ae4c27',
        'timed_runtime_source_archive_sha256':sha256(out/'runtime_sources_v1.tar'),
        'amended_engine_sha256':sha256(ROOT/'src/gog_fraud/streaming/selective_engine.py'),
        'amendment':'constructor validates original weight/reference bytes and includes full policy-fit file SHA for all lanes; predict removes retrieval fallback metadata before composing final timeout/invalid-score outcome; numeric scoring and routing unchanged',
        'observations':observations,
        'runtime_scope':'original 540 offline +18 event runs timed v1; v2 identity amendment is not retroactively described as timed',
        'restart_scope':'rerun before/during/after local-file failure separately under amended identity; see restart_audit.json'})
    print('CHECKPOINT IDENTITY AMENDMENT PASS', len(observations), flush=True)


if __name__ == '__main__':
    main()
