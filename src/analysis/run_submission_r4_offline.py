"""Frozen-model R4 offline evaluation:
Enforces strict historical relation eligibility (t_ref <= t_target),
Branch MC-B deterministic execution (T=1), policy selection, and cross-chain transfers.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from evidence.experiment_identity import digest, file_digest
from gog_fraud.production.submission_r4 import (
    TimestampedFrozenRelations, apply_policy, infer_local, load_models,
    metrics, read_config, save_json, select_policy, set_seed
)


def main():
    cfg = read_config()
    root = Path(cfg['output_root'])
    offline = root / 'offline'
    offline.mkdir(parents=True, exist_ok=True)
    canonical = root / 'canonical'
    canonical.mkdir(parents=True, exist_ok=True)

    config_sha = digest(cfg)
    protocol_path = root / 'protocol_predeclaration.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text())['config_sha256'] == config_sha, 'frozen protocol changed'
    else:
        mc_predec_path = 'configs/sci_v3_submission_r4/mc_identity_predeclaration.json'
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        save_json(protocol_path, {
            'config': cfg,
            'config_sha256': config_sha,
            'git_sha': git_sha,
            'status': 'PREDECLARED_BEFORE_R4_TEST_EVALUATION',
            'branch': 'MC-B',
            'mc_T': 1,
            'source_hashes': {
                cfg[k]: file_digest(cfg[k])
                for k in ('graph_cache', 'raw_events', 'source_config', 'selection_config')
            },
            'mc_predeclaration_hash': file_digest(mc_predec_path)
        })

    freeze_path = root / 'primary_policy_freeze.json'
    if not freeze_path.exists():
        save_json(freeze_path, {
            'mc_T': 1,
            'branch': 'MC-B',
            'selection_partition': 'validation',
            'test_labels_used': False,
            'statistical_track': cfg['statistical_track'],
            'selection_rule': cfg['mc_selection'],
            'config_sha256': config_sha,
            'relation_mode': cfg['relation']['mode']
        })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_json(root / 'environment.json', {
        'platform': sys.platform,
        'python': sys.version,
        'executable': sys.executable,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else None
    })

    print(f'Loading graph cache from {cfg["graph_cache"]} ...', flush=True)
    cache = torch.load(cfg['graph_cache'], map_location='cpu', weights_only=False)
    print('Cache splits loaded:', list(cache['graphs'].keys()), flush=True)

    # Prepare split metadata
    train_meta = cache['metadata']['train']
    train_end = np.array([m['event_end'] for m in train_meta], dtype=np.int64)
    train_start = np.array([m['event_start'] for m in train_meta], dtype=np.int64)
    train_chain = np.array([m['chain'] for m in train_meta], dtype=object)
    train_cid = np.array([m['contract_id'] for m in train_meta], dtype=object)
    train_y = np.array([int(m['label']) for m in train_meta], dtype=np.int64)

    val_meta = cache['metadata']['validation']
    val_graphs = cache['graphs']['validation']
    val_end = np.array([m['event_end'] for m in val_meta], dtype=np.int64)
    val_chain = np.array([m['chain'] for m in val_meta], dtype=object)
    val_cid = np.array([m['contract_id'] for m in val_meta], dtype=object)
    val_y = np.array([int(g.y.item()) for g in val_graphs], dtype=np.int64)
    val_ids = [m['sample_id'] for m in val_meta]

    test_meta = cache['metadata']['test']
    test_graphs = cache['graphs']['test']
    test_end = np.array([m['event_end'] for m in test_meta], dtype=np.int64)
    test_chain = np.array([m['chain'] for m in test_meta], dtype=object)
    test_cid = np.array([m['contract_id'] for m in test_meta], dtype=object)
    test_y = np.array([int(g.y.item()) for g in test_graphs], dtype=np.int64)
    test_ids = [m['sample_id'] for m in test_meta]

    all_rows = []
    all_x_rows = []
    canonical_audits = []

    for seed in cfg['seeds']:
        dest = offline / f'seed{seed}'
        dest.mkdir(exist_ok=True)
        print(f'\n--- Starting Seed {seed} ---', flush=True)

        set_seed(seed)
        local, deep = load_models(cfg, seed, device)

        # 1. Extract and persist timestamped training reference
        ref_path = dest / 'training_reference_temporal.npz'
        if not ref_path.exists():
            print(f'Computing training references for seed {seed} ...', flush=True)
            train_score, train_emb, _ = infer_local(local, cache['graphs']['train'], 1, device)
            np.savez_compressed(
                ref_path,
                embedding=train_emb,
                score=train_score,
                event_end=train_end,
                event_start=train_start,
                chain=train_chain,
                contract_id=train_cid,
                label=train_y,
                split=np.array(['train'] * len(train_y), dtype=object),
                source_hash=file_digest(cfg['graph_cache'])
            )
        else:
            ref_data = np.load(ref_path)
            train_emb = ref_data['embedding']
            train_score = ref_data['score']

        relations = TimestampedFrozenRelations(
            embeddings=train_emb,
            scores=train_score,
            timestamps=train_end,
            chains=train_chain,
            contract_ids=train_cid,
            k=cfg.get('base', {}).get('level2', {}).get('knn_k', 8)
        )

        # 2. Validation evaluation and policy selection
        set_seed(seed * 100 + 1)
        fs_val, em_val, va_val = infer_local(local, val_graphs, 1, device)
        ds_val, val_audits = relations.predict(
            deep, em_val, fs_val, val_end, device,
            target_chains=val_chain, target_contract_ids=val_cid
        )
        for a in val_audits:
            assert a['temporal_violation_count'] == 0, f"Validation temporal violation: {a}"
            assert a['max_selected_reference_cutoff'] <= a['target_cutoff']

        selection = select_policy(fs_val, ds_val, val_y, cfg, 1)
        checkpoint = Path(cfg['checkpoint_root']) / f'seed{seed}'
        selection['model_id'] = digest({name: file_digest(checkpoint / f'{name}.pt') for name in ('level1', 'level2')})
        selection['reference_sha256'] = file_digest(ref_path)
        selection['policy_config_id'] = digest(selection)
        selection['relation_mode'] = 'timestamp_filtered_star'
        save_json(dest / 'selection.json', selection)
        print(f'Seed {seed} Validation Selection F1: {selection["validation_f1"]:.4f}, budget: {selection["requested_budget"]}', flush=True)

        # 3. Test evaluation
        set_seed(seed * 100 + 1 + 10000)
        fs_test, em_test, va_test = infer_local(local, test_graphs, 1, device)
        ds_test, test_audits = relations.predict(
            deep, em_test, fs_test, test_end, device,
            target_chains=test_chain, target_contract_ids=test_cid
        )
        for a in test_audits:
            assert a['temporal_violation_count'] == 0, f"Test temporal violation: {a}"
            assert a['max_selected_reference_cutoff'] <= a['target_cutoff']

        # Save canonical audit records from primary seed
        if seed == cfg['primary_seed']:
            for a in test_audits:
                canonical_audits.append({'split': 'test', **a})
            for a in val_audits:
                canonical_audits.append({'split': 'validation', **a})

        # 4. Generate predictions and compute metrics
        rows = []
        for split, (ids, labels, fs, ds, va) in [
            ('validation', (val_ids, val_y, fs_val, ds_val, va_val)),
            ('test', (test_ids, test_y, fs_test, ds_test, va_test))
        ]:
            frame = pd.DataFrame({'sample_id': ids, 'label': labels, 'raw_fast': fs, 'raw_deep': ds, 'mc_variance': va})
            for policy in ('direct_only', 'primary', 'full_deep'):
                final, route, pred = apply_policy(fs, ds, selection, policy)
                frame[f'{policy}_score'] = final
                frame[f'{policy}_prediction'] = pred
                frame[f'{policy}_route'] = route

                for scope in ('pooled', 'ethereum', 'bsc', 'polygon'):
                    mask = np.ones(len(labels), bool) if scope == 'pooled' else np.array([str(s).startswith(scope + ':') for s in ids])
                    if not mask.any():
                        continue
                    rows.append({
                        'seed': seed, 'split': split, 'chain_scope': scope, 'policy_family': policy,
                        'mc_T': 1, 'deep_rate': float(route[mask].mean()),
                        'N_direct': int((~route[mask]).sum()), 'N_deep': int(route[mask].sum()),
                        'N_fraud_direct': int(((labels == 1) & ~route & mask).sum()),
                        'policy_config_id': selection['policy_config_id'],
                        **metrics(labels[mask], final[mask], pred[mask])
                    })
            frame.to_csv(dest / f'{split}_predictions.csv', index=False)

        save_json(dest / 'completed.json', {'rows': rows})
        all_rows.extend(rows)

        # 5. Cross-Chain Transfers under Strict Cutoff-Safe Filter
        transfers = [
            ('bsc+polygon', 'ethereum', ['bsc', 'polygon']),
            ('ethereum+bsc', 'polygon', ['ethereum', 'bsc']),
            ('ethereum+polygon', 'bsc', ['ethereum', 'polygon'])
        ]
        for sources_str, target_chain, sources_list in transfers:
            mask_tgt = test_chain == target_chain
            tgt_y = test_y[mask_tgt]
            tgt_fs = fs_test[mask_tgt]
            tgt_em = em_test[mask_tgt]
            tgt_end = test_end[mask_tgt]
            tgt_cid = test_cid[mask_tgt]

            ds_x, x_audits = relations.predict(
                deep, tgt_em, tgt_fs, tgt_end, device,
                target_chains=[target_chain] * len(tgt_y),
                target_contract_ids=tgt_cid,
                source_chains=sources_list
            )
            for a in x_audits:
                assert a['temporal_violation_count'] == 0, f"Cross-chain violation: {a}"
                assert a['max_selected_reference_cutoff'] <= a['target_cutoff']

            for policy in ('direct_only', 'primary', 'full_deep'):
                final_x, route_x, pred_x = apply_policy(tgt_fs, ds_x, selection, policy)
                m = metrics(tgt_y, final_x, pred_x)
                all_x_rows.append({
                    'seed': seed,
                    'sources': sources_str,
                    'target_chain': target_chain,
                    'policy_family': policy,
                    'mc_T': 1,
                    'deep_rate': float(route_x.mean()),
                    'min_eligible_references': min(a['eligible_reference_count'] for a in x_audits),
                    'max_eligible_references': max(a['eligible_reference_count'] for a in x_audits),
                    'temporal_violation_count': sum(a['temporal_violation_count'] for a in x_audits),
                    **m
                })

        print(f'Seed {seed} complete.', flush=True)

    # Save aggregated canonical outputs
    pred_df = pd.DataFrame(all_rows)
    pred_df.to_csv(offline / 'prediction_metrics.csv', index=False)
    pred_df.to_csv(canonical / 'prediction_metrics.csv', index=False)
    print(f'Wrote {len(pred_df)} prediction metric rows to canonical and offline.', flush=True)

    # Save relation temporal audit
    audit_df = pd.DataFrame(canonical_audits)
    audit_cols = [
        'target_contract_id', 'target_chain', 'split', 'target_cutoff',
        'eligible_reference_count', 'selected_neighbor_count',
        'max_selected_reference_cutoff', 'temporal_violation_count'
    ]
    audit_df = audit_df[audit_cols]
    audit_df.to_csv(canonical / 'relation_temporal_audit.csv', index=False)
    print(f'Wrote {len(audit_df)} relation temporal audit rows to canonical/relation_temporal_audit.csv.', flush=True)
    assert (audit_df['temporal_violation_count'] == 0).all(), "Temporal violations found in canonical audit!"
    assert (audit_df['max_selected_reference_cutoff'] <= audit_df['target_cutoff']).all(), "Max reference cutoff exceeds target cutoff!"

    # Save cross-chain transfer metrics
    x_df = pd.DataFrame(all_x_rows)
    x_df.to_csv(canonical / 'cross_chain_temporal_r4.csv', index=False)
    # Also save standard cross-chain slice
    cross_df = pred_df[pred_df['chain_scope'] != 'pooled'].copy()
    cross_df.to_csv(canonical / 'cross_chain.csv', index=False)
    print(f'Wrote cross_chain_temporal_r4.csv ({len(x_df)} rows) and cross_chain.csv ({len(cross_df)} rows).', flush=True)


if __name__ == '__main__':
    main()
