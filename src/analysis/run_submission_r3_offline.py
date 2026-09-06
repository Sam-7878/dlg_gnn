"""Frozen-model R3 evaluation: validation selects; test only measures."""
from __future__ import annotations
import platform
import subprocess
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from evidence.experiment_identity import file_digest, digest
from gog_fraud.production.submission_r3 import (CONFIG, FrozenRelations, apply_policy,
    infer_local, load_models, metrics, read_config, save_json, select_policy, set_seed)


def main():
    cfg = read_config(); root = Path(cfg['output_root']); offline = root / 'offline'
    offline.mkdir(parents=True, exist_ok=True)
    config_sha = digest(cfg)
    protocol = root / 'protocol_predeclaration.json'
    if protocol.exists():
        assert json.loads(protocol.read_text())['config_sha256'] == config_sha, 'frozen protocol changed'
    else:
        save_json(protocol, {'config': cfg, 'config_sha256': config_sha,
            'git_sha': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'status': 'PREDECLARED_BEFORE_R3_TEST_EVALUATION',
            'source_hashes': {cfg[k]: file_digest(cfg[k]) for k in ('graph_cache','raw_events','source_config','selection_config')}})
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_json(root / 'environment.json', {'platform':platform.platform(),
        'os_release':Path('/etc/os-release').read_text(), 'python':sys.version,
        'executable':sys.executable,'torch':torch.__version__, 'cuda':torch.version.cuda,
        'device':str(device), 'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None})
    cache = torch.load(cfg['graph_cache'], map_location='cpu', weights_only=False)
    print('cache keys', list(cache), flush=True)
    freeze_path = root / 'primary_policy_freeze.json'
    all_rows = []
    for seed in cfg['seeds']:
        dest = offline / f'seed{seed}'; dest.mkdir(exist_ok=True)
        if (dest/'completed.json').exists():
            all_rows.extend(json.loads((dest/'completed.json').read_text())['rows']); continue
        set_seed(seed); local, deep = load_models(cfg, seed, device)
        train_score, train_emb, _ = infer_local(local, cache['graphs']['train'], 1, device)
        np.savez_compressed(dest/'training_reference.npz', embedding=train_emb, score=train_score)
        relations = FrozenRelations(train_emb, train_score, cfg['base']['level2']['knn_k'])
        validation = cache['graphs']['validation']
        yv = np.array([int(g.y.item()) for g in validation])
        candidates = cfg['mc_selection']['candidates'] if not freeze_path.exists() else [json.loads(freeze_path.read_text())['mc_T']]
        choices = {}; valid_values = {}
        for T in candidates:
            set_seed(seed*100+T)
            fs, em, va = infer_local(local, validation, T, device)
            ds = relations.predict(deep, em, fs, device)
            choices[T] = select_policy(fs, ds, yv, cfg, T); valid_values[T] = (fs,ds,va)
            print(f'seed={seed} validation T={T} F1={choices[T]["validation_f1"]:.6f}', flush=True)
        if not freeze_path.exists():
            assert seed == cfg['primary_seed']
            best = max(c['validation_f1'] for c in choices.values())
            T = min(t for t,c in choices.items() if c['validation_f1'] >= best-cfg['mc_selection']['tolerance'])
            save_json(freeze_path, {'mc_T':T, 'selection_partition':'validation',
                'test_labels_used':False, 'statistical_track':cfg['statistical_track'],
                'selection_rule':cfg['mc_selection'], 'candidate_results':choices,
                'config_sha256':config_sha,'relation_mode':cfg['relation']['mode']})
        else:
            T = json.loads(freeze_path.read_text())['mc_T']
        selection = choices[T]
        checkpoint = Path(cfg['checkpoint_root'])/f'seed{seed}'
        selection['model_id'] = digest({name:file_digest(checkpoint/f'{name}.pt') for name in ('level1','level2')})
        selection['reference_sha256'] = file_digest(dest/'training_reference.npz')
        selection['policy_config_id'] = digest(selection)
        save_json(dest/'selection.json', selection)  # persisted before any test prediction
        rows = []
        for split in ('validation','test'):
            graphs = cache['graphs'][split]; labels = np.array([int(g.y.item()) for g in graphs])
            ids = [g['sample_id'] for g in cache['metadata'][split]]
            if split == 'validation': fs, ds, va = valid_values[T]
            else:
                set_seed(seed*100+T+10000)
                fs, em, va = infer_local(local, graphs, T, device)
                ds = relations.predict(deep, em, fs, device)
            frame = pd.DataFrame({'sample_id':ids,'label':labels,'raw_fast':fs,'raw_deep':ds,'mc_variance':va})
            for policy in ('direct_only','primary','full_deep'):
                final, route, pred = apply_policy(fs,ds,selection,policy)
                frame[f'{policy}_score']=final; frame[f'{policy}_prediction']=pred; frame[f'{policy}_route']=route
                for scope in ('pooled','ethereum','bsc','polygon'):
                    mask = np.ones(len(labels),bool) if scope=='pooled' else np.array([str(s).startswith(scope+':') for s in ids])
                    if not mask.any(): continue
                    rows.append({'seed':seed,'split':split,'chain_scope':scope,'policy_family':policy,
                        'mc_T':T,'deep_rate':float(route[mask].mean()), 'N_direct':int((~route[mask]).sum()),
                        'N_deep':int(route[mask].sum()),'N_fraud_direct':int(((labels==1)&~route&mask).sum()),
                        'policy_config_id':selection['policy_config_id'], **metrics(labels[mask],final[mask],pred[mask])})
            frame.to_csv(dest/f'{split}_predictions.csv', index=False)
        save_json(dest/'completed.json', {'rows':rows}); all_rows.extend(rows)
        print(f'seed {seed} complete', flush=True)
    pd.DataFrame(all_rows).to_csv(offline/'prediction_metrics.csv', index=False)


if __name__ == '__main__': main()
