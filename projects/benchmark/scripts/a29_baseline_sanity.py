"""A29-04 baseline soundness checks and equal-budget tuning (protocol: A29_PROTOCOL.yaml).

  python a29_baseline_sanity.py cola     # PyGOD CoLA (primary, paper-like) vs original-design reference
  python a29_baseline_sanity.py ocgnn    # OCGNN hypersphere diagnostics
  python a29_baseline_sanity.py tune GRAPH [MODEL ...]
"""
from __future__ import annotations

import itertools
import sys
import time

from a29_common import LOCAL, RESULTS, dump, read_json
from a29_models import load_primary, metrics, run, val_ap

import numpy as np
import torch

OUT = RESULTS / 'baseline_sanity'
RUNS = LOCAL / 'baseline_sanity'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [42, 43, 44, 45, 46]
CITATION = ['Cora-Syn', 'CiteSeer-Syn', 'PubMed-Syn']


def cola():
    from a29_cola_reference import CoLAReference
    from a29_models import make
    rows = []
    for g in CITATION:
        data, y, masks = load_primary(g)
        for seed in SEEDS:
            # (1) PyGOD primary configuration, with the per-epoch training loss recorded
            for variant, hp, rounds in (('PyGOD-CoLA-primary', {}, 1), ('PyGOD-CoLA-paperlike', {'lr': 1e-3}, 256)):
                folder = RUNS / 'cola' / g / variant / str(seed)
                if (folder / 'result.json').exists():
                    rows.append(read_json(folder / 'result.json')); continue
                epoch = 50 if variant.endswith('primary') else 100
                det = make('CoLA', epoch=epoch, **hp)
                losses = []
                fwd = det.forward_model

                def rec_fwd(d, fwd=fwd, losses=losses, det=det):
                    loss, score = fwd(d)
                    if det.model.training:
                        losses.append(float(loss))
                    return loss, score
                det.forward_model = rec_fwd
                r = run('CoLA', data, y, masks[seed], seed, folder, epoch=epoch, hp=hp, detector=det,
                        extra={'graph': g, 'variant': variant, 'scoring_rounds': rounds}, save_scores=False)
                d = r.pop('_det', None)
                if r['status'] == 'success' and rounds > 1:
                    from torch_geometric.data import Data
                    dd = Data(x=data.x.clone().float(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
                    acc = np.zeros(len(y))
                    for _ in range(rounds):
                        acc += d.decision_function(dd).detach().cpu().numpy()
                    s = acc / rounds
                    r.update(metrics(y, s, masks[seed])); r['val_ap'] = val_ap(y, s, masks[seed])
                    r['score_sd'] = float(s.std()); r['score_unique_frac'] = float(len(np.unique(s)) / len(s))
                r['loss_first_last'] = [losses[0], losses[-1]] if losses else None
                r['loss_min'] = min(losses) if losses else None
                dump(folder / 'result.json', r); rows.append(r)
                del d; torch.cuda.empty_cache()
                print(g, seed, variant, r.get('status'), 'AP', r.get('pr_auc'), 'ROC', r.get('roc_auc'), 'loss', r['loss_first_last'], flush=True)
            # (2) original-design reference
            folder = RUNS / 'cola' / g / 'CoLA-RWR-reference' / str(seed)
            if (folder / 'result.json').exists():
                rows.append(read_json(folder / 'result.json')); continue
            folder.mkdir(parents=True, exist_ok=True)
            t = time.perf_counter()
            ref = CoLAReference(seed=seed).fit(data.x.numpy(), data.edge_index.numpy(), data.num_nodes)
            fit = time.perf_counter() - t
            s = ref.decision_function()
            r = {'graph': g, 'variant': 'CoLA-RWR-reference', 'model': 'CoLA', 'seed': seed, 'status': 'success',
                 'epoch': ref.epoch, 'scoring_rounds': ref.rounds, 'fit_seconds': fit, 'wall_seconds': time.perf_counter() - t,
                 **metrics(y, s, masks[seed]), 'val_ap': val_ap(y, s, masks[seed]),
                 'score_sd': float(s.std()), 'score_unique_frac': float(len(np.unique(s)) / len(s)),
                 'loss_first_last': [ref.loss_history_[0], ref.loss_history_[-1]], 'loss_min': min(ref.loss_history_)}
            dump(folder / 'result.json', r); rows.append(r)
            print(g, seed, 'RWR-ref AP', r['pr_auc'], 'ROC', r['roc_auc'], 'loss', r['loss_first_last'], flush=True)
    dump(OUT / 'cola_runs.json', rows)


def ocgnn():
    rows = []
    for g in CITATION:
        data, y, masks = load_primary(g)
        for seed in SEEDS:
            folder = RUNS / 'ocgnn' / g / str(seed)
            if (folder / 'result.json').exists():
                rows.append(read_json(folder / 'result.json')); continue
            r = run('OCGNN', data, y, masks[seed], seed, folder, extra={'graph': g}, save_scores=False)
            det = r.pop('_det', None)
            if det is not None:
                from torch_geometric.data import Data
                dd = Data(x=data.x.clone().float(), edge_index=data.edge_index.clone(), num_nodes=data.num_nodes)
                s = det.decision_function(dd).detach().cpu().numpy()
                rad = float(det.model.r if not torch.is_tensor(det.model.r) else det.model.r.item())
                dist = s + rad ** 2
                r.update(radius=rad, frac_inside_sphere=float((s <= 0).mean()), dist_mean=float(dist.mean()),
                         dist_sd=float(dist.std()), dist_cv=float(dist.std() / max(dist.mean(), 1e-30)),
                         center_norm=float(det.model.c.norm()))
                dump(folder / 'result.json', r)
            rows.append(r)
            del det; torch.cuda.empty_cache()
            print(g, seed, 'OCGNN AP', r.get('pr_auc'), 'inside', r.get('frac_inside_sphere'), 'dist_cv', r.get('dist_cv'), flush=True)
    dump(OUT / 'ocgnn_runs.json', rows)


GRID_HLE = [dict(hid_dim=h, lr=l, epoch=e) for h, l, e in itertools.product((32, 64, 128), (0.004, 0.001), (50, 100))]
GRID_OCGNN = [dict(hid_dim=h, lr=l, beta=b, epoch=50) for h, l, b in itertools.product((32, 64, 128), (0.004, 0.001), (0.1, 0.5))]


def key(c):  # tie rule: smaller hid, larger lr, fewer epochs, then smaller beta
    return (c['hid_dim'], -c['lr'], c['epoch'], c.get('beta', 0))


def tune(graph, models):
    data, y, masks = load_primary(graph)
    out = []
    for m in models:
        grid = GRID_OCGNN if m == 'OCGNN' else GRID_HLE
        sel_path = RUNS / 'tune' / graph / m / 'selection.json'
        if sel_path.exists():
            sel = read_json(sel_path)
        else:
            trials = []
            for i, c in enumerate(grid):
                folder = RUNS / 'tune' / graph / m / f'cfg{i:02d}' / '42'
                if (folder / 'result.json').exists():
                    r = read_json(folder / 'result.json')
                else:
                    hp = {k: v for k, v in c.items() if k != 'epoch'}
                    r = run(m, data, y, masks[42], 42, folder, epoch=c['epoch'], hp=hp, test=False, save_scores=False,
                            extra={'graph': graph, 'config_index': i, 'config': c})
                    r.pop('_det', None)
                trials.append({'config_index': i, 'config': c, 'status': r['status'], 'val_ap': r.get('val_ap'),
                               'fit_seconds': r.get('fit_seconds')})
                print(graph, m, i, c, r['status'], 'valAP', r.get('val_ap'), flush=True)
            ok = [t for t in trials if t['status'] == 'success' and t['val_ap'] is not None]
            best = max(v['val_ap'] for v in ok)
            winners = sorted([t for t in ok if t['val_ap'] >= best - 1e-12], key=lambda t: key(t['config']))
            sel = {'graph': graph, 'model': m, 'selected': winners[0], 'trials': trials,
                   'primary_default_index': next(i for i, c in enumerate(grid) if c['hid_dim'] == 64 and c['lr'] == 0.004 and c['epoch'] == 50 and c.get('beta', 0.5) == 0.5)}
            dump(sel_path, sel)
        c = sel['selected']['config']
        hp = {k: v for k, v in c.items() if k != 'epoch'}
        for seed in SEEDS:  # test metrics read once, for the selected configuration only
            folder = RUNS / 'tune' / graph / m / 'selected' / str(seed)
            if (folder / 'result.json').exists():
                r = read_json(folder / 'result.json')
            else:
                r = run(m, data, y, masks[seed], seed, folder, epoch=c['epoch'], hp=hp, save_scores=False,
                        extra={'graph': graph, 'config': c, 'role': 'selected'})
                r.pop('_det', None)
            out.append(r)
            print(graph, m, 'selected', c, seed, r['status'], 'AP', r.get('pr_auc'), flush=True)
    dump(OUT / f'tune_{graph}.json', out)


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'cola':
        cola()
    elif stage == 'ocgnn':
        ocgnn()
    elif stage == 'tune':
        tune(sys.argv[2], sys.argv[3:] or ['DOMINANT', 'DLG-Base', 'DLG-Aug', 'AnomalyDAE', 'CoLA', 'OCGNN', 'GADNR'])
