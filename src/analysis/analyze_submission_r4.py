"""Statistical/calibration/risk analysis of the frozen R4 predictions."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.special import expit

from analysis.submission_r3_statistics import (
    METRICS, paired_bootstrap, mcnemar, holm,
    calibration_metrics, reliability, exact_upper
)
from gog_fraud.production.submission_r4 import (
    read_config, save_json, calibrate, logits, best_threshold, metrics
)
from evidence.experiment_identity import digest


def main():
    cfg = read_config('configs/sci_v3_submission_r4/primary_operating_point.yaml')
    root = Path(cfg['output_root'])
    for sub in ('statistics', 'calibration', 'risk'):
        (root / sub).mkdir(parents=True, exist_ok=True)

    boot = []
    mc = []
    calibration = []
    bins = []
    risks = []
    paired_samples = []

    for seed in cfg['seeds']:
        source = root / 'offline' / f'seed{seed}'
        selection = json.loads((source / 'selection.json').read_text())
        frames = {s: pd.read_csv(source / f'{s}_predictions.csv') for s in ('validation', 'test')}
        test = frames['test']
        y = test.label.to_numpy()

        result, samples = paired_bootstrap(
            y, test.direct_only_score.to_numpy(), test.primary_score.to_numpy(),
            test.direct_only_prediction.to_numpy(), test.primary_prediction.to_numpy(),
            resamples=cfg['statistics']['resamples'], seed=cfg['statistics']['bootstrap_seed'] + seed
        )
        paired_samples.append(samples)
        for metric, values in result.items():
            boot.append({'seed': seed, 'metric': metric, **values})

        mc.append({'seed': seed, **mcnemar(y, test.direct_only_prediction.to_numpy(), test.primary_prediction.to_numpy())})

        for split, frame in frames.items():
            for name, p in [
                ('Raw Level-1 GIN', frame.raw_fast.to_numpy()),
                ('Calibrated Level-1 GIN', frame.direct_only_score.to_numpy()),
                ('Final selective score', frame.primary_score.to_numpy())
            ]:
                calibration.append({'seed': seed, 'split': split, 'score_stage': name, **calibration_metrics(frame.label, p)})
                for row in reliability(frame.label, p, 10, True):
                    bins.append({'seed': seed, 'split': split, 'score_stage': name, **row})

        valid = frames['validation']
        fv = valid.direct_only_score.to_numpy()
        ft = test.direct_only_score.to_numpy()
        dv = calibrate(valid.raw_deep.to_numpy(), selection['deep_map'])
        dt = calibrate(test.raw_deep.to_numpy(), selection['deep_map'])
        w = selection['fast_weight']
        fusedv = expit(w * logits(fv) + (1 - w) * logits(dv))
        fusedt = expit(w * logits(ft) + (1 - w) * logits(dt))

        for budget in cfg['risk']['budgets']:
            margin = float(np.quantile(abs(fv - selection['fast_threshold']), budget, method='higher')) if budget > 0 else -1.
            routev = abs(fv - selection['fast_threshold']) <= margin
            routet = abs(ft - selection['fast_threshold']) <= margin
            threshold, _ = best_threshold(valid.label.to_numpy(), np.where(routev, fusedv, fv))
            point_id = digest({'seed': seed, 'budget': budget, 'margin': margin, 'threshold': threshold, 'source': selection['policy_config_id']})

            for split, frame, fast, fused, route in [
                ('validation', valid, fv, fusedv, routev),
                ('test', test, ft, fusedt, routet)
            ]:
                score = np.where(route, fused, fast)
                pred = score >= threshold
                yy = frame.label.to_numpy()
                for scope in ('pooled', 'ethereum', 'bsc', 'polygon'):
                    mask = np.ones(len(frame), bool) if scope == 'pooled' else frame.sample_id.str.startswith(scope + ':').to_numpy()
                    if not mask.any():
                        continue
                    direct = mask & ~route
                    support = int((direct & (yy == 1)).sum())
                    missed = int((direct & (yy == 1) & ~pred).sum())
                    risks.append({
                        'seed': seed, 'split': split, 'chain_scope': scope, 'requested_budget': budget,
                        'risk_policy_id': point_id, 'margin': margin, 'threshold': threshold,
                        'coverage': float((~route[mask]).mean()), 'N_direct': int(direct.sum()), 'N_deep': int(route[mask].sum()),
                        'N_fraud_direct': support, 'fraud_misses_direct': missed,
                        'observed_fraud_miss_risk': missed / support if support else None,
                        'exact_one_sided_upper': exact_upper(missed, support, cfg['risk']['delta']),
                        'risk_definition': 'P(prediction=benign | direct route, fraud)',
                        'bound_scope': 'pointwise binomial descriptive; not simultaneous or deployment guarantee',
                        'support_status': 'supported' if support else 'undefined / insufficient fraud support',
                        **metrics(yy[mask], score[mask], pred[mask])
                    })
        print(f'Seed {seed} statistics/calibration/risk done', flush=True)

    for row, adjusted in zip(mc, holm([r['p_value'] for r in mc])):
        row['adjusted_p_value'] = float(adjusted)

    pd.DataFrame(mc).to_csv(root / 'statistics/production_mcnemar_holm.csv', index=False)
    pd.DataFrame(boot).to_csv(root / 'statistics/production_seed_pairs.csv', index=False)
    save_json(root / 'statistics/production_paired_bootstrap.json', {
        'rows': boot, 'resamples': cfg['statistics']['resamples'],
        'method': 'within-model class-stratified prediction-paired percentile bootstrap', 'claim_status': 'descriptive',
        'assumptions': 'contracts treated as exchangeable within class; cross-contract dependence can invalidate nominal coverage'
    })

    aggregate = []
    for metric in METRICS:
        values = np.array([r['delta'] for r in boot if r['metric'] == metric])
        aggregate.append({
            'metric': metric, 'mean_delta': float(values.mean()), 'median_delta': float(np.median(values)),
            'sd_delta': float(values.std(ddof=1)), 'positive_seeds': int((values > 0).sum()),
            'zero_seeds': int((values == 0).sum()), 'negative_seeds': int((values < 0).sum()),
            'claim_gate': 'DESCRIPTIVE_POSITIVE' if values.mean() > 0 else 'NO_ROBUST_GAIN', 'claim_status': 'descriptive'
        })

    save_json(root / 'statistics/claim_status.json', {
        'track': 'B', 'predeclared': True, 'aggregate': aggregate,
        'confirmatory_claim_allowed': False,
        'hierarchical_bootstrap': 'not estimated: same test contracts across five models; conditional per-model CIs only'
    })
    pd.DataFrame(calibration).to_csv(root / 'calibration/metrics.csv', index=False)
    pd.DataFrame(bins).to_csv(root / 'calibration/reliability_bins.csv', index=False)
    pd.DataFrame(risks).to_csv(root / 'risk/risk_coverage_dense.csv', index=False)
    print("All statistics, calibration, and risk artifacts saved successfully.", flush=True)


if __name__ == '__main__':
    main()
