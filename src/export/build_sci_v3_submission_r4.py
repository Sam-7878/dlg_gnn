"""Build R4 canonical registries, short publication tables, and source-linked claims."""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from evidence.experiment_identity import ExperimentIdentity, digest, file_digest
from gog_fraud.production.submission_r4 import read_config, save_json, metrics


def load(path):
    return json.loads(Path(path).read_text())


def clean(row):
    return {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in row.items()}


class Builder:
    def __init__(self):
        self.cfg = read_config('configs/sci_v3_submission_r4/primary_operating_point.yaml')
        self.root = Path(self.cfg['output_root'])
        self.protocol = load(self.root / 'protocol_predeclaration.json')
        self.selections = {seed: load(self.root / f'offline/seed{seed}/selection.json') for seed in self.cfg['seeds']}
        self.registry = []
        self.claims = []
        self.table_sources = {}
        self._pop_cache = {}
        for name in ('canonical', 'tables', 'figures', 'profiling'):
            (self.root / name).mkdir(exist_ok=True)

    def identify(self, row, kind, lane='offline_contract_detection', stage=''):
        row = clean(row.copy())
        seed = int(row.get('seed', 11))
        s = self.selections.get(seed, self.selections[11])
        split = row.get('split', 'test')
        scope = row.get('chain_scope', 'pooled')
        policy = row.get('policy_family', 'primary')
        unit = {
            'offline_contract_detection': 'contract',
            'raw_event_runtime': 'raw_event_runtime_trace',
            'integrated_streaming': 'event_with_inherited_contract_label'
        }.get(lane, 'contract')

        if lane == 'offline_contract_detection':
            ckey = (seed, split, scope)
            if ckey not in self._pop_cache:
                pred_path = self.root / f'offline/seed{seed}/{split}_predictions.csv'
                if pred_path.exists():
                    frame = pd.read_csv(pred_path)
                    if scope != 'pooled':
                        frame = frame[frame.sample_id.str.startswith(scope + ':')]
                    self._pop_cache[ckey] = digest(frame.sample_id.tolist())
                else:
                    self._pop_cache[ckey] = digest(f'{seed}_{split}_{scope}')
            population = self._pop_cache[ckey]
        else:
            population = row.get('prefix_id', 'not_applicable')

        policy_id = digest({'frozen_selection': s['policy_config_id'], 'policy': policy})
        identity = ExperimentIdentity(
            lane, unit, population,
            'frozen-GoG-bounded-cache-' + self.protocol['source_hashes'][self.cfg['graph_cache']][:12],
            split, scope, seed, s['model_id'], policy, policy_id,
            digest({'fast': s['fast_map'], 'deep': s['deep_map']}),
            digest({k: s[k] for k in ('fast_threshold', 'final_threshold', 'full_threshold')}),
            digest(s['fast_weight']), s['mc_T'],
            row.get('prefix_id', 'not_applicable'), str(row.get('repeat', 'not_applicable')),
            int(row.get('warmup_events', 0)),
            kind + (':' + stage if stage else ''),
            self.protocol['git_sha'], self.protocol['config_sha256']
        ).row()
        result = {
            **row, **identity,
            'frozen_selection_id': s['policy_config_id'],
            'display_name': self.cfg['display_names'].get(policy, policy),
            'scientific_scope': 'held-out contract; strict historical reference eligibility' if lane == 'offline_contract_detection' else 'retrospective replay systems workload'
        }
        self.registry.append(result)
        return result

    def claim(self, name, rows, metric, status='descriptive'):
        assert rows and len({r['evidence_lane'] for r in rows}) == 1
        assert len({r['evaluation_population_id'] for r in rows}) == 1
        values = np.array([r[metric] for r in rows], dtype=float)
        assert np.isfinite(values).all()
        value = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else None
        first = rows[0]
        record = {
            'claim_id': name, 'display_text': name.replace('_', ' '), 'evidence_lane': first['evidence_lane'],
            'experiment_id': [r['experiment_id'] for r in rows], 'population_id': first['evaluation_population_id'],
            'policy_config_id': [r['policy_config_id'] for r in rows], 'prediction_unit': first['prediction_unit'],
            'metric': metric, 'raw_value': value, 'formatted_value': f'{value:.3f}', 'sd': sd,
            'ci_low': None, 'ci_high': None, 'p_value': None, 'adjusted_p_value': None, 'claim_status': status,
            'canonical_row_id': [r['canonical_row_id'] for r in rows],
            'config_sha256': first['config_sha256'], 'git_sha': first['git_sha']
        }
        self.claims.append(record)
        return value

    def table(self, name, caption, headers, rows, source_ids, note=''):
        assert 4 <= len(headers) <= 8
        text = '% canonical_row_id: ' + ','.join(str(s) for s in source_ids[:50]) + '\n'
        text += r'\begin{table*}[t]\centering\small' + '\n' + r'\caption{' + caption + r'}\label{tab:' + name + '}\n'
        text += r'\begin{tabular}{' + 'l' + 'r' * (len(headers) - 1) + r'}\toprule' + '\n'
        text += ' & '.join(headers) + r'\\\midrule' + '\n'
        text += '\n'.join(' & '.join(str(v) for v in row) + r'\\' for row in rows)
        text += '\n' + r'\bottomrule\end{tabular}' + '\n'
        if note:
            text += r'\par\medskip\begin{minipage}{0.96\textwidth}\footnotesize ' + note + r'\end{minipage}' + '\n'
        text += r'\end{table*}' + '\n'
        (self.root / 'tables' / f'table_{name}.tex').write_text(text)
        pd.DataFrame(rows, columns=headers).assign(canonical_row_id=';'.join(str(s) for s in source_ids[:50])).to_csv(self.root / 'tables' / f'table_{name}.csv', index=False)
        self.table_sources[name] = [str(s) for s in source_ids[:50]]

    def build(self):
        pred = [self.identify(r, 'predictive') for r in pd.read_csv(self.root / 'offline/prediction_metrics.csv').to_dict('records')]
        cal = [self.identify(r, 'calibration', stage=r['score_stage']) for r in pd.read_csv(self.root / 'calibration/metrics.csv').to_dict('records')]
        stat = [self.identify(r, 'paired_statistics', stage=r['metric']) for r in pd.read_csv(self.root / 'statistics/production_seed_pairs.csv').to_dict('records')]
        risk = [self.identify(r, 'risk_sweep', stage=r['risk_policy_id']) for r in pd.read_csv(self.root / 'risk/risk_coverage_dense.csv').to_dict('records')]

        runtime = []
        stream = []
        for name, lane, target in [('runtime', 'raw_event_runtime', runtime), ('streaming', 'integrated_streaming', stream)]:
            for path in sorted((self.root / name).glob('*/summary.json')):
                r = load(path)
                r.pop('diagnostic_metrics', None)
                target.append(self.identify(r, 'measured_e2e', lane))
        assert len(runtime) == 10 and len(stream) == 2, 'incomplete measured workload; refuse final export'

        # Canonical audit dataframe
        audit_records = pd.read_csv(self.root / 'canonical/relation_temporal_audit.csv')

        registries = {
            'prediction_metrics': pred,
            'routing_metrics': risk,
            'runtime_metrics': runtime,
            'streaming_metrics': stream,
            'calibration_metrics': cal,
            'statistics': stat,
            'cross_chain': [r for r in pred if r['chain_scope'] != 'pooled']
        }
        for name, rows in registries.items():
            pd.DataFrame(rows).to_csv(self.root / 'canonical' / f'{name}.csv', index=False)
        pd.DataFrame(self.registry).to_csv(self.root / 'canonical/experiment_registry.csv', index=False)

        test = [r for r in pred if r['split'] == 'test' and r['chain_scope'] == 'pooled']

        def choose(rows, **filters):
            return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]

        def avg(rows, key):
            values = [r[key] for r in rows if r.get(key) is not None]
            if not values:
                return '---'
            return f'{np.mean(values):.3f}' + (r'$\pm$' + f'{np.std(values, ddof=1):.3f}' if len(values) > 1 else '')

        ids = lambda rows: [r['canonical_row_id'] for r in rows]

        # 1. Main predictive table
        main = []
        for policy in ('direct_only', 'primary', 'full_deep'):
            rows = choose(test, policy_family=policy)
            main.append([self.cfg['display_names'][policy]] + [avg(rows, k) for k in ('pr_auc', 'f1', 'fraud_recall', 'mcc')])
            for key in ('f1', 'pr_auc', 'fraud_recall', 'mcc', 'deep_rate'):
                self.claim(policy + '_' + key, rows, key)
        self.table(
            'main_predictive',
            'Held-out contract performance across five frozen models under cutoff-safe Level-2 relations.',
            ['Policy', 'PR-AUC', 'F1', 'Fraud recall', 'MCC'],
            main, ids(test),
            'Mean and sample standard deviation across seeds. Evaluated under strict per-target historical eligibility ($t_{\\text{ref}} \\le t_{\\text{target}}$). Statistical Track~B descriptive estimates.'
        )

        # 2. Primary selective frontier table
        frontier = []
        for lane, rows in [('Contract test', test), ('Runtime prefix', runtime), ('Integrated replay', stream)]:
            for policy in ('primary', 'full_deep'):
                group = choose(rows, policy_family=policy)
                routekey = 'deep_rate' if lane == 'Contract test' else 'deep_route_rate'
                frontier.append([lane, self.cfg['display_names'][policy], avg(group, routekey), avg(group, 'mean_latency_ms'), avg(group, 'P99_latency_ms'), avg(group, 'throughput_events_per_second')])
                if lane != 'Contract test':
                    for key in ('deep_route_rate', 'mean_latency_ms', 'P99_latency_ms', 'throughput_events_per_second'):
                        self.claim(lane.split()[0].lower() + '_' + policy + '_' + key, group, key, 'systems_measurement')
        self.table(
            'primary_selective_frontier',
            'Policy-aligned evidence lanes; detection and runtime populations remain separate.',
            ['Population', 'Policy', 'Deep fraction', 'Mean (ms)', 'P99 (ms)', 'Events/s'],
            frontier, ids(test + runtime + stream),
            'A dash marks an unmeasured or inapplicable quantity. Runtime repeats use identical prefixes and warmed models; they do not estimate multi-model accuracy variability.'
        )

        # 3. Statistical evidence table
        srows = []
        for r in stat:
            if r['metric'] == 'f1':
                srows.append([str(r['seed']), f'{r["delta"]:+.3f}', f'[{r["ci_low"]:+.3f}, {r["ci_high"]:+.3f}]', 'Descriptive'])
        self.table(
            'statistical_evidence',
            'Prediction-paired conditional intervals for the selective-minus-local F1 difference.',
            ['Seed', r'$\Delta$F1', r'95\% interval', 'Claim status'],
            srows, ids(stat),
            'Class-stratified matched-contract bootstrap, 2,000 resamples per model under Statistical Track~B.'
        )

        # 4. Calibration metrics table
        crows = []
        for split in ('validation', 'test'):
            for name in ('Raw Level-1 GIN', 'Calibrated Level-1 GIN', 'Final selective score'):
                group = choose(cal, split=split, score_stage=name)
                crows.append([split.title(), name] + [avg(group, k) for k in ('nll', 'brier', 'ece_10', 'adaptive_ece')])
        self.table(
            'calibration_metrics',
            'Quantitative calibration across five seeds.',
            ['Partition', 'Score', 'NLL', 'Brier', 'ECE-10', 'Adaptive ECE'],
            crows, ids(cal),
            'Validation is reused for calibration and policy selection; test partition evaluates held-out calibration.'
        )

        # 5. Cross-chain breakdown table
        cross = []
        cross_source = []
        for scope in ('ethereum', 'bsc', 'polygon'):
            group = choose(pred, split='test', chain_scope=scope, policy_family='primary')
            cross_source.extend(group)
            cross.append([scope.title(), str(group[0]['N_positive'])] + [avg(group, k) for k in ('pr_auc', 'f1', 'mcc')])
        self.table(
            'cross_chain_summary',
            'Per-chain held-out support and primary performance (pooled training model).',
            ['Test scope', 'Fraud support', 'PR-AUC', 'F1', 'MCC'],
            cross, ids(cross_source),
            'Pooled models trained across chains. Polygon test partition contains zero fraud-positive contracts; discrimination metrics are mathematically undefined.'
        )

        # 6. Strict cross-chain transfer table (R4)
        x_raw = pd.read_csv(self.root / 'canonical/cross_chain_temporal_r4.csv')
        x_table_rows = []
        for (srcs, tgt, pol), grp in x_raw.groupby(['sources', 'target_chain', 'policy_family']):
            if pol == 'primary':
                f1_vals = grp['f1'].dropna().values
                pr_vals = grp['pr_auc'].dropna().values
                f1_str = f'{np.mean(f1_vals):.3f}' if len(f1_vals) else 'undefined'
                pr_str = f'{np.mean(pr_vals):.3f}' if len(pr_vals) else 'undefined'
                x_table_rows.append([srcs, tgt.title(), f1_str, pr_str, f'{grp["deep_rate"].mean():.3f}', '0'])
        self.table(
            'cross_chain_summary_r4',
            'Strict source-only cross-chain transfer under cutoff-safe Level-2 reference filtering.',
            ['Source Chains', 'Target Chain', 'Primary F1', 'PR-AUC', 'Deep fraction', 'Temporal Violations'],
            x_table_rows, [str(i) for i in range(len(x_table_rows))],
            'Level-2 relational reasoning queries candidates strictly from source chains where $t_{\\text{ref}} \\le t_{\\text{target}}$. Polygon has zero fraud test contracts.'
        )

        # 7. Integrated streaming table
        self.table(
            'integrated_streaming',
            'Retrospective integrated replay using the frozen primary policy and full relational reference.',
            ['Policy', 'Events', 'Deep fraction', 'Events/s', 'P99 (ms)', 'Peak RSS (MiB)'],
            [[r['display_name'], str(r['events_processed']), f'{r["deep_route_rate"]:.3f}', f'{r["throughput_events_per_second"]:.1f}', f'{r["P99_latency_ms"]:.2f}', f'{r["RSS_peak"] / 1024**2:.1f}'] for r in stream],
            ids(stream),
            'Event labels are inherited from contracts and excluded from this systems table. Workload is a retrospective replay.'
        )

        # 8. Operating point identity table
        identityrows = [[
            r['display_name'], r['evidence_lane'].replace('_', ' '), r['prediction_unit'].replace('_', ' '),
            r['mc_T'], r.get('N', r.get('events_processed')), r.get('deep_rate', r.get('deep_route_rate')), r['canonical_row_id']
        ] for r in choose(test, seed=11) + runtime[:2] + stream]
        pd.DataFrame(identityrows, columns=['Display label', 'Evidence lane', 'Prediction unit', 'T', 'N', 'Deep-route rate', 'canonical_row_id']).to_csv(self.root / 'tables/table_operating_point_identity.csv', index=False)
        self.table(
            'operating_point_identity',
            'Primary operating-point identity by evidence lane (Branch MC-B, T=1).',
            ['Lane', 'Unit', 'T', 'N', 'Deep fraction'],
            [[lane, unit, str(group[0]['mc_T']), str(group[0].get('N', group[0].get('events_processed'))), avg(group, key)] for lane, unit, group, key in [
                ('Offline', 'Contract', choose(test, policy_family='primary'), 'deep_rate'),
                ('Runtime', 'Event trace', choose(runtime, policy_family='primary'), 'deep_route_rate'),
                ('Integrated', 'Event trace', choose(stream, policy_family='primary'), 'deep_route_rate')
            ]],
            ids(test + runtime + stream),
            'All lanes use deterministic T=1 inference under validation-calibrated confidence margins.'
        )

        # 9. Relation temporal audit table (R4)
        audit_summary_rows = []
        for chain, cgrp in audit_records.groupby('target_chain'):
            for split_name, sgrp in cgrp.groupby('split'):
                audit_summary_rows.append([
                    chain.title(), split_name.title(), str(len(sgrp)),
                    f'{int(sgrp["eligible_reference_count"].min()):,}',
                    f'{int(sgrp["eligible_reference_count"].max()):,}',
                    str(int(sgrp["temporal_violation_count"].sum()))
                ])
        self.table(
            'relation_temporal_audit',
            'Historical relational reference audit across evaluated contracts.',
            ['Chain', 'Partition', 'Targets', 'Min Eligible', 'Max Eligible', 'Violations'],
            audit_summary_rows, [str(i) for i in range(len(audit_summary_rows))],
            'Every target contract evaluates against candidate training references satisfying $t_{\\text{reference}} \\le t_{\\text{target}}$. Temporal violations are strictly zero.'
        )

        save_json(self.root / 'revision_manifest.json', {
            'claims': self.claims,
            'table_sources': self.table_sources,
            'statistical_track': 'B',
            'branch': 'MC-B',
            'mc_T': 1,
            'scope': 'held-out descriptive detection and retrospective systems; strict per-target cutoff safety verified',
            'source_hashes': {str(p): file_digest(p) for p in (self.root / 'canonical').glob('*.csv')}
        })
        return self, registries


if __name__ == '__main__':
    b, registries = Builder().build()
    from export.submission_r4_artifacts import finish
    finish(b, registries)
