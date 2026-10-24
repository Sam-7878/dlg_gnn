#!/usr/bin/env python3
"""A11 Freeze & Integrity Inventory Generation."""
import csv
import json
from pathlib import Path

ROOT = Path('/mnt/d/_Work/goat_bank/dlg_gnn')
BENCH = ROOT / 'projects/benchmark'
PUB = BENCH / 'publication/a11'
E = BENCH / 'evidence/a08_data_repair'

# 1. evidence_inventory.csv
num_reg = json.loads((E / 'number_registry.json').read_text(encoding='utf-8'))
cells = {(c['dataset'], c['model'], c['metric']): c for c in num_reg['cells']}
support = list(csv.DictReader((E / 'tables/support_matrix.csv').open(encoding='utf-8')))

inventory = []
for s in support:
    d, m = s['dataset'], s['model']
    pr = cells.get((d, m, 'pr_auc'), {})
    roc = cells.get((d, m, 'roc_auc'), {})
    f1 = cells.get((d, m, 'validation_f1'), {})
    
    tier = 'external_evaluation' if d == 'LANL-RedTeam' else (
        'crypto_a08_clean' if d in ['Ethereum', 'BSC', 'Polygon'] else (
            'canonical_bitcoin_otc' if d == 'BitcoinOTC' else 'historical_round5'
        )
    )
    
    inventory.append({
        'dataset': d,
        'model': m,
        'tier': tier,
        'support_status': s['support_status'],
        'successes': s['successes'],
        'seeds': s['seeds'],
        'pr_auc_mean': f"{pr.get('mean', 0.0):.4f}" if pr else 'N/A',
        'pr_auc_sd': f"{pr.get('sd', 0.0):.4f}" if pr else 'N/A',
        'roc_auc_mean': f"{roc.get('mean', 0.0):.4f}" if roc else 'N/A',
        'roc_auc_sd': f"{roc.get('sd', 0.0):.4f}" if roc else 'N/A',
        'validation_f1_mean': f"{f1.get('mean', 0.0):.4f}" if f1 else 'N/A',
        'validation_f1_sd': f"{f1.get('sd', 0.0):.4f}" if f1 else 'N/A',
        'run_ids': ';'.join(pr.get('run_ids', []))
    })

with (PUB / 'evidence_inventory.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(inventory[0].keys()))
    writer.writeheader()
    writer.writerows(inventory)

print(f'Created evidence_inventory.csv with {len(inventory)} rows')

# 2. claims_to_evidence.csv
claims = [
    {
        'claim_id': 'CLM-01-SUPPORT',
        'claim_text': '84 of 91 primary model-dataset pairs are fully supported across 5 seeds (420 primary successes).',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/number_registry.json',
        'metric_or_value': 'primary_supported_pairs=84, primary_successes=420',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-02-LANL',
        'claim_text': 'LANL-RedTeam external cybersecurity dataset contains 35 evaluated runs (7 models x 5 seeds).',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/number_registry.json',
        'metric_or_value': 'external_successes=35',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-03-FRIEDMAN-S1',
        'claim_text': 'Friedman test S1 (8 datasets, 7 models): Q=29.0357, Asymptotic p=5.99e-5, Permutation p=5.0e-6.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/statistics/statistics_s1_s4.json',
        'metric_or_value': 'Q=29.0357, asym_p=5.99e-5, perm_p=5.0e-6',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-04-FRIEDMAN-S2',
        'claim_text': 'Friedman test S2 (13 datasets, 5 models): Q=23.7538, Asymptotic p=8.95e-5, Permutation p=2.0e-5.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/statistics/statistics_s1_s4.json',
        'metric_or_value': 'Q=23.7538, asym_p=8.95e-5, perm_p=2.0e-5',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-05-FRIEDMAN-S3',
        'claim_text': 'Friedman test S3 (6 datasets, 5 models): Q=6.1333, Asymptotic p=0.1894, Permutation p=0.193584.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/statistics/statistics_s1_s4.json',
        'metric_or_value': 'Q=6.1333, asym_p=0.1894, perm_p=0.193584',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-06-FRIEDMAN-S4',
        'claim_text': 'Friedman test S4 (7 datasets, 5 models): Q=21.8286, Asymptotic p=2.17e-4, Permutation p=5.0e-6.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/statistics/statistics_s1_s4.json',
        'metric_or_value': 'Q=21.8286, asym_p=2.17e-4, perm_p=5.0e-6',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-07-ELLIPTIC-DELTA',
        'claim_text': 'Elliptic conditional AP delta: DLG-Aug (0.1037) vs DLG-Base (0.0687), delta = +0.0350.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/tables/table_pr_auc.csv',
        'metric_or_value': 'Aug=0.1037, Base=0.0687, Delta=+0.0350',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-08-REDDIT-DELTA',
        'claim_text': 'Reddit-Syn conditional AP delta: DLG-Aug (0.0573) vs DLG-Base (0.1229), delta = -0.0656.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/tables/table_pr_auc.csv',
        'metric_or_value': 'Aug=0.0573, Base=0.1229, Delta=-0.0656',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-09-BLOCKCHAIN-CLAIM',
        'claim_text': 'At least one established comparator exceeds DLG-Aug in mean AP on each of the three blockchain datasets (Ethereum: DOMINANT, BSC: DOMINANT, Polygon: OCGNN).',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/tables/table_pr_auc.csv',
        'metric_or_value': 'Ethereum DOM=0.7410 vs Aug=0.6976; BSC DOM=0.6657 vs Aug=0.6276; Polygon OCGNN=0.4851 vs Aug=0.4578',
        'audit_status': 'VERIFIED_EXACT'
    },
    {
        'claim_id': 'CLM-10-LOW-ROC-AUC',
        'claim_text': 'Elliptic ROC-AUC values for reconstruction models are below random: DOMINANT 0.3432, AnomalyDAE 0.1296, OCGNN 0.3140, DLG-Base 0.3500; DLG-Aug is 0.5705.',
        'evidence_file': 'projects/benchmark/evidence/a08_data_repair/tables/table_roc_auc.csv',
        'metric_or_value': 'DOM=0.3432, Anom=0.1296, OCGNN=0.3140, Base=0.3500, Aug=0.5705',
        'audit_status': 'AUDITED_VALID_PHENOMENON'
    }
]

with (PUB / 'claims_to_evidence.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(claims[0].keys()))
    writer.writeheader()
    writer.writerows(claims)

print(f'Created claims_to_evidence.csv with {len(claims)} claims')
