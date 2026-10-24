#!/usr/bin/env python3
"""
Inspect all 14 datasets in a08 approved_registry.json for DLG-Base vs DLG-Aug.
"""

from pathlib import Path
import pandas as pd
import json

BASE_DIR = Path(__file__).resolve().parents[1]

a08_reg_json = BASE_DIR / 'evidence' / 'a08_data_repair' / 'approved_registry.json'
with open(a08_reg_json) as f:
    data = json.load(f)

records = data.get('records', [])
df = pd.DataFrame(records)

# Ensure numeric pr_auc
df['pr_auc'] = pd.to_numeric(df['pr_auc'], errors='coerce')
df['roc_auc'] = pd.to_numeric(df['roc_auc'], errors='coerce')
df['f1'] = pd.to_numeric(df.get('f1', df.get('validation_f1', None)), errors='coerce')

print(f"Total rows: {len(df)}")
print("Columns:", df.columns.tolist()[:15])

datasets = [
    'Elliptic', 'DGraphFin', 'BitcoinOTC', 'Ethereum', 'BSC', 'Polygon',
    'Yelp-Syn', 'Amazon-Syn', 'Reddit-Syn', 'Flickr-Syn', 'Cora-Syn',
    'CiteSeer-Syn', 'PubMed-Syn', 'LANL-RedTeam'
]

print("\n=== All 14 Datasets: DLG-Base vs DLG-Aug (A08 Canonical Registry) ===")
rows = []
for ds in datasets:
    sub = df[df['dataset'] == ds]
    base_sub = sub[sub['model'] == 'DLG-Base'].sort_values('seed')
    aug_sub = sub[sub['model'] == 'DLG-Aug'].sort_values('seed')
    
    if len(base_sub) == 5 and len(aug_sub) == 5:
        merged = pd.merge(base_sub, aug_sub, on='seed', suffixes=('_base', '_aug'))
        merged['delta'] = merged['pr_auc_aug'] - merged['pr_auc_base']
        b_mean = merged['pr_auc_base'].mean()
        a_mean = merged['pr_auc_aug'].mean()
        delta_mean = merged['delta'].mean()
        delta_se = merged['delta'].std(ddof=1) / (5 ** 0.5)
        
        rows.append({
            'dataset': ds,
            'base_mean': b_mean,
            'aug_mean': a_mean,
            'delta_mean': delta_mean,
            'delta_se': delta_se,
            'seeds': sorted(merged['seed'].tolist())
        })
        print(f"{ds:15s} | Base: {b_mean:.4f} | Aug: {a_mean:.4f} | Delta: {delta_mean:+.4f} ± {delta_se:.4f} | seeds: {sorted(merged['seed'].tolist())}")
    else:
        print(f"{ds:15s} | Base count: {len(base_sub)}, Aug count: {len(aug_sub)}")

res_df = pd.DataFrame(rows)
print("\nSummary of all 14 Deltas:")
print(res_df[['dataset', 'base_mean', 'aug_mean', 'delta_mean', 'delta_se']])
