#!/usr/bin/env python3
"""
Inspect a08 approved_registry.json and compare with astra_revision approved_run_registry.csv.
"""

from pathlib import Path
import pandas as pd
import json

BASE_DIR = Path(__file__).resolve().parents[1]

a08_reg_json = BASE_DIR / 'evidence' / 'a08_data_repair' / 'approved_registry.json'
with open(a08_reg_json) as f:
    data = json.load(f)

print(f"campaign_id: {data.get('campaign_id')}")
records = data.get('records', [])
print(f"Total records in a08 approved_registry.json: {len(records)}")

df_a08 = pd.DataFrame(records)
print("\nDataset counts in a08 approved_registry.json:")
print(df_a08['dataset'].value_counts())

for ds in ['Ethereum', 'BSC', 'Polygon', 'Elliptic', 'Reddit-Syn']:
    sub = df_a08[(df_a08['dataset'] == ds) & (df_a08['model'].isin(['DLG-Base', 'DLG-Aug']))]
    if len(sub) > 0:
        print(f"\n--- {ds} in a08 approved_registry.json ---")
        for idx, row in sub.iterrows():
            print(f"  {row['model']} seed={row['seed']} pr_auc={row.get('pr_auc')}")
        b = sub[sub['model'] == 'DLG-Base']['pr_auc'].mean()
        a = sub[sub['model'] == 'DLG-Aug']['pr_auc'].mean()
        print(f"  Mean Base: {b:.6f}, Aug: {a:.6f}, Delta: {a - b:.6f}")
