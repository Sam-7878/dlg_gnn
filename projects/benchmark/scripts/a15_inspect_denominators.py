import pandas as pd
import json
import os

reg_path = '/mnt/d/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a14/canonical_selected_run_registry.csv'
df = pd.read_csv(reg_path)

print(f'Total rows in canonical registry: {len(df)}')

# Build map of all JSON files in the workspace
json_map = {}
search_dirs = [
    '/mnt/d/_Work/goat_bank/dlg_gnn/archive/benchmark/outputs',
    '/mnt/d/_Work/goat_bank/dlg_gnn/projects/benchmark/evidence',
    '/mnt/d/_Work/goat_bank/dlg_gnn/evaluation/benchmark'
]
for sdir in search_dirs:
    for root, _, files in os.walk(sdir):
        for f in files:
            if f.endswith('.json'):
                json_map[f] = os.path.join(root, f)

print(f'Indexed {len(json_map)} json files.')

results = []
for idx, row in df.iterrows():
    ds = row['dataset']
    model = row['model']
    seed = int(row['seed'])
    camp = row['campaign_id']
    
    # Try finding json
    metric_f = row.get('metric_file_path')
    base_name = os.path.basename(str(metric_f)) if pd.notna(metric_f) else ''
    
    found_path = None
    if base_name in json_map:
        found_path = json_map[base_name]
    else:
        # Check standard pattern
        # Note: some files use 'Amazon' instead of 'Amazon-Syn'
        ds_clean = ds.replace('-Syn', '').replace('-RedTeam', '')
        pat1 = f'{ds}__{model}__seed{seed}'
        pat2 = f'{ds_clean}__{model}__seed{seed}'
        for k, v in json_map.items():
            if (pat1 in k or pat2 in k) and (camp in v or 'clean_v1' in v or 'round5' in v or 'bitcoinotc' in v or 'defense' in v or 'anomalydae' in v):
                found_path = v
                break
        if not found_path:
            # looser match
            for k, v in json_map.items():
                if pat1 in k or pat2 in k:
                    found_path = v
                    break
                
    # Read json data if found
    test_pos = None
    test_neg = None
    test_total = None
    test_prev = None
    val_pos = None
    val_neg = None
    split_strat = None
    nodes = None
    edges = None
    
    if found_path and os.path.exists(found_path):
        try:
            with open(found_path) as jf:
                jd = json.load(jf)
                test_pos = jd.get('test_positive') or jd.get('test_positives')
                test_neg = jd.get('test_negative') or jd.get('test_negatives')
                test_total = jd.get('test_total')
                if test_total is None and test_pos is not None and test_neg is not None:
                    test_total = test_pos + test_neg
                test_prev = jd.get('test_prevalence')
                if test_prev is None and test_total and test_total > 0 and test_pos is not None:
                    test_prev = test_pos / test_total
                val_pos = jd.get('validation_positive') or jd.get('validation_positives')
                val_neg = jd.get('validation_negative') or jd.get('validation_negatives')
                split_strat = jd.get('split_strategy')
                nodes = jd.get('nodes')
                edges = jd.get('edges')
        except Exception as e:
            pass

    # Fallback to row registry values if json didn't have them
    if test_prev is None and pd.notna(row.get('test_prevalence')):
        test_prev = float(row['test_prevalence'])
    if test_total is None and pd.notna(row.get('test_total')):
        test_total = float(row['test_total'])
    if test_pos is None and pd.notna(row.get('test_positive')):
        test_pos = float(row['test_positive'])
    if nodes is None and pd.notna(row.get('nodes')):
        nodes = int(row['nodes'])
    if edges is None and pd.notna(row.get('edges')):
        edges = int(row['edges'])

    results.append({
        'dataset': ds,
        'model': model,
        'seed': seed,
        'campaign_id': camp,
        'nodes': nodes,
        'edges': edges,
        'val_pos': val_pos,
        'val_neg': val_neg,
        'test_pos': test_pos,
        'test_neg': test_neg,
        'test_total': test_total,
        'test_prev': test_prev,
        'split_strategy': split_strat,
        'found_path': found_path
    })

res_df = pd.DataFrame(results)

print('\nSummary by dataset:')
for ds, grp in res_df.groupby('dataset'):
    found_cnt = grp['found_path'].notna().sum()
    has_pos = grp['test_pos'].notna().sum()
    sample_prev = grp['test_prev'].dropna().unique()
    sample_tot = grp['test_total'].dropna().unique()
    sample_pos = grp['test_pos'].dropna().unique()
    sample_p = grp['found_path'].dropna().iloc[0] if found_cnt > 0 else 'None'
    print(f'{ds:15s}: jsons={found_cnt:2d}/{len(grp):2d}, test_prev={sample_prev[:3]}, test_total={sample_tot[:3]}, test_pos={sample_pos[:3]}')
    print(f'                 sample path: {sample_p}')
