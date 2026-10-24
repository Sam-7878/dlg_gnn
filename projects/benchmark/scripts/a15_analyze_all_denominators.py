import pandas as pd
import json
import os
import glob
import numpy as np

reg_path = '/mnt/d/_Work/goat_bank/dlg_gnn/projects/benchmark/publication/a14/canonical_selected_run_registry.csv'
df = pd.read_csv(reg_path)

# Map jsons
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

seed_rows = []

for ds in sorted(df['dataset'].unique()):
    sub_ds = df[df['dataset'] == ds]
    for seed in [42, 43, 44, 45, 46]:
        sub_seed = sub_ds[sub_ds['seed'] == seed]
        
        # We need test_total, test_positive, test_prevalence for this (dataset, seed)
        # All models evaluated on this (dataset, seed) share the test split
        # Let's check across models for this seed
        t_pos, t_neg, t_tot, t_prev = None, None, None, None
        val_pos, val_neg, val_tot = None, None, None
        nodes, edges = None, None
        split_strat = None
        source_p = None
        
        for _, row in sub_seed.iterrows():
            model = row['model']
            camp = row['campaign_id']
            mf = row.get('metric_file_path')
            bn = os.path.basename(str(mf)) if pd.notna(mf) else ''
            
            cand_path = None
            if bn in json_map:
                cand_path = json_map[bn]
            else:
                ds_clean = ds.replace('-Syn', '').replace('-RedTeam', '')
                pat1 = f'{ds}__{model}__seed{seed}'
                pat2 = f'{ds_clean}__{model}__seed{seed}'
                for k, v in json_map.items():
                    if (pat1 in k or pat2 in k) and (camp in v or 'clean_v1' in v or 'round5' in v or 'bitcoinotc' in v or 'defense' in v or 'anomalydae' in v):
                        cand_path = v
                        break
            
            if cand_path and os.path.exists(cand_path):
                try:
                    with open(cand_path) as jf:
                        jd = json.load(jf)
                        tp = jd.get('test_positive') or jd.get('test_positives') or jd.get('n_test_positives')
                        tn = jd.get('test_negative') or jd.get('test_negatives')
                        tt = jd.get('test_total') or jd.get('n_test_samples')
                        if tt is None and tp is not None and tn is not None:
                            tt = tp + tn
                        pr = jd.get('test_prevalence')
                        if pr is None and tt and tt > 0 and tp is not None:
                            pr = tp / tt
                        
                        vp = jd.get('validation_positive') or jd.get('validation_positives')
                        vn = jd.get('validation_negative') or jd.get('validation_negatives')
                        vt = jd.get('validation_total')
                        if vt is None and vp is not None and vn is not None:
                            vt = vp + vn
                        
                        if tp is not None:
                            t_pos = tp
                            t_neg = tn
                            t_tot = tt
                            t_prev = pr
                            val_pos = vp
                            val_neg = vn
                            val_tot = vt
                            nodes = jd.get('nodes')
                            edges = jd.get('edges')
                            split_strat = jd.get('split_strategy')
                            source_p = cand_path
                            break
                except Exception:
                    pass
        
        # Fallback to registry if not found in json
        if t_prev is None and pd.notna(sub_seed.iloc[0].get('test_prevalence')):
            t_prev = float(sub_seed.iloc[0]['test_prevalence'])
        if t_tot is None and pd.notna(sub_seed.iloc[0].get('test_total')):
            t_tot = float(sub_seed.iloc[0]['test_total'])
        if t_pos is None and pd.notna(sub_seed.iloc[0].get('test_positive')):
            t_pos = float(sub_seed.iloc[0]['test_positive'])
        if nodes is None and pd.notna(sub_seed.iloc[0].get('nodes')):
            nodes = int(sub_seed.iloc[0]['nodes'])
        if edges is None and pd.notna(sub_seed.iloc[0].get('edges')):
            edges = int(sub_seed.iloc[0]['edges'])

        seed_rows.append({
            'dataset': ds,
            'seed': seed,
            'campaign_id': sub_seed.iloc[0]['campaign_id'],
            'nodes': nodes,
            'edges': edges,
            'val_pos': val_pos,
            'val_neg': val_neg,
            'val_total': val_tot,
            'test_pos': t_pos,
            'test_neg': t_neg,
            'test_total': t_tot,
            'test_prevalence': t_prev,
            'split_strategy': split_strat,
            'source_path': source_p
        })

df_seeds = pd.DataFrame(seed_rows)

print('=== 14 DATASETS SEED-LEVEL SUMMARY ===')
summary_list = []
for ds, grp in df_seeds.groupby('dataset'):
    mean_prev = grp['test_prevalence'].mean()
    sd_prev = grp['test_prevalence'].std()
    min_prev = grp['test_prevalence'].min()
    max_prev = grp['test_prevalence'].max()
    t_tot_mean = grp['test_total'].mean()
    t_pos_mean = grp['test_pos'].mean()
    nodes = grp['nodes'].iloc[0]
    edges = grp['edges'].iloc[0]
    strat = grp['split_strategy'].dropna().iloc[0] if grp['split_strategy'].notna().sum() > 0 else 'fixed'
    
    summary_list.append({
        'dataset': ds,
        'nodes': nodes,
        'edges': edges,
        'test_total': t_tot_mean,
        'test_pos': t_pos_mean,
        'mean_test_prev': mean_prev,
        'sd_test_prev': sd_prev,
        'min_test_prev': min_prev,
        'max_test_prev': max_prev,
        'split_strategy': strat
    })

sum_df = pd.DataFrame(summary_list)
print(sum_df.to_string(index=False))
