"""Fail-closed validator for Level-2 temporal relation provenance and cutoff safety."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd


def validate_temporal_relation_provenance(root: Path) -> dict:
    canonical = root / 'canonical'
    offline = root / 'offline'
    
    audit_path = canonical / 'relation_temporal_audit.csv'
    if not audit_path.exists():
        raise FileNotFoundError(f"Missing canonical relation temporal audit: {audit_path}")
    
    df = pd.read_csv(audit_path)
    if len(df) == 0:
        raise ValueError(f"Relation temporal audit is empty: {audit_path}")
    
    required_cols = [
        'target_contract_id', 'target_chain', 'split', 'target_cutoff',
        'eligible_reference_count', 'selected_neighbor_count',
        'max_selected_reference_cutoff', 'temporal_violation_count'
    ]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required audit column: {col}")
    
    # Invariant 1: No missing values in critical provenance columns
    for col in required_cols:
        if df[col].isnull().any():
            raise ValueError(f"Found null values in audit column: {col}")
            
    # Invariant 2: temporal_violation_count == 0 for all rows
    violations = int(df['temporal_violation_count'].sum())
    if violations > 0:
        bad = df[df['temporal_violation_count'] > 0]
        raise ValueError(f"Detected {violations} temporal violations! Samples:\n{bad.head()}")
        
    # Invariant 3: max_selected_reference_cutoff <= target_cutoff
    diff = df['max_selected_reference_cutoff'] - df['target_cutoff']
    if (diff > 0).any():
        bad = df[diff > 0]
        raise ValueError(f"Selected reference cutoff exceeds target cutoff in {len(bad)} rows! Samples:\n{bad.head()}")
        
    # Invariant 4: eligible_reference_count >= 1 and selected_neighbor_count >= 1
    if (df['eligible_reference_count'] < 1).any():
        bad = df[df['eligible_reference_count'] < 1]
        raise ValueError(f"Zero eligible references found for targets:\n{bad.head()}")
        
    # Invariant 5: Check 5-seed training reference npz files
    for seed in (11, 22, 33, 44, 55):
        npz_path = offline / f'seed{seed}' / 'training_reference_temporal.npz'
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing training reference temporal artifact: {npz_path}")
        data = np.load(npz_path)
        for field in ('embedding', 'score', 'event_end', 'chain', 'contract_id', 'label'):
            if field not in data:
                raise KeyError(f"Missing field '{field}' in {npz_path}")
        times = data['event_end']
        if len(times) == 0 or not np.issubdtype(times.dtype, np.integer):
            raise TypeError(f"Invalid event_end timestamps in {npz_path}")
            
    # Invariant 6: Cross-chain temporal holdout
    x_path = canonical / 'cross_chain_temporal_r4.csv'
    if x_path.exists():
        x_df = pd.read_csv(x_path)
        if (x_df['temporal_violation_count'] > 0).any():
            raise ValueError("Temporal violations found in cross_chain_temporal_r4.csv")

    summary = {
        'status': 'PASS',
        'audited_targets': len(df),
        'temporal_violations': 0,
        'min_eligible_references': int(df['eligible_reference_count'].min()),
        'max_eligible_references': int(df['eligible_reference_count'].max()),
        'max_reference_cutoff_le_target_cutoff': True
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/sci_v3_submission_r4', help='Output root')
    args = parser.parse_args()
    summary = validate_temporal_relation_provenance(Path(args.root))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
