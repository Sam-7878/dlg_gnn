"""Fail-closed validator for Monte-Carlo method identity and protocol alignment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def validate_mc_method_identity(root: Path, predec_path: Path) -> dict:
    if not predec_path.exists():
        raise FileNotFoundError(f"Missing MC predeclaration: {predec_path}")
    predec = json.loads(predec_path.read_text())
    
    branch = predec.get('branch')
    if branch not in ('MC-A', 'MC-B'):
        raise ValueError(f"Invalid branch '{branch}', expected MC-A or MC-B")
        
    freeze_path = root / 'primary_policy_freeze.json'
    if not freeze_path.exists():
        raise FileNotFoundError(f"Missing primary policy freeze: {freeze_path}")
    freeze = json.loads(freeze_path.read_text())
    
    if branch == 'MC-B':
        # Branch MC-B Invariants
        if freeze.get('mc_T') != 1:
            raise ValueError(f"Branch MC-B requires frozen mc_T == 1, got {freeze.get('mc_T')}")
        if freeze.get('branch') != 'MC-B':
            raise ValueError(f"Freeze branch mismatch: {freeze.get('branch')}")
        if predec['protocol_decisions']['stochastic_dropout_in_production'] is not False:
            raise ValueError("Branch MC-B requires stochastic_dropout_in_production == False")
            
        prohibited = predec['protocol_decisions'].get('prohibited_claims', [])
        
        # Check manuscript files if present
        manuscript_dir = Path('manuscript')
        if manuscript_dir.exists():
            for tex in manuscript_dir.glob('*.tex'):
                text = tex.read_text(encoding='utf-8', errors='ignore')
                for claim in prohibited:
                    if claim.lower() in text.lower():
                        raise ValueError(f"Manuscript {tex.name} contains prohibited claim: '{claim}'")
                        
        summary = {
            'status': 'PASS',
            'branch': 'MC-B',
            'primary_operating_point_T': 1,
            'method_name': predec['protocol_decisions'].get('primary_method_name', 'DLG-SelectiveStream'),
            'production_stochastic_dropout': False,
            'primary_router': 'validation-calibrated confidence margin',
            'mc_status': 'ablation_only'
        }
        return summary
        
    elif branch == 'MC-A':
        # Branch MC-A Invariants
        T = freeze.get('mc_T')
        if T is None or T < 2:
            raise ValueError(f"Branch MC-A requires frozen mc_T >= 2, got {T}")
        if predec['protocol_decisions']['stochastic_dropout_in_production'] is not True:
            raise ValueError("Branch MC-A requires stochastic_dropout_in_production == True")
        summary = {
            'status': 'PASS',
            'branch': 'MC-A',
            'primary_operating_point_T': T,
            'production_stochastic_dropout': True
        }
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='results/sci_v3_submission_r4', help='Output root')
    parser.add_argument('--predec', default='configs/sci_v3_submission_r4/mc_identity_predeclaration.json', help='Predeclaration path')
    args = parser.parse_args()
    summary = validate_mc_method_identity(Path(args.root), Path(args.predec))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
