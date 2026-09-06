"""Fail-closed main-paper terminology check, including recursively included tables."""
import re
from pathlib import Path

FORBIDDEN=('ProductionLevel1GIN','ProductionLevel2GATv2','XGBoostFastTriage','LightGBMFastTriage',
    'validation_calibrated_dual','legacy_dual','legacy_risk_controlled','FAIL-C',
    'configs/sci_v3_submission','results/sci_v3_submission','roc_auc_mean','deep_route_rate_std','NaN')


def validate_text(text):
    # Reproducibility links in LaTeX source are not printed prose.
    rendered='\n'.join(line.split('%')[0] for line in text.splitlines())
    rendered=re.sub(r'\\(?:input|includegraphics)(?:\[[^\]]*\])?\{[^}]*\}','',rendered)
    errors=[f'forbidden academic token: {word}' for word in FORBIDDEN if word in rendered]
    if re.search(r'/(?:mnt|home)/|[A-Z]:\\',rendered):errors.append('local filesystem path in manuscript text')
    if re.search(r'\b(?:True|False)\b',rendered):errors.append('raw boolean in manuscript')
    if re.search(r'\\begin\{tabular\}\{[^}]{8,}\}',rendered):errors.append('oversized raw table')
    return errors


def validate(path):
    path=Path(path); text=path.read_text(); errors=validate_text(text)
    for name in re.findall(r'\\input\{([^}]+)\}',text):
        included=path.parent/name
        if not included.suffix:included=included.with_suffix('.tex')
        if not included.exists(): errors.append(f'missing input: {name}')
        else: errors.extend(validate_text(included.read_text()))
    return errors
