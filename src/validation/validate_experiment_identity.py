"""Canonical records and claim references must agree field-for-field."""
from dataclasses import fields
import json
from pathlib import Path
import pandas as pd
from evidence.experiment_identity import ExperimentIdentity


def validate(root):
    root=Path(root); registry=pd.read_csv(root/'canonical/experiment_registry.csv',keep_default_na=False)
    errors=[]; columns=[f.name for f in fields(ExperimentIdentity)]
    missing=set(columns+['experiment_id','canonical_row_id'])-set(registry.columns)
    if missing:return [f'missing identity columns: {sorted(missing)}']
    if registry.canonical_row_id.duplicated().any():errors.append('duplicate canonical row identifiers')
    index=registry.set_index('canonical_row_id').to_dict('index')
    for row in registry.to_dict('records'):
        values={k:row[k] for k in columns}
        for key in ('seed','mc_T','warmup_events'):values[key]=int(values[key])
        try:
            expected=ExperimentIdentity(**values).row()
            if row['experiment_id']!=expected['experiment_id']:errors.append(f'identity hash mismatch {row["canonical_row_id"]}')
        except ValueError as exc:errors.append(str(exc))
    manifest=json.loads((root/'revision_manifest.json').read_text())
    for claim in manifest['claims']:
        ids=claim['canonical_row_id'] if isinstance(claim['canonical_row_id'],list) else [claim['canonical_row_id']]
        if any(i not in index for i in ids):errors.append(f'unknown canonical claim row {claim["claim_id"]}');continue
        sources=[index[i] for i in ids]
        for key in ('evidence_lane','prediction_unit'):
            if any(r[key]!=claim[key] for r in sources):errors.append(f'claim {key} mismatch {claim["claim_id"]}')
        if claim['claim_status']=='confirmatory':errors.append('Track B disallows confirmatory claims')
        if claim['claim_status'] not in ('confirmatory','descriptive','systems_measurement','diagnostic_only','undefined','excluded'):errors.append('invalid claim status')
    return errors
