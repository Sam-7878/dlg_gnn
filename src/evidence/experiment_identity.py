"""Content-addressed identities prevent cross-population result substitution."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ExperimentIdentity:
    evidence_lane: str
    prediction_unit: str
    evaluation_population_id: str
    dataset_version: str
    split_id: str
    chain_scope: str
    seed: int
    model_id: str
    policy_family: str
    policy_config_id: str
    calibration_id: str
    threshold_id: str
    fusion_id: str
    mc_T: int
    prefix_id: str
    replay_id: str
    warmup_events: int
    measurement_type: str
    git_sha: str
    config_sha256: str

    def row(self) -> dict:
        value = asdict(self)
        allowed = {
            'offline_contract_detection': {'contract'},
            'raw_event_runtime': {'raw_event_runtime_trace'},
            'integrated_streaming': {'event_with_inherited_contract_label'},
        }
        if self.prediction_unit not in allowed.get(self.evidence_lane, set()):
            raise ValueError('evidence lane/prediction unit mismatch')
        if self.mc_T < 1 or not self.evaluation_population_id or not self.policy_config_id:
            raise ValueError('incomplete experiment identity')
        value['experiment_id'] = digest(value)
        value['canonical_row_id'] = value['experiment_id'][:20]
        return value


def require_compatible(rows: list[dict], *, group_columns: tuple[str, ...] = ()) -> None:
    for field in ('evidence_lane', 'prediction_unit', 'evaluation_population_id', 'policy_config_id', 'mc_T'):
        if len({row[field] for row in rows}) > 1 and field not in group_columns:
            raise ValueError(f'incompatible {field}: explicit grouping required')
    for row in rows:
        if 'legacy' in row['policy_family'] and 'diagnostic' not in row.get('display_name', '').lower():
            raise ValueError('legacy policy requires diagnostic display label')
