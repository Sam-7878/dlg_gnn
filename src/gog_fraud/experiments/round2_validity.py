"""Round-2 architecture naming, graph invariance, and readiness gates."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class VariantIdentity:
    paper_name: str
    historical_name: str
    variant_name: str
    python_module: str
    python_class: str
    score_definition: str

    def to_dict(self) -> dict[str, str]: return asdict(self)


VARIANT_IDENTITIES = {
    "global_only": VariantIdentity("DLG-Base", "DLG-Base", "global_only", "gog_fraud.models.pygod.dlg", "DLG", "same-graph local/global fused reconstruction error"),
    "local_only": VariantIdentity("DLG-Local", "DLG local score (Round 1 export)", "local_only", "gog_fraud.models.pygod.dlg_full", "DLGFull._pretrain_level1", "local feature reconstruction MSE"),
    "local_augmented_global": VariantIdentity("DLG-Aug", "DLG", "local_augmented_global", "gog_fraud.models.pygod.dlg_full", "DLGFull", "global original-feature/adjacency reconstruction error from concat(X,H_local)"),
    "local_global_fusion": VariantIdentity("DLG-Fusion", "Round 1 full", "local_global_fusion", "gog_fraud.pipelines.run_sci_round1_ablation", "validation-selected weighted fusion", "validation-scaled weighted local and augmented-global score"),
}


def variant_identity(name: str) -> VariantIdentity:
    try: return VARIANT_IDENTITIES[name]
    except KeyError as exc: raise ValueError(f"unknown DLG variant: {name}") from exc


def _array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"): value = value.detach().cpu().numpy()
    return np.ascontiguousarray(value)


def tensor_hash(value: Any) -> str:
    array = _array(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode()); digest.update(str(array.shape).encode()); digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def graph_fingerprints(data: Any, *, injection_config: dict[str, Any] | None = None) -> dict[str, str]:
    config = repr(sorted((injection_config or {}).items())).encode()
    return {
        "dataset_hash": hashlib.sha256((tensor_hash(data.x) + tensor_hash(data.edge_index) + tensor_hash(data.y)).encode()).hexdigest(),
        "feature_hash": tensor_hash(data.x), "edge_hash": tensor_hash(data.edge_index),
        "label_hash": tensor_hash(data.y), "injection_hash": hashlib.sha256(config + tensor_hash(data.y).encode()).hexdigest(),
    }


def validation_support(y_validation: Any, y_test: Any, *, warning_threshold: int = 20) -> dict[str, Any]:
    validation = _array(y_validation).reshape(-1); test = _array(y_test).reshape(-1)
    validation_positive, test_positive = int((validation == 1).sum()), int((test == 1).sum())
    return {
        "validation_positive": validation_positive, "validation_negative": int((validation == 0).sum()),
        "test_positive": test_positive, "test_negative": int((test == 0).sum()),
        "threshold_unstable_warning": validation_positive < warning_threshold,
        "metric_low_support_warning": test_positive < warning_threshold,
        "support_warning_threshold": warning_threshold,
    }


def decide_readiness(gates: dict[str, bool]) -> str:
    """Conservative three-state gate; critical false gates prevent readiness."""
    critical = ("architecture_identity", "score_semantics", "partition_fidelity", "pilot_execution")
    if any(not gates.get(name, False) for name in critical): return "NOT_READY"
    return "READY_FOR_FULL_RUN" if all(gates.values()) else "READY_WITH_FIXES"
