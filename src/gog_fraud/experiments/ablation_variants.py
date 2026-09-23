"""Canonical, non-simulated DLG component definitions for SCI round 1."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AblationVariant:
    name: str
    use_local: bool
    use_global: bool
    use_fusion: bool
    score_source: str
    implementation_note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ABLATION_VARIANTS = {
    "l1_only": AblationVariant(
        "l1_only", True, False, False, "local_reconstruction",
        "Separately trained local GCN reconstruction score only.",
    ),
    "global_only": AblationVariant(
        "global_only", False, True, False, "global_reconstruction",
        "Historical paper DLG-Base: gog_fraud.models.pygod.dlg.DLG.",
    ),
    "l1_l2": AblationVariant(
        "l1_l2", True, True, False, "global_reconstruction",
        "Frozen local embeddings augment global input; global score only.",
    ),
    "full": AblationVariant(
        "full", True, True, True, "weighted_local_global",
        "Local and global scores are fused using a validation-fixed policy.",
    ),
    "dlg_base": AblationVariant(
        "dlg_base", False, True, False, "global_reconstruction",
        "Compatibility alias for the historical DLG-Base detector.",
    ),
}


def get_ablation_variant(name: str) -> AblationVariant:
    try:
        return ABLATION_VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown ablation variant: {name}") from exc

