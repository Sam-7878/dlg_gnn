"""Round 4A reconstruction/backend and receptive-field evidence contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ReceptiveFieldRecord:
    model: str
    message_passing_layers: int | str
    effective_receptive_field_hops: int | str
    partition_halo_hops: int
    representation_equivalence_expected: bool
    reason: str

    def to_dict(self):
        return asdict(self)


RECEPTIVE_FIELD_MANIFEST = (
    ReceptiveFieldRecord("DOMINANT", 4, 4, 0, True, "full sparse shared graph; encoder/decoder maximum path"),
    ReceptiveFieldRecord("CONAD", 4, 4, 0, True, "full sparse shared graph plus sparse augmentation"),
    ReceptiveFieldRecord("DLG-Base", 6, 6, 0, True, "2 local + 2 global + 2 attribute-decoder layers"),
    ReceptiveFieldRecord("DLG-Aug", 6, 6, 0, True, "2 frozen local + 2 global + 2 attribute-decoder layers"),
    ReceptiveFieldRecord("AnomalyDAE", 1, "global_dense_attribute_axis", 0, True, "one GAT layer plus globally coupled attribute decoder"),
    ReceptiveFieldRecord("CoLA", "model_defined", "model_defined", 0, True, "full graph PyGOD execution; no SCI partition"),
    ReceptiveFieldRecord("GADNR", "model_defined", "model_defined", 0, True, "full graph PyGOD execution; no SCI partition"),
    ReceptiveFieldRecord("OCGNN", "model_defined", "model_defined", 0, True, "full graph PyGOD execution; no SCI partition"),
)


def receptive_field_manifest() -> list[dict[str, object]]:
    return [record.to_dict() for record in RECEPTIVE_FIELD_MANIFEST]


CRITICAL_GATES = (
    "exact_sparse_mathematical_equivalence",
    "shared_full_graph_semantic_equivalence",
    "dgraphfin_resource",
    "yelp_resource",
    "reddit_resource",
    "yelp_six_run",
    "reddit_conad",
    "dlg_component_repilot",
    "representative_80_run",
    "score_semantics",
    "provenance",
)


def decide_round4a_readiness(gates: dict[str, bool], *, restrictions: list[str] | None = None) -> str:
    """Fail closed; restrictions never excuse a false scientific gate."""
    if any(not gates.get(name, False) for name in CRITICAL_GATES):
        return "NOT_READY"
    return "READY_WITH_RESTRICTIONS" if restrictions else "READY_FOR_FULL_RUN"
