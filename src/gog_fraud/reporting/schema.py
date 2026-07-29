from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


REPORT_TOP_LEVEL_FIELDS = (
    "report_metadata", "executive_summary", "implementation_status",
    "dataset_audit", "leakage_audit", "experiment_registry",
    "baseline_fairness", "main_results", "routing_analysis", "mc_analysis",
    "calibration", "streaming", "resources", "latency", "ablations",
    "temporal", "cross_chain", "statistics", "failures",
    "claim_verification", "consistency_issues", "reproducibility",
    "submission_readiness", "evidence_index",
)


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    category: str
    experiment_id: str
    path: str
    sha256: str
    generated_at: str
    producer: str
    config_hash: str
    git_sha: str
    status: str
    used_in_report: bool
    report_sections: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationIssue:
    issue_id: str
    severity: str
    category: str
    message: str
    evidence: tuple[str, ...] = field(default_factory=tuple)
    resolution: str = "UNRESOLVED"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = list(self.evidence)
        return data
