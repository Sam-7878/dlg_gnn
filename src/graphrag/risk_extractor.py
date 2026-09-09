"""
graphrag/risk_extractor.py

Extracts the 5-component risk vector r_t = [s_t, q_t, k_t, a_t, h_t]
from GraphRAG evidence items.

Components:
    s_t  semantic risk score        : weighted average of evidence scores
    q_t  extraction confidence      : fraction of top-k evidence with score > threshold
    k_t  risk category (int)        : most prevalent ScamType node mapped to category ID
    a_t  context age (sec)          : pre_transaction_gap_sec from context metadata
    h_t  relation hint (int)        : most prevalent Cue node mapped to cue ID

NOTE: This module does NOT access fraud labels. All values are derived from
GraphRAG evidence alone.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from graphrag.retriever import EvidenceItem

logger = logging.getLogger(__name__)


# ── Category mappings (must stay in sync with risk_vectorizer.py) ────────────

SCAM_TYPE_TO_CATEGORY_ID: Dict[str, int] = {
    "ST_investment":    1,
    "ST_romance":       2,
    "ST_phishing":      3,
    "ST_impersonation": 4,
    "ST_urgent":        5,
    "ST_fake_support":  6,
    "ST_migration":     7,
    "ST_recovery":      8,
    "ST_high_yield":    9,
    "ST_grooming":      10,
}

CUE_TO_HINT_ID: Dict[str, int] = {
    "CUE_guaranteed_return": 1,
    "CUE_urgent_transfer":   2,
    "CUE_wallet_request":    3,
    "CUE_identity_verify":   4,
    "CUE_escrow":            5,
    "CUE_seed_phrase":       5,   # map to "external escrow contract" bucket
    "CUE_url_link":          3,   # map to "external wallet request" bucket
    "CUE_authority":         4,   # map to "identity verification" bucket
}


class RiskExtractor:
    """
    Converts a list of GraphRAG EvidenceItems into a raw risk context dict
    that can be passed to RiskVectorizer.

    Usage:
        extractor = RiskExtractor()
        context_dict = extractor.extract(evidence_items, pre_transaction_gap_sec=300)
    """

    def __init__(self, confidence_threshold: float = 0.15):
        """
        Args:
            confidence_threshold: Minimum evidence score to count toward confidence q_t.
        """
        self.confidence_threshold = confidence_threshold

    def extract(
        self,
        evidence: List[EvidenceItem],
        *,
        event_id: Optional[str] = None,
        pre_transaction_gap_sec: int = 0,
    ) -> dict:
        """
        Extract r_t = [s_t, q_t, k_t, a_t, h_t] from evidence items.

        Args:
            evidence: Top-k evidence items from GraphRAGRetriever.
            event_id: Transaction/context identifier.
            pre_transaction_gap_sec: Time gap between context and transaction.

        Returns:
            context_dict suitable for RiskVectorizer.vectorize().
        """
        if not evidence:
            return {
                "event_id": event_id,
                "local_risk_score": 0.0,
                "confidence": 0.0,
                "scenario_type": "benign",
                "risk_cues": [],
                "pre_transaction_gap_sec": pre_transaction_gap_sec,
            }

        # ── s_t: semantic risk score ─────────────────────────────────────────
        # Weighted average of evidence scores (top-k items).
        weights = [e.score for e in evidence]
        total_w = sum(weights)
        if total_w > 0:
            s_t = sum(e.score * e.score for e in evidence) / total_w
        else:
            s_t = 0.0
        s_t = float(min(s_t, 1.0))

        # ── q_t: extraction confidence ───────────────────────────────────────
        # Fraction of evidence items that exceed the confidence threshold.
        high_conf = sum(1 for e in evidence if e.score >= self.confidence_threshold)
        q_t = float(high_conf) / max(len(evidence), 1)

        # ── k_t: dominant scam category ──────────────────────────────────────
        scam_votes: Dict[str, float] = {}
        for ev in evidence:
            if ev.node_type == "ScamType":
                scam_votes[ev.node_id] = scam_votes.get(ev.node_id, 0.0) + ev.score
        if scam_votes:
            dominant_scam = max(scam_votes, key=scam_votes.get)
            k_t_id = SCAM_TYPE_TO_CATEGORY_ID.get(dominant_scam, 0)
            scenario_type = self._scam_node_to_scenario(dominant_scam)
        else:
            k_t_id = 0
            scenario_type = "benign"

        # ── h_t: dominant cue / relation hint ───────────────────────────────
        cue_votes: Dict[str, float] = {}
        for ev in evidence:
            if ev.node_type == "Cue":
                cue_votes[ev.node_id] = cue_votes.get(ev.node_id, 0.0) + ev.score
        if cue_votes:
            dominant_cue = max(cue_votes, key=cue_votes.get)
            h_t_id = CUE_TO_HINT_ID.get(dominant_cue, 0)
            risk_cues = [self._cue_node_to_label(dominant_cue)]
        else:
            h_t_id = 0
            risk_cues = []

        return {
            "event_id": event_id,
            "local_risk_score": s_t,           # s_t
            "confidence": q_t,                  # q_t (drives k_t lookup)
            "risk_type_id": k_t_id,             # k_t (direct integer — bypass SCENARIO_TO_ID lookup)
            "scenario_type": scenario_type,     # human-readable k_t
            "risk_cues": risk_cues,             # h_t labels
            "relation_hint_id": h_t_id,         # h_t (direct integer)
            "pre_transaction_gap_sec": pre_transaction_gap_sec,  # a_t
            # Evidence metadata (for logging / audit)
            "_num_evidence": len(evidence),
            "_top_evidence_scores": [round(e.score, 4) for e in evidence[:3]],
        }

    @staticmethod
    def _scam_node_to_scenario(node_id: str) -> str:
        _MAP = {
            "ST_investment":    "investment_scam",
            "ST_romance":       "romance_scam",
            "ST_phishing":      "phishing_url_scam",
            "ST_impersonation": "impersonation_scam",
            "ST_urgent":        "urgent_transfer_request",
            "ST_fake_support":  "fake_customer_support",
            "ST_migration":     "crypto_wallet_migration_scam",
            "ST_recovery":      "recovery_phrase_stealing_attempt",
            "ST_high_yield":    "high_yield_guaranteed_return_scam",
            "ST_grooming":      "multi_stage_grooming_scam",
        }
        return _MAP.get(node_id, "benign")

    @staticmethod
    def _cue_node_to_label(node_id: str) -> str:
        _MAP = {
            "CUE_guaranteed_return": "guaranteed return",
            "CUE_urgent_transfer":   "urgent transfer",
            "CUE_wallet_request":    "external wallet request",
            "CUE_identity_verify":   "identity verification",
            "CUE_escrow":            "external escrow contract",
            "CUE_seed_phrase":       "external escrow contract",
            "CUE_url_link":          "external wallet request",
            "CUE_authority":         "identity verification",
        }
        return _MAP.get(node_id, "none")
