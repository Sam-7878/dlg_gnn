"""
risk_vectorizer.py

Transforms a context dict (from GraphRAG retrieval) into a sanitized 5-component
risk vector r_t = [s_t, q_t, k_t, a_t, h_t] for server-side transmission.

LABEL LEAKAGE POLICY:
  - This module does NOT access the 'label' field of the context dict.
  - risk scores must come from GraphRAG retrieval output, not from ground truth.
  - If 'local_risk_score' is absent (e.g. GraphRAG not run yet), score defaults
    to 0.0 (unknown risk) — NOT derived from label.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

SCENARIO_TO_ID = {
    "benign": 0,
    "investment_scam": 1,
    "romance_scam": 2,
    "phishing_url_scam": 3,
    "impersonation_scam": 4,
    "urgent_transfer_request": 5,
    "fake_customer_support": 6,
    "crypto_wallet_migration_scam": 7,
    "recovery_phrase_stealing_attempt": 8,
    "high_yield_guaranteed_return_scam": 9,
    "multi_stage_grooming_scam": 10,
    "hard_negative": 11,
    "hard_positive": 12,
}

CUE_TO_ID = {
    "none": 0,
    "guaranteed return": 1,
    "urgent transfer": 2,
    "external wallet request": 3,
    "identity verification": 4,
    "external escrow contract": 5,
}


class RiskVectorizer:
    """
    Transforms a context dict produced by GraphRAG retrieval into a
    privacy-preserving numeric risk vector for server-side use.

    The vector r_t = [s_t, q_t, k_t, a_t, h_t] contains:
        s_t  (local_risk_score)  : semantic risk score  ∈ [0, 1]
        q_t  (confidence)        : extraction confidence ∈ [0, 1]
        k_t  (risk_type_id)      : scam category integer
        a_t  (context_age_sec)   : pre-transaction gap in seconds
        h_t  (relation_hint_id)  : primary risk cue integer

    Privacy modes control how the vector is transformed before transmission:
        full_risk_vector     : r_t as-is
        quantized_risk_vector: continuous values → discrete buckets
        noisy_risk_vector    : Gaussian noise added to continuous values
        minimal_risk_token   : score + category only (3-level bucketing)
        raw_context          : baseline — no privacy protection (upper bound)
    """

    def __init__(self, privacy_mode: str = "full_risk_vector", noise_scale: float = 0.05):
        self.privacy_mode = privacy_mode.lower()
        self.noise_scale = noise_scale

    def quantize_score(self, score: float) -> float:
        """Quantize continuous score into 3 discrete buckets."""
        if score < 0.4:
            return 0.2
        elif score < 0.75:
            return 0.6
        else:
            return 0.9

    def add_noise(self, score: float) -> float:
        """Inject Gaussian noise to privatize a continuous risk score."""
        noise = np.random.normal(0, self.noise_scale)
        return float(np.clip(score + noise, 0.0, 1.0))

    def vectorize(self, context_dict: dict) -> dict:
        """
        Transform a GraphRAG output context dict into a sanitized risk vector.

        Expected keys (set by GraphRAG retrieval, NOT by label):
            local_risk_score      : float  (from retrieval scoring)
            confidence            : float  (extraction quality)
            scenario_type         : str    (detected scam category)
            risk_cues             : list   (surface cues)
            pre_transaction_gap_sec: int   (age of context)
            event_id              : str    (transaction ID)

        Forbidden keys (must not be used):
            label                 : NEVER accessed here — label leakage prevention
        """
        event_id = context_dict.get("event_id")

        # ── s_t: semantic risk score ────────────────────────────────────────
        # Comes from GraphRAG retrieval scoring, NOT from label.
        # Default = 0.0 (unknown) if GraphRAG has not computed it yet.
        raw_score = float(context_dict.get("local_risk_score", 0.0))

        # Validate: refuse to use label as a score proxy
        if "label" in context_dict and "local_risk_score" not in context_dict:
            logger.warning(
                "context_dict contains 'label' but no 'local_risk_score'. "
                "Defaulting risk score to 0.0 (label ignored — leakage prevention)."
            )

        # ── Apply privacy transformation ────────────────────────────────────
        if self.privacy_mode == "quantized_risk_vector":
            risk_score = self.quantize_score(raw_score)
        elif self.privacy_mode == "noisy_risk_vector":
            risk_score = self.add_noise(raw_score)
        elif self.privacy_mode == "minimal_risk_token":
            risk_score = 0.9 if raw_score >= 0.75 else (0.5 if raw_score >= 0.4 else 0.1)
        else:
            # full_risk_vector or raw_context
            risk_score = raw_score

        # ── k_t: scam category ──────────────────────────────────────────────
        scenario_type = context_dict.get("scenario_type", "benign")
        risk_type_id = SCENARIO_TO_ID.get(scenario_type, 0)

        # ── h_t: relation hint ──────────────────────────────────────────────
        cues = context_dict.get("risk_cues", [])
        relation_hint_id = CUE_TO_ID.get(cues[0], 0) if cues else 0

        # ── q_t: extraction confidence ──────────────────────────────────────
        confidence = float(context_dict.get("confidence", 0.90))

        # ── a_t: context age ────────────────────────────────────────────────
        age = int(context_dict.get("pre_transaction_gap_sec", 0))

        return {
            "event_id": event_id,
            "local_risk_score": risk_score,   # s_t
            "confidence": confidence,          # q_t
            "risk_type_id": risk_type_id,      # k_t
            "context_age_sec": age,            # a_t
            "relation_hint_id": relation_hint_id,  # h_t
            "privacy_mode": self.privacy_mode,
        }
