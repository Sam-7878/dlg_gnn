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
    "hard_negative": 11
}

CUE_TO_ID = {
    "none": 0,
    "guaranteed return": 1,
    "urgent transfer": 2,
    "external wallet request": 3,
    "identity verification": 4,
    "external escrow contract": 5
}

class RiskVectorizer:
    """
    Transforms raw text context into privacy-preserving abstract risk vectors.
    Strictly filters out any raw string logs before sending data to the server.
    """
    def __init__(self, privacy_mode: str = "full_risk_vector", noise_scale: float = 0.05):
        self.privacy_mode = privacy_mode.lower()
        self.noise_scale = noise_scale

    def quantize_score(self, score: float) -> float:
        """Quantizes the raw score into discrete buckets (Low: 0.2, Medium: 0.6, High: 0.9)."""
        if score < 0.4:
            return 0.2
        elif score < 0.75:
            return 0.6
        else:
            return 0.9

    def add_noise(self, score: float) -> float:
        """Injects Laplace/Gaussian noise to privatize risk scores."""
        noise = np.random.normal(0, self.noise_scale)
        return float(np.clip(score + noise, 0.0, 1.0))

    def vectorize(self, context_dict: dict) -> dict:
        """
        Transforms a context dictionary into a sanitized numeric vector.
        """
        event_id = context_dict.get("event_id")
        raw_score = float(context_dict.get("local_risk_score", 0.0))
        
        # If score is not present, calculate based on label/risk cues
        if "local_risk_score" not in context_dict:
            label = int(context_dict.get("label", 0))
            if label == 1:
                # High risk base
                raw_score = 0.85
            else:
                scenario = context_dict.get("scenario_type", "benign")
                raw_score = 0.55 if scenario == "hard_negative" else 0.10
                
        # Resolve privacy mode transformations
        if self.privacy_mode == "quantized_risk_vector":
            risk_score = self.quantize_score(raw_score)
        elif self.privacy_mode == "noisy_risk_vector":
            risk_score = self.add_noise(raw_score)
        elif self.privacy_mode == "minimal_risk_token":
            risk_score = 0.9 if raw_score >= 0.75 else (0.5 if raw_score >= 0.4 else 0.1)
        elif self.privacy_mode == "raw_context":
            # Baseline (no privacy protection)
            risk_score = raw_score
        else:  # full_risk_vector
            risk_score = raw_score

        scenario_type = context_dict.get("scenario_type", "benign")
        risk_type_id = SCENARIO_TO_ID.get(scenario_type, 0)
        
        cues = context_dict.get("risk_cues", [])
        relation_hint_id = CUE_TO_ID.get(cues[0], 0) if cues else 0
        
        confidence = float(context_dict.get("confidence", 0.90))
        age = int(context_dict.get("pre_transaction_gap_sec", 0))

        # Output sanitized risk vector payload
        return {
            "event_id": event_id,
            "local_risk_score": risk_score,
            "risk_type_id": risk_type_id,
            "confidence": confidence,
            "context_age_sec": age,
            "relation_hint_id": relation_hint_id,
            "privacy_mode": self.privacy_mode
        }
