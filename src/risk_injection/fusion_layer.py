import numpy as np
import logging

logger = logging.getLogger(__name__)

class FusionLayer:
    """
    Fuses server GNN prediction with local GraphRAG risk vector.
    Supports fixed-weight, rule-constrained, and uncertainty-weighted dynamic fusion.
    """
    def __init__(self, 
                 method: str = "uncertainty_weighted", 
                 lambda_uncertainty: float = 5.0, 
                 min_local_weight: float = 0.1, 
                 max_local_weight: float = 0.7, 
                 use_confidence: bool = True, 
                 use_context_age_decay: bool = True, 
                 age_decay_tau_sec: float = 3600):
        self.method = method.lower()
        self.lambda_uncertainty = lambda_uncertainty
        self.min_local_weight = min_local_weight
        self.max_local_weight = max_local_weight
        self.use_confidence = use_confidence
        self.use_context_age_decay = use_context_age_decay
        self.age_decay_tau_sec = age_decay_tau_sec

    def fuse(self, gnn_prob: float, u_mc: float, risk_vector: dict) -> tuple[float, float, float]:
        """
        Combines GNN fraud probability and local risk score.
        
        Returns:
            final_prob: float
            alpha: GNN weight
            beta: Local risk weight
        """
        r_local = float(risk_vector.get("local_risk_score", 0.0))
        confidence = float(risk_vector.get("confidence", 1.0))
        age = float(risk_vector.get("context_age_sec", 0.0))
        risk_type_id = int(risk_vector.get("risk_type_id", 0))

        # 1. Apply decay to local risk based on age
        if self.use_context_age_decay and self.age_decay_tau_sec > 0:
            decay_factor = np.exp(-age / self.age_decay_tau_sec)
            r_local = r_local * decay_factor

        # 2. Adjust local risk with confidence
        if self.use_confidence:
            r_local = r_local * confidence

        # 3. Dynamic Weighting calculations
        if self.method == "fixed_weight":
            beta = self.min_local_weight
            alpha = 1.0 - beta
            final_prob = alpha * gnn_prob + beta * r_local

        elif self.method == "uncertainty_weighted":
            # Sigmoid response to GNN epistemic uncertainty U_mc
            # If GNN has high variance (uncertainty), beta increases to trust GraphRAG text cues
            raw_beta = 1.0 / (1.0 + np.exp(-self.lambda_uncertainty * u_mc + 2.0)) # shift curve
            # Scale beta into allowed bounds
            beta = self.min_local_weight + (self.max_local_weight - self.min_local_weight) * raw_beta
            alpha = 1.0 - beta
            final_prob = alpha * gnn_prob + beta * r_local

        elif self.method == "rule_constrained":
            # Direct overriding rules for high severity scam types
            # 8: recovery phrase theft, 3: phishing URL
            beta = self.min_local_weight
            alpha = 1.0 - beta
            final_prob = alpha * gnn_prob + beta * r_local
            
            if risk_type_id in [3, 8] and r_local >= 0.7:
                # Force override to alert category
                final_prob = max(final_prob, 0.90)
        else:
            # Fallback to pure GNN
            beta = 0.0
            alpha = 1.0
            final_prob = gnn_prob

        return float(np.clip(final_prob, 0.0, 1.0)), float(alpha), float(beta)
