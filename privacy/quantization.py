"""
privacy/quantization.py

float32 → int8 quantization for the risk vector.

Reduces payload size and provides a privacy mechanism by reducing precision.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


class Quantizer:
    """
    Quantizes continuous float32 components of a risk vector to int8.

    Components quantized: local_risk_score (s_t), confidence (q_t), context_age_sec (a_t)
    Components kept as int: risk_type_id (k_t), relation_hint_id (h_t)

    Quantization scheme:
        float32 ∈ [0, 1]  →  int8 ∈ [0, 127]   (scale = 127)
        age in seconds     →  int8 clamped to [0, 127] after log1p normalization
    """

    SCALE_01 = 127.0   # for values in [0, 1]
    AGE_LOG_NORM = 10.0  # log1p normalization factor for age

    def quantize(self, risk_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Quantize a risk vector dict. Returns a new dict with int8-range values."""
        s = float(risk_dict.get("local_risk_score", 0.0))
        q = float(risk_dict.get("confidence", 0.0))
        a = float(risk_dict.get("context_age_sec", 0))

        s_q = int(np.clip(round(s * self.SCALE_01), 0, 127))
        q_q = int(np.clip(round(q * self.SCALE_01), 0, 127))
        # Age: log1p normalize then scale to [0, 127]
        a_norm = np.log1p(a) / self.AGE_LOG_NORM
        a_q = int(np.clip(round(a_norm * self.SCALE_01), 0, 127))

        return {
            "event_id": risk_dict.get("event_id"),
            "local_risk_score": s_q / self.SCALE_01,     # dequantized for model input
            "local_risk_score_int8": s_q,                 # raw int8 for byte measurement
            "confidence": q_q / self.SCALE_01,
            "confidence_int8": q_q,
            "risk_type_id": int(risk_dict.get("risk_type_id", 0)),
            "context_age_sec": a_q / self.SCALE_01 * self.AGE_LOG_NORM,
            "context_age_sec_int8": a_q,
            "relation_hint_id": int(risk_dict.get("relation_hint_id", 0)),
            "privacy_mode": "quantized_risk_vector",
        }

    def dequantize_score(self, int8_val: int) -> float:
        return float(np.clip(int8_val, 0, 127)) / self.SCALE_01

    def payload_bytes(self) -> int:
        """Theoretical binary payload: 3 int8 + 2 uint8 = 5 bytes."""
        return 5
