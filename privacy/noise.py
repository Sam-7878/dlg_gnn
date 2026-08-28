"""
privacy/noise.py

Controlled noise injection for privacy protection of continuous risk vector components.

Mechanisms:
    Gaussian  : score += N(0, σ²)
    Laplace   : score += Laplace(0, b)  [b = σ/√2 for same variance]
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Any


class NoiseMechanism:
    """
    Injects calibrated noise into continuous components of the risk vector.

    Only s_t and q_t are perturbed (continuous risk values).
    k_t and h_t (categorical IDs) and a_t (age) are left unchanged
    to preserve utility while providing privacy.

    Args:
        mechanism : 'gaussian' or 'laplace'
        scale     : Noise scale (σ for Gaussian, b for Laplace). Default=0.05.
        seed      : Random seed for reproducibility.
    """

    def __init__(
        self,
        mechanism: str = "gaussian",
        scale: float = 0.05,
        seed: int = 42,
    ):
        if mechanism not in ("gaussian", "laplace"):
            raise ValueError(f"mechanism must be 'gaussian' or 'laplace', got {mechanism!r}")
        self.mechanism = mechanism
        self.scale = float(scale)
        self.rng = np.random.RandomState(seed)

    def _sample_noise(self) -> float:
        if self.mechanism == "gaussian":
            return float(self.rng.normal(0, self.scale))
        else:
            return float(self.rng.laplace(0, self.scale))

    def apply(self, risk_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Return a new risk dict with noise applied to s_t and q_t."""
        result = dict(risk_dict)

        s = float(risk_dict.get("local_risk_score", 0.0))
        q = float(risk_dict.get("confidence", 0.0))

        result["local_risk_score"] = float(np.clip(s + self._sample_noise(), 0.0, 1.0))
        result["confidence"]       = float(np.clip(q + self._sample_noise(), 0.0, 1.0))
        result["privacy_mode"]     = "noisy_risk_vector"
        return result

    def apply_batch(self, risk_dicts: list) -> list:
        return [self.apply(d) for d in risk_dicts]
