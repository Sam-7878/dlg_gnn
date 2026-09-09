"""
privacy/__init__.py

Privacy module for the dlg_gnn GraphRAG pipeline.

Provides:
    VectorCodec     : serialize risk vectors → bytes (JSON / compact binary)
    Quantizer       : float32 → int8 quantization
    NoiseMechanism  : Gaussian / Laplace noise injection
    LeakageAttack   : attribute inference attack on risk representations
"""

from privacy.vector_codec import VectorCodec, PayloadBreakdown
from privacy.quantization import Quantizer
from privacy.noise import NoiseMechanism
from privacy.leakage_attack import LeakageAttack

__all__ = ["VectorCodec", "PayloadBreakdown", "Quantizer", "NoiseMechanism", "LeakageAttack"]
