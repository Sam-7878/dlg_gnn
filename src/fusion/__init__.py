"""
fusion/__init__.py

Fusion strategies for combining GNN prediction (p̄_t) with risk encoder output (p_t^R).

Available strategies:
    FixedFusion         : R = (1-α)*p̄_t + α*p_t^R  (fixed α)
    UncertaintyFusion   : R = (1-β_t)*p̄_t + β_t*p_t^R  (β_t = σ(λ*Ũ_t + b))
    LearnedFusion       : R = MLP([p̄_t, p_t^R]) — no uncertainty

Usage:
    from fusion import UncertaintyFusion, FixedFusion, LearnedFusion
"""

from fusion.fixed_fusion import FixedFusion
from fusion.uncertainty_fusion import UncertaintyFusion
from fusion.learned_fusion import LearnedFusion

__all__ = ["FixedFusion", "UncertaintyFusion", "LearnedFusion"]
