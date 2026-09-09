"""
fusion/fixed_fusion.py

Fixed-weight fusion baseline:

    R_t = (1 - α) * p̄_t + α * p_t^R

where α is a fixed scalar tuned on the validation set.

This is the primary ablation baseline for the uncertainty-weighted fusion.
The weight α is selected on the validation set and frozen for test evaluation.
"""

from __future__ import annotations

from typing import Optional
import torch


class FixedFusion:
    """
    Fixed-weight linear combination of GNN and risk encoder predictions.

    R_t = (1 - alpha) * p_gnn + alpha * p_risk

    Args:
        alpha: Weight for the risk encoder branch ∈ [0, 1].
               alpha=0   → GNN only
               alpha=1   → Semantic only (risk branch only)
               alpha=0.5 → Equal weight (default)
    """

    def __init__(self, alpha: float = 0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = float(alpha)

    def fuse(
        self,
        p_gnn: torch.Tensor,
        p_risk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p_gnn  : [N] GNN fraud probability from MC inference
            p_risk : [N] risk encoder fraud probability from risk branch

        Returns:
            final_prob : [N] fused probability
            alpha_vec  : [N] GNN weight (= 1 - alpha, constant)
            beta_vec   : [N] risk weight (= alpha, constant)
        """
        p_gnn  = p_gnn.float().view(-1)
        p_risk = p_risk.float().view(-1)

        beta  = torch.full_like(p_gnn, self.alpha)
        alpha = 1.0 - beta
        final = alpha * p_gnn + beta * p_risk
        return torch.clamp(final, 0.0, 1.0), alpha, beta

    @classmethod
    def gnn_only(cls) -> "FixedFusion":
        """Ablation: GNN only (α=0)."""
        return cls(alpha=0.0)

    @classmethod
    def semantic_only(cls) -> "FixedFusion":
        """Ablation: Semantic (risk) only (α=1)."""
        return cls(alpha=1.0)

    @classmethod
    def equal_weight(cls) -> "FixedFusion":
        """Standard 50-50 fixed fusion baseline."""
        return cls(alpha=0.5)

    def __repr__(self) -> str:
        return f"FixedFusion(alpha={self.alpha:.3f})"
