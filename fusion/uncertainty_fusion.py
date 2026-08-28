"""
fusion/uncertainty_fusion.py

Uncertainty-weighted fusion (the proposed method):

    R_t = (1 - β_t) * p̄_t + β_t * p_t^R

    β_t = σ(λ * Ũ_t + b)

where:
    p̄_t  = MC-dropout mean GNN prediction
    p_t^R = risk encoder output (from RiskEncoder)
    Ũ_t  = normalized MC uncertainty (variance / max_variance)
    λ     = sensitivity parameter (tuned on validation set)
    b     = bias term (tuned on validation set)

Interpretation:
    When GNN epistemic uncertainty is HIGH  → β_t → 1 → trust the risk branch more
    When GNN epistemic uncertainty is LOW   → β_t → 0 → trust the GNN more
"""

from __future__ import annotations

import numpy as np
import torch

from typing import Optional


class UncertaintyFusion:
    """
    Uncertainty-gated fusion between MC-GNN and risk encoder predictions.

    R_t = (1 - β_t) * p̄_t + β_t * p_t^R
    β_t = σ(λ * Ũ_t + b)

    Args:
        lambda_u     : Sensitivity to uncertainty (>0 → higher uncertainty → higher β).
                       Tuned on validation set; default=5.0.
        bias         : Logit bias for β_t. Negative values bias toward GNN.
                       Tuned on validation set; default=-2.0.
        min_beta     : Minimum β_t (ensure some risk branch contribution). Default=0.05.
        max_beta     : Maximum β_t (ensure some GNN contribution). Default=0.70.
        norm_mode    : How to normalize uncertainty before sigmoid.
                       'minmax'  — (u - u_min) / (u_max - u_min)
                       'zscore'  — (u - u_mean) / u_std
                       'none'    — use raw variance
    """

    def __init__(
        self,
        lambda_u: float = 5.0,
        bias: float = -2.0,
        min_beta: float = 0.05,
        max_beta: float = 0.70,
        norm_mode: str = "minmax",
    ):
        self.lambda_u = float(lambda_u)
        self.bias = float(bias)
        self.min_beta = float(min_beta)
        self.max_beta = float(max_beta)
        self.norm_mode = norm_mode

    def _normalize_uncertainty(self, u: torch.Tensor) -> torch.Tensor:
        """Normalize uncertainty tensor to [0, 1] range."""
        if self.norm_mode == "minmax":
            u_min = u.min()
            u_max = u.max()
            if (u_max - u_min).abs() < 1e-9:
                return torch.zeros_like(u)
            return (u - u_min) / (u_max - u_min + 1e-9)
        elif self.norm_mode == "zscore":
            mean = u.mean()
            std  = u.std()
            if std.abs() < 1e-9:
                return torch.zeros_like(u)
            return torch.sigmoid((u - mean) / (std + 1e-9))
        else:
            return u

    def fuse(
        self,
        p_gnn: torch.Tensor,
        u_mc: torch.Tensor,
        p_risk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p_gnn  : [N] MC-dropout mean GNN fraud probability
            u_mc   : [N] MC-dropout variance (epistemic uncertainty proxy)
            p_risk : [N] risk encoder fraud probability

        Returns:
            final_prob : [N] fused probability
            alpha_vec  : [N] GNN weight (= 1 - β_t)
            beta_vec   : [N] risk weight (= β_t)
        """
        p_gnn  = p_gnn.float().view(-1)
        u_mc   = u_mc.float().view(-1)
        p_risk = p_risk.float().view(-1)

        # Normalize uncertainty
        u_norm = self._normalize_uncertainty(u_mc)

        # β_t = σ(λ * Ũ_t + b)
        raw_beta = torch.sigmoid(self.lambda_u * u_norm + self.bias)

        # Clamp into [min_beta, max_beta]
        beta = self.min_beta + (self.max_beta - self.min_beta) * raw_beta
        alpha = 1.0 - beta

        # R_t = (1 - β_t) * p̄_t + β_t * p_t^R
        final = alpha * p_gnn + beta * p_risk
        return torch.clamp(final, 0.0, 1.0), alpha, beta

    @classmethod
    def from_config(cls, cfg: dict) -> "UncertaintyFusion":
        fc = cfg.get("fusion", {})
        return cls(
            lambda_u=float(fc.get("lambda_uncertainty", 5.0)),
            bias=float(fc.get("bias", -2.0)),
            min_beta=float(fc.get("min_local_weight", 0.05)),
            max_beta=float(fc.get("max_local_weight", 0.70)),
            norm_mode=str(fc.get("norm_mode", "minmax")),
        )

    def __repr__(self) -> str:
        return (
            f"UncertaintyFusion(λ={self.lambda_u}, b={self.bias}, "
            f"β∈[{self.min_beta},{self.max_beta}], norm={self.norm_mode})"
        )
