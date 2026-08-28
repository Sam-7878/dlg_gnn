"""
fusion/learned_fusion.py

Learned fusion (ablation baseline — no uncertainty):

    z = [p̄_t || p_t^R]   (concatenation)
    R_t = σ(MLP(z))

This is the "fixed structure, learned weight" baseline. Unlike UncertaintyFusion,
it does NOT use MC uncertainty to gate the fusion. It learns the optimal combination
from training data via a small MLP.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class LearnedFusion(nn.Module):
    """
    MLP-based learned fusion of GNN and risk encoder predictions.

    Concatenates [p̄_t, p_t^R] → 2-layer MLP → R_t

    Args:
        hidden_dim  : MLP hidden size. Default=16.
        dropout     : Dropout rate. Default=0.1.
    """

    def __init__(self, hidden_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Initialize output layer toward equal weighting
        with torch.no_grad():
            self.net[-1].weight.fill_(0.5)

    def forward(
        self,
        p_gnn: torch.Tensor,
        p_risk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            p_gnn  : [N] GNN probability
            p_risk : [N] risk encoder probability

        Returns:
            final_prob : [N]
            alpha_vec  : [N] (dummy — not meaningful for learned fusion)
            beta_vec   : [N] (dummy)
        """
        p_gnn  = p_gnn.float().view(-1, 1)
        p_risk = p_risk.float().view(-1, 1)
        x = torch.cat([p_gnn, p_risk], dim=-1)          # [N, 2]
        logits = self.net(x).squeeze(-1)                  # [N]
        final  = torch.sigmoid(logits)
        dummy  = torch.full_like(final, 0.5)
        return final, dummy, dummy

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "LearnedFusion":
        fc = cfg.get("fusion", {}).get("learned", {})
        return cls(
            hidden_dim=int(fc.get("hidden_dim", 16)),
            dropout=float(fc.get("dropout", 0.1)),
        )
