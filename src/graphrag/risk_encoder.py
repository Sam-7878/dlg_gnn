"""
graphrag/risk_encoder.py

Risk Encoder φ(r_t) — converts the 5-component risk vector into a learned
latent representation z_t^R, then produces a semantic fraud probability p_t^R.

Architecture:
    s_t ──────────────┐
    q_t ──────────────┤
    a_t (normalized)  ├── MLP → z_t^R → Linear → p_t^R ∈ [0,1]
    Emb(k_t) ─────────┤
    Emb(h_t) ─────────┘

This replaces the scalar s_t late-fusion with a proper learned representation
that uses ALL five components of r_t.

Config (configs/base.yaml → risk_encoder):
    hidden_dim  : 32
    output_dim  : 16
    num_k_categories: 13   (number of scam categories)
    num_h_categories: 6    (number of relation hints)
    k_embed_dim : 8
    h_embed_dim : 8
    dropout     : 0.1
    age_norm_tau: 3600.0   (seconds — normalizes a_t to [0, 1] range)
"""

from __future__ import annotations

import math
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskEncoder(nn.Module):
    """
    Encodes the 5-component risk vector r_t into z_t^R and p_t^R.

    Input (per sample):
        s_t  : float scalar ∈ [0, 1]   — semantic risk score
        q_t  : float scalar ∈ [0, 1]   — extraction confidence
        a_t  : float scalar ≥ 0         — context age in seconds
        k_t  : int                       — scam category ID
        h_t  : int                       — relation hint ID

    Output:
        z_t_R : Tensor [*, output_dim]  — risk latent representation
        p_t_R : Tensor [*]              — fraud probability from risk branch
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        output_dim: int = 16,
        num_k_categories: int = 13,
        num_h_categories: int = 6,
        k_embed_dim: int = 8,
        h_embed_dim: int = 8,
        dropout: float = 0.1,
        age_norm_tau: float = 3600.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.age_norm_tau = age_norm_tau

        # Categorical embeddings
        self.k_embed = nn.Embedding(num_k_categories, k_embed_dim, padding_idx=0)
        self.h_embed = nn.Embedding(num_h_categories, h_embed_dim, padding_idx=0)

        # Input dimension: s_t (1) + q_t (1) + a_t_norm (1) + k_embed + h_embed
        in_dim = 3 + k_embed_dim + h_embed_dim

        # MLP: in_dim → hidden_dim → hidden_dim → output_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Final classifier head: z_t^R → p_t^R
        self.classifier = nn.Linear(output_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def _normalize_age(self, a_t: torch.Tensor) -> torch.Tensor:
        """Normalize age to [0, 1] using exponential decay proxy."""
        return torch.exp(-a_t / max(self.age_norm_tau, 1e-6))

    def forward(
        self,
        s_t: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        a_t: torch.Tensor,
        h_t: torch.Tensor,
    ):
        """
        Args:
            s_t : [N] float — semantic risk score
            q_t : [N] float — extraction confidence
            k_t : [N] long  — scam category IDs
            a_t : [N] float — context age (seconds)
            h_t : [N] long  — relation hint IDs

        Returns:
            z_t_R : [N, output_dim]
            p_t_R : [N]
        """
        # Ensure float32 for continuous features
        s_t = s_t.float().view(-1, 1)
        q_t = q_t.float().view(-1, 1)
        a_t_norm = self._normalize_age(a_t.float()).view(-1, 1)

        k_emb = self.k_embed(k_t.long())   # [N, k_embed_dim]
        h_emb = self.h_embed(h_t.long())   # [N, h_embed_dim]

        x = torch.cat([s_t, q_t, a_t_norm, k_emb, h_emb], dim=-1)  # [N, in_dim]
        z_t_R = self.mlp(x)                                           # [N, output_dim]
        p_t_R = torch.sigmoid(self.classifier(z_t_R)).squeeze(-1)    # [N]

        return z_t_R, p_t_R

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "RiskEncoder":
        """Construct from a config dict (e.g. loaded from configs/base.yaml)."""
        rc = cfg.get("risk_encoder", {})
        return cls(
            hidden_dim=int(rc.get("hidden_dim", 32)),
            output_dim=int(rc.get("output_dim", 16)),
            num_k_categories=int(rc.get("num_k_categories", 13)),
            num_h_categories=int(rc.get("num_h_categories", 6)),
            k_embed_dim=int(rc.get("k_embed_dim", 8)),
            h_embed_dim=int(rc.get("h_embed_dim", 8)),
            dropout=float(rc.get("dropout", 0.1)),
            age_norm_tau=float(rc.get("age_norm_tau", 3600.0)),
        )

    def encode_risk_dict_batch(self, risk_dicts: list, device: Optional[torch.device] = None):
        """
        Convenience method: encode a list of risk_vector dicts (from RiskVectorizer).

        Args:
            risk_dicts: List of dicts with keys: local_risk_score, confidence,
                        risk_type_id, context_age_sec, relation_hint_id
            device: target device

        Returns:
            z_t_R : [N, output_dim]
            p_t_R : [N]
        """
        dev = device or next(self.parameters()).device
        n = len(risk_dicts)
        s = torch.tensor([d.get("local_risk_score", 0.0) for d in risk_dicts], dtype=torch.float32)
        q = torch.tensor([d.get("confidence", 0.9) for d in risk_dicts], dtype=torch.float32)
        k = torch.tensor([int(d.get("risk_type_id", 0)) for d in risk_dicts], dtype=torch.long)
        a = torch.tensor([float(d.get("context_age_sec", 0)) for d in risk_dicts], dtype=torch.float32)
        h = torch.tensor([int(d.get("relation_hint_id", 0)) for d in risk_dicts], dtype=torch.long)
        return self.forward(s.to(dev), q.to(dev), k.to(dev), a.to(dev), h.to(dev))
