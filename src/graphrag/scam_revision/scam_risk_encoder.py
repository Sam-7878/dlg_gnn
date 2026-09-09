"""
graphrag/scam_revision/scam_risk_encoder.py

Phase K (Round 2 Anti-Circularity Sanitized): Risk Vector v2 & Neural Risk Encoder

Sanitization Policy:
- Zero ground-truth label leakage (CST/CSDB membership, is_scam, linked_to_scam, category flags are strictly forbidden as input features).
- Purely observable features available at pre-detection inference time:
    s_t : Linguistic / semantic risk score (giveaway, yield, urgency, free coins keyword signals)
    q_t : Semantic retrieval confidence (mean similarity of top retrieved items)
    k_t : Observable promotional structure category
    a_t : Temporal freshness factor (evidence age relative to query)
    h_t : Structural relational support (graph neighborhood density / degree)
    b_t : Bridge topology strength (presence of associated domains and wallets)
    c_t : Multi-platform entity corroboration count (number of connected domains/handles)

The neural risk head projects r_t into risk prediction probability p_t^RAG.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from dlg_gnn.graphrag.scam_revision.scam_graphrag_retriever import (
    RetrievedEvidence,
    RetrievalQueryResult,
)


@dataclass
class RiskVectorV2:
    s_t: float  # semantic scam signal score
    q_t: float  # retrieval confidence
    k_t: float  # promotional category structure
    a_t: float  # freshness / age
    h_t: float  # relational graph support
    b_t: float  # bridge topology connectivity
    c_t: float  # multi-platform corroboration count

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.s_t, self.q_t, self.k_t, self.a_t, self.h_t, self.b_t, self.c_t],
            dtype=torch.float32,
        )

    def to_list(self) -> List[float]:
        return [self.s_t, self.q_t, self.k_t, self.a_t, self.h_t, self.b_t, self.c_t]


# Pre-compiled high-risk linguistic patterns (observable text only)
REGEX_HIGH_RISK_TERMS = re.compile(
    r"\b(giveaway|double|airdrop|free\s*crypto|500%|100%\s*profit|guaranteed\s*return|"
    r"claim\s*now|urgent|deposit\s*to|send\s*to|private\s*key|seed\s*phrase|"
    r"telegram\s*bot|bounty\s*stakes|exclusive\s*pool)\b",
    re.IGNORECASE,
)


class ScamRiskExtractor:
    """
    Sanitized Risk Vector v2 extractor operating exclusively on observable features.
    """
    def __init__(self, current_reference_time: int = 1700000000):
        self.ref_time = current_reference_time

    def extract(self, retrieval_result: RetrievalQueryResult) -> RiskVectorV2:
        evidence = retrieval_result.evidence_list
        if not evidence:
            return RiskVectorV2(
                s_t=0.05, q_t=0.05, k_t=0.1, a_t=0.5, h_t=0.0, b_t=0.0, c_t=0.0
            )

        # 1. Semantic score s_t (observable keyword and semantic match)
        semantic_scores = [e.semantic_score for e in evidence]
        s_t = float(np.mean(semantic_scores)) if semantic_scores else 0.1

        # Check for observable high-risk linguistic triggers in evidence text
        text_hits = 0
        for e in evidence:
            if REGEX_HIGH_RISK_TERMS.search(e.label_name):
                text_hits += 1
        linguistic_boost = min(0.4, text_hits * 0.1)
        s_t = float(np.clip(s_t + linguistic_boost, 0.0, 1.0))

        # 2. Retrieval confidence q_t (variance of semantic similarity)
        q_t = float(np.mean(semantic_scores)) if semantic_scores else 0.1

        # 3. Promotional structure k_t (observable entity type distribution)
        has_campaign = any(e.node_type == "Campaign" for e in evidence)
        has_domain = any(e.node_type == "Domain" for e in evidence)
        has_wallet = any(e.node_type == "Wallet" for e in evidence)
        k_t = (0.4 if has_campaign else 0.0) + (0.3 if has_domain else 0.0) + (0.3 if has_wallet else 0.0)

        # 4. Freshness / age a_t
        timestamps = [e.timestamp for e in evidence if e.timestamp is not None]
        if timestamps:
            delta_days = max(0, (self.ref_time - np.mean(timestamps)) / 86400.0)
            a_t = float(np.exp(-delta_days / 365.0))  # Exponential time decay
        else:
            a_t = 0.5

        # 5. Relational support strength h_t (number of non-zero hop neighbors)
        non_zero_hops = sum(1 for e in evidence if e.hop_distance > 0)
        h_t = min(1.0, non_zero_hops / 5.0)

        # 6. Bridge topology connectivity b_t (cross-layer structural connectivity)
        # Higher if evidence connects both social and financial entities
        b_t = 1.0 if (has_domain and has_wallet) else (0.6 if (has_domain or has_wallet) else 0.2)

        # 7. Multi-platform corroboration count c_t (diversity of distinct entity types)
        distinct_types = len(set(e.node_type for e in evidence))
        c_t = min(1.0, distinct_types / 3.0)

        return RiskVectorV2(
            s_t=float(np.clip(s_t, 0.0, 1.0)),
            q_t=float(np.clip(q_t, 0.0, 1.0)),
            k_t=float(np.clip(k_t, 0.0, 1.0)),
            a_t=float(np.clip(a_t, 0.0, 1.0)),
            h_t=float(np.clip(h_t, 0.0, 1.0)),
            b_t=float(np.clip(b_t, 0.0, 1.0)),
            c_t=float(np.clip(c_t, 0.0, 1.0)),
        )


class ScamRiskEncoderHead(nn.Module):
    """
    MLP risk classification head projecting sanitized Risk Vector v2 into p_t^RAG.
    """
    def __init__(self, in_dim: int = 7, hidden_dim: int = 32, dropout_p: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, r_vecs: torch.Tensor) -> torch.Tensor:
        if r_vecs.dim() == 1:
            r_vecs = r_vecs.unsqueeze(0)
        return self.net(r_vecs).squeeze(-1)
