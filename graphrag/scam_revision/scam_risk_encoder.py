"""
graphrag/scam_revision/scam_risk_encoder.py

Phase K: Risk Vector v2 & Neural Risk Encoder

Encodes retrieved multi-hop graph context into 7-dimensional Risk Vector v2:
    r_t = [s_t, q_t, k_t, a_t, h_t, b_t, c_t]

where:
    s_t : Semantic scam risk score (0.0 to 1.0)
    q_t : Evidence retrieval confidence (0.0 to 1.0)
    k_t : Scam/campaign category severity index (0.0 to 1.0)
    a_t : Evidence age / freshness factor (0.0 to 1.0)
    h_t : Relational support strength (graph neighborhood density)
    b_t : Bridge confidence (Tier 1/3 exact = 1.0, Tier 2 = 0.85, Tier 4 = 0.5)
    c_t : Cross-source corroboration indicator (1.0 if dual-verified CST+CSDB)

The neural risk head projects r_t into risk prediction probability p_t^RAG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from dlg_gnn.graphrag.scam_revision.scam_graphrag_retriever import RetrievedEvidence, RetrievalQueryResult


@dataclass
class RiskVectorV2:
    s_t: float  # semantic score
    q_t: float  # confidence
    k_t: float  # category severity
    a_t: float  # freshness / age
    h_t: float  # relational support
    b_t: float  # bridge confidence
    c_t: float  # corroboration count

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.s_t, self.q_t, self.k_t, self.a_t, self.h_t, self.b_t, self.c_t],
            dtype=torch.float32,
        )

    def to_list(self) -> List[float]:
        return [self.s_t, self.q_t, self.k_t, self.a_t, self.h_t, self.b_t, self.c_t]


class ScamRiskExtractor:
    """
    Extracts Risk Vector v2 from retrieved GraphRAG evidence.
    """
    def __init__(self, current_reference_time: int = 1700000000):
        self.ref_time = current_reference_time

    def extract(self, retrieval_result: RetrievalQueryResult) -> RiskVectorV2:
        evidence = retrieval_result.evidence_list
        if not evidence:
            return RiskVectorV2(
                s_t=0.1, q_t=0.1, k_t=0.0, a_t=0.5, h_t=0.0, b_t=0.0, c_t=0.0
            )

        # 1. Semantic score (mean of top evidence)
        s_t = float(np.mean([e.semantic_score for e in evidence]))

        # 2. Evidence confidence
        scam_flags = [1.0 if e.is_scam_ground_truth else 0.0 for e in evidence]
        q_t = float(np.mean(scam_flags)) if scam_flags else 0.1

        # 3. Category severity
        k_t = 0.95 if any("phishing" in e.label_name.lower() or "scam" in e.label_name.lower() for e in evidence) else 0.4

        # 4. Freshness / age
        timestamps = [e.timestamp for e in evidence if e.timestamp is not None]
        if timestamps:
            delta_days = max(0, (self.ref_time - np.mean(timestamps)) / 86400.0)
            a_t = float(np.exp(-delta_days / 365.0))  # 1-year half-life decay
        else:
            a_t = 0.5

        # 5. Relational support strength (ratio of non-zero hop evidence)
        relational_count = sum(1 for e in evidence if e.hop_distance > 0)
        h_t = min(1.0, relational_count / 5.0)

        # 6. Bridge confidence
        bridge_scores = []
        for e in evidence:
            if "MultiSource" in e.provenance or "CST" in e.provenance or "CSDB" in e.provenance:
                bridge_scores.append(1.0)
            elif e.hop_distance > 0:
                bridge_scores.append(0.85)
            else:
                bridge_scores.append(0.5)
        b_t = float(np.mean(bridge_scores)) if bridge_scores else 0.5

        # 7. Cross-source corroboration indicator
        c_t = 1.0 if any("MultiSource" in e.provenance for e in evidence) or (len(set(e.provenance for e in evidence)) >= 2) else 0.0

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
    MLP risk classification head projecting Risk Vector v2 into p_t^RAG.
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
