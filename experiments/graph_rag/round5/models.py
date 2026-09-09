"""Temporal and fraud-specific baseline definitions for GoG-SCIMain-v1.

These implementations are deliberately kept separate from the publication
results.  A class being available here does not make its result complete; the
Round 5 gate additionally requires training on the frozen chronological data.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINConv, SAGEConv, global_max_pool, global_mean_pool


class HarmonicTimeEncoding(nn.Module):
    """Learnable harmonic encoding for normalized event time."""

    def __init__(self, dimensions: int = 16):
        super().__init__()
        self.frequency = nn.Parameter(torch.logspace(0, 3, dimensions))
        self.phase = nn.Parameter(torch.zeros(dimensions))

    def forward(self, normalized_time: torch.Tensor) -> torch.Tensor:
        value = normalized_time.float().reshape(-1, 1)
        return torch.cos(value * self.frequency.reshape(1, -1) + self.phase)


class TGATBaseline(nn.Module):
    """TGAT-style causal local-graph baseline with event-time encoding.

    It does not claim bit-level equivalence to the original TGAT repository;
    the method name in result tables must therefore remain ``TGAT-style``.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 48, dropout: float = 0.3, num_chains: int = 3):
        super().__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=2, concat=False, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=2, concat=False, dropout=dropout)
        self.time = HarmonicTimeEncoding(16)
        self.dropout = dropout
        self.num_chains = num_chains
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 16 + num_chains, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, data, normalized_time: torch.Tensor) -> torch.Tensor:
        hidden = F.elu(self.conv1(data.x, data.edge_index))
        hidden = F.dropout(hidden, self.dropout, self.training)
        hidden = F.elu(self.conv2(hidden, data.edge_index))
        pooled = torch.cat((global_mean_pool(hidden, data.batch), global_max_pool(hidden, data.batch)), dim=1)
        chain = F.one_hot(data.chain_id.long(), num_classes=self.num_chains).float()
        return self.head(torch.cat((pooled, self.time(normalized_time), chain), dim=1)).squeeze(-1)


class TemporalMemoryBaseline(nn.Module):
    """TGN-style event-memory baseline over causal local graph summaries.

    Memory is indexed by chain and updated only in chronological order.  The
    caller must reset memory between train/validation/test phases and must not
    shuffle event order.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 48, num_chains: int = 3):
        super().__init__()
        self.encoder = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        ))
        self.memory_cell = nn.GRUCell(hidden_dim + 1, hidden_dim)
        self.head = nn.Linear(hidden_dim * 2, 1)
        self.hidden_dim = hidden_dim
        self.num_chains = num_chains

    def initial_memory(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_chains, self.hidden_dim, device=device)

    def forward_event(
        self,
        data,
        normalized_time: torch.Tensor,
        memory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.encoder(data.x, data.edge_index))
        summary = global_mean_pool(hidden, data.batch)
        chain = data.chain_id.long().reshape(-1)
        previous = memory[chain]
        updated = self.memory_cell(torch.cat((summary, normalized_time.reshape(-1, 1)), dim=1), previous)
        logits = self.head(torch.cat((summary, previous), dim=1)).squeeze(-1)
        next_memory = memory.clone()
        next_memory[chain] = updated
        return logits, next_memory


class FraudSAGEBaseline(nn.Module):
    """Reproducible fraud-specific GraphSAGE baseline.

    The model combines class-balanced focal loss with neighbor aggregation. It
    is an equivalent fraud-oriented baseline, not a CARE-GNN reproduction,
    because GoG-SCIMain-v1 has no relation types required by CARE-GNN.
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 48, dropout: float = 0.3, num_chains: int = 3):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.dropout = dropout
        self.num_chains = num_chains
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_chains, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        hidden = F.dropout(F.relu(self.conv1(data.x, data.edge_index)), self.dropout, self.training)
        hidden = F.dropout(F.relu(self.conv2(hidden, data.edge_index)), self.dropout, self.training)
        pooled = torch.cat((global_mean_pool(hidden, data.batch), global_max_pool(hidden, data.batch)), dim=1)
        chain = F.one_hot(data.chain_id.long(), num_classes=self.num_chains).float()
        return self.head(torch.cat((pooled, chain), dim=1)).squeeze(-1)


def class_balanced_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    positive_weight: float,
    gamma: float = 2.0,
) -> torch.Tensor:
    labels = labels.float()
    base = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none",
        pos_weight=torch.as_tensor(positive_weight, device=logits.device),
    )
    probability = torch.sigmoid(logits)
    correct_probability = labels * probability + (1 - labels) * (1 - probability)
    return ((1 - correct_probability).pow(gamma) * base).mean()
