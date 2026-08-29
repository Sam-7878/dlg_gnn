"""Causal local-transaction GNN used by the Round 4 SCI main track."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool


class CausalLocalGIN(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 48, dropout: float = 0.3, num_chains: int = 3):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GINConv(nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        ))
        self.conv2 = GINConv(nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        ))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_chains, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )
        self.num_chains = num_chains

    def forward(self, data):
        h = F.dropout(F.relu(self.conv1(data.x, data.edge_index)), self.dropout, self.training)
        h = F.dropout(F.relu(self.conv2(h, data.edge_index)), self.dropout, self.training)
        pooled = torch.cat((global_mean_pool(h, data.batch), global_max_pool(h, data.batch)), dim=1)
        chain = F.one_hot(data.chain_index.long(), num_classes=self.num_chains).float()
        return self.head(torch.cat((pooled, chain), dim=1)).squeeze(-1)

    def forward_mc(self, data, passes: int = 10):
        self.train()
        predictions = []
        with torch.no_grad():
            for _ in range(passes):
                predictions.append(torch.sigmoid(self(data)))
        samples = torch.stack(predictions)
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0, unbiased=False)
        eps = 1e-8
        entropy = -(mean * torch.log(mean + eps) + (1 - mean) * torch.log(1 - mean + eps))
        self.eval()
        return samples, mean, variance, entropy
