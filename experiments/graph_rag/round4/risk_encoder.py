"""Small auxiliary RiskEncoder trained only on label-independent observable context."""
import torch.nn as nn


class ObservableRiskEncoder(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 24, dropout: float = 0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)
