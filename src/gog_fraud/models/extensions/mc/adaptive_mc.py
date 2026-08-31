# src/gog_fraud/models/extensions/mc/adaptive_mc.py
"""
Adaptive Monte Carlo Dropout with Online Convergence Stopping.
Reduces redundant stochastic passes when predictive mean and variance stabilize early.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AdaptiveMCOutput:
    mean: torch.Tensor
    variance: torch.Tensor
    t_effective: float
    early_stopped_fraction: float
    total_passes: int


class AdaptiveMCEngine:
    """
    Adaptive MC Dropout evaluator using Welford's online algorithm.
    Stops sampling early if running mean and variance converge within tolerance.
    """

    def __init__(
        self,
        t_min: int = 3,
        t_max: int = 8,
        mean_tol: float = 0.01,
        var_tol: float = 0.005,
    ) -> None:
        if t_min < 2:
            raise ValueError("t_min must be at least 2 for variance estimation.")
        if t_max < t_min:
            raise ValueError("t_max must be >= t_min.")
        self.t_min = t_min
        self.t_max = t_max
        self.mean_tol = mean_tol
        self.var_tol = var_tol

    def forward_pass(
        self,
        model_fn: Callable[[], torch.Tensor],
    ) -> AdaptiveMCOutput:
        """
        Executes stochastic forward passes online.
        model_fn: Callable returning a 1D or 2D score Tensor of shape (N,)
        """
        # Pass 1
        x1 = model_fn()
        n = x1.shape[0]
        device = x1.device

        mean = x1.clone()
        m2 = torch.zeros_like(x1)
        prev_var = torch.zeros_like(x1)

        # Track active masks for early stopping if evaluated per-sample
        active = torch.ones(n, dtype=torch.bool, device=device)
        stopped_at = torch.full((n,), self.t_max, dtype=torch.float32, device=device)

        passes = 1
        for t in range(2, self.t_max + 1):
            xt = model_fn()
            passes = t
            
            # Welford update for online mean and M2 (variance numerator)
            delta = xt - mean
            mean = mean + delta / t
            delta2 = xt - mean
            m2 = m2 + delta * delta2
            current_var = m2 / (t - 1)

            if t >= self.t_min:
                mean_change = torch.abs(delta / t)
                var_change = torch.abs(current_var - prev_var)
                
                # Check convergence
                converged = (mean_change < self.mean_tol) & (var_change < self.var_tol)
                just_stopped = converged & active
                stopped_at[just_stopped] = float(t)
                active[just_stopped] = False

                if not active.any():
                    break

            prev_var = current_var

        final_var = m2 / max(passes - 1, 1)
        t_effective = float(stopped_at.mean().item())
        early_stopped_fraction = float((stopped_at < self.t_max).float().mean().item())

        return AdaptiveMCOutput(
            mean=mean,
            variance=final_var,
            t_effective=t_effective,
            early_stopped_fraction=early_stopped_fraction,
            total_passes=passes,
        )
