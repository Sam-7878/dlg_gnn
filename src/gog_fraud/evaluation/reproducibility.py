"""Reproducibility helpers shared by SCI benchmark runners."""
from __future__ import annotations

import os
import random
import warnings
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedState:
    seed: int
    deterministic_requested: bool
    deterministic_enabled: bool
    cuda_seeded: bool
    cublas_workspace_config: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, PyTorch CPU/CUDA and request deterministic kernels.

    ``warn_only=True`` keeps a long benchmark running when PyTorch encounters an
    operation without a deterministic implementation while making the limitation
    visible in the experiment log.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_seeded = bool(torch.cuda.is_available())
    if cuda_seeded:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    enabled = False
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            enabled = bool(torch.are_deterministic_algorithms_enabled())
        except (AttributeError, RuntimeError) as exc:
            warnings.warn(f"deterministic PyTorch mode could not be enabled: {exc}")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    return SeedState(
        seed=seed,
        deterministic_requested=bool(deterministic),
        deterministic_enabled=enabled,
        cuda_seeded=cuda_seeded,
        cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    )


def make_torch_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator

