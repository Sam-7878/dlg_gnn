"""Fail-closed non-finite diagnostics; values are never repaired."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class NonFiniteDiagnostic:
    first_nonfinite_stage: str
    dataset: str | None
    model: str | None
    partition_id: int | None
    node_range: str | None
    tensor_shape: tuple[int, ...]
    nan_count: int
    inf_count: int
    tensor_min: float | None
    tensor_max: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NonFiniteTensorError(FloatingPointError):
    def __init__(self, diagnostic: NonFiniteDiagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"non-finite tensor at {diagnostic.first_nonfinite_stage}: {diagnostic.to_dict()}")


def assert_finite_tensor(value: Any, *, stage: str, dataset: str | None = None,
                         model: str | None = None, partition_id: int | None = None,
                         node_range: str | None = None) -> None:
    tensor = value.detach().cpu() if torch.is_tensor(value) else torch.as_tensor(np.asarray(value))
    finite = torch.isfinite(tensor)
    if bool(finite.all()):
        return
    valid = tensor[finite]
    diagnostic = NonFiniteDiagnostic(
        first_nonfinite_stage=stage, dataset=dataset, model=model, partition_id=partition_id,
        node_range=node_range, tensor_shape=tuple(tensor.shape), nan_count=int(torch.isnan(tensor).sum()),
        inf_count=int(torch.isinf(tensor).sum()), tensor_min=float(valid.min()) if valid.numel() else None,
        tensor_max=float(valid.max()) if valid.numel() else None)
    raise NonFiniteTensorError(diagnostic)
