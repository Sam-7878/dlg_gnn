from dataclasses import dataclass
from typing import Literal

@dataclass
class MCDropoutConfig:
    mc_samples: int = 8
    dropout_p: float = 0.10
    execution_mode: Literal["sequential", "batched", "auto"] = "sequential"
    parallel_chunk_size: int = 2
    keep_raw_scores: bool = False
    inject_into_aux: bool = False
    seed: int = 42

    def __post_init__(self):
        if self.mc_samples < 1:
            raise ValueError("mc_samples must be at least 1")
        if not 0.0 <= self.dropout_p < 1.0:
            raise ValueError("dropout_p must be within [0, 1)")
        if self.parallel_chunk_size < 1:
            raise ValueError("parallel_chunk_size must be positive")
