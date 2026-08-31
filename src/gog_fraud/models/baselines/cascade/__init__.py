"""Strong fast-path to relational-stage cascade controls."""

from .tabular_l2_cascade import apply_budgeted_cascade, ambiguity_cutoff

__all__ = ["apply_budgeted_cascade", "ambiguity_cutoff"]
