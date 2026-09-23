"""Detector score contracts and validation-only calibration utilities."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class ScoreSemantics:
    paper_name: str
    score_type: str
    score_definition: str
    higher_is_more_anomalous: bool
    probability_like: bool = False
    normalized: bool = False
    postprocessing: str = "none"

    @property
    def fixed_05_applicable(self) -> bool:
        return self.probability_like

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"fixed_05_applicable": self.fixed_05_applicable}


SCORE_SEMANTICS = {
    "DOMINANT": ScoreSemantics("DOMINANT", "unbounded_reconstruction_error", "weighted L2 attribute and adjacency reconstruction error", True),
    "AnomalyDAE": ScoreSemantics("AnomalyDAE", "unbounded_reconstruction_error", "weighted attribute and structure reconstruction error with positive-entry weighting", True),
    "CoLA": ScoreSemantics("CoLA", "unbounded_contrastive_logit_difference", "negative-subgraph logit minus positive-subgraph logit", True),
    "CONAD": ScoreSemantics("CONAD", "unbounded_reconstruction_error", "contrastively trained attribute and structure reconstruction error", True),
    "GADNR": ScoreSemantics("GADNR", "unbounded_neighborhood_reconstruction_loss", "weighted neighborhood, degree, and feature reconstruction loss", True),
    "OCGNN": ScoreSemantics("OCGNN", "unbounded_one_class_distance", "one-class hypersphere deviation score", True),
    "DLG-Base": ScoreSemantics("DLG-Base", "unbounded_reconstruction_error", "weighted attribute and adjacency reconstruction error from same-graph local/global GCN stack", True),
    "DLG": ScoreSemantics("DLG", "unbounded_reconstruction_error", "weighted original-feature and adjacency reconstruction error after frozen local-embedding augmentation", True),
}


def get_score_semantics(model: str) -> ScoreSemantics:
    try:
        return SCORE_SEMANTICS[model]
    except KeyError as exc:
        raise ValueError(f"score semantics are not registered for model: {model}") from exc


def audit_score_orientation(y_true: Any, score: Any, *, expected_higher: bool = True) -> dict[str, Any]:
    """Compare raw and inverted ranking without silently changing orientation."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    values = np.asarray(score, dtype=float).reshape(-1)
    if len(y) != len(values) or len(np.unique(y)) != 2:
        raise ValueError("binary labels and equally sized scores are required")
    raw_auc = float(roc_auc_score(y, values))
    inverted_auc = float(roc_auc_score(y, -values))
    return {
        "expected_higher_is_more_anomalous": bool(expected_higher),
        "raw_roc_auc": raw_auc,
        "inverted_roc_auc": inverted_auc,
        "orientation_warning": bool(expected_higher and inverted_auc > raw_auc + 0.10),
        "orientation_action": "report_only_no_silent_inversion",
    }


class ValidationCalibrator:
    """Fit score transformations using validation observations only."""

    def __init__(self, method: str = "robust_percentile") -> None:
        self.method = method
        self.parameters: dict[str, float] = {}
        self._model: LogisticRegression | None = None

    def fit(self, validation_score: Any, validation_y: Any | None = None) -> "ValidationCalibrator":
        score = np.asarray(validation_score, dtype=float).reshape(-1)
        if score.size == 0 or not np.isfinite(score).all():
            raise ValueError("finite validation scores are required")
        if self.method == "minmax":
            self.parameters = {"low": float(score.min()), "high": float(score.max())}
        elif self.method == "robust_percentile":
            self.parameters = {"low": float(np.quantile(score, 0.01)), "high": float(np.quantile(score, 0.99))}
        elif self.method == "platt":
            if validation_y is None:
                raise ValueError("Platt calibration requires validation labels")
            y = np.asarray(validation_y, dtype=np.int64).reshape(-1)
            if len(y) != len(score) or len(np.unique(y)) != 2:
                raise ValueError("Platt calibration requires aligned binary validation labels")
            self._model = LogisticRegression(random_state=0).fit(score[:, None], y)
            self.parameters = {"coefficient": float(self._model.coef_[0, 0]), "intercept": float(self._model.intercept_[0])}
        else:
            raise ValueError(f"unsupported calibration method: {self.method}")
        return self

    def transform(self, score: Any) -> np.ndarray:
        values = np.asarray(score, dtype=float).reshape(-1)
        if self.method == "platt":
            if self._model is None: raise RuntimeError("calibrator is not fitted")
            return self._model.predict_proba(values[:, None])[:, 1]
        if not self.parameters: raise RuntimeError("calibrator is not fitted")
        low, high = self.parameters["low"], self.parameters["high"]
        if high <= low: return np.zeros_like(values)
        return np.clip((values - low) / (high - low), 0.0, 1.0)

