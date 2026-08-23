"""Evaluation eligibility rules for labeled defense extensions.

The policy is deliberately independent of detector implementations so that
single-class outputs cannot silently enter derived performance statistics.
"""

from __future__ import annotations

from typing import Any, Mapping


THEIA_MIN_POSITIVES = 10
THEIA_MIN_SPLIT_POSITIVES = 2
THEIA_RECOMMENDED_POSITIVES = 20


def theia_performance_eligibility(
    positive_nodes: int,
    validation_positives: int,
    test_positives: int,
) -> dict[str, Any]:
    """Return the fail-closed THEIA node-level performance decision."""

    eligible = (
        positive_nodes >= THEIA_MIN_POSITIVES
        and validation_positives >= THEIA_MIN_SPLIT_POSITIVES
        and test_positives >= THEIA_MIN_SPLIT_POSITIVES
    )
    return {
        "positive_nodes": int(positive_nodes),
        "validation_positives": int(validation_positives),
        "test_positives": int(test_positives),
        "minimum_positive_nodes": THEIA_MIN_POSITIVES,
        "recommended_positive_nodes": THEIA_RECOMMENDED_POSITIVES,
        "minimum_validation_positives": THEIA_MIN_SPLIT_POSITIVES,
        "minimum_test_positives": THEIA_MIN_SPLIT_POSITIVES,
        "performance_eligible": eligible,
        "evaluation_role": "performance_and_scalability" if eligible else "scalability_only",
        "performance_metrics_valid": eligible,
        "metric_status": "defined" if eligible else "undefined_single_class",
    }


def record_is_performance_eligible(record: Mapping[str, Any]) -> bool:
    """Whether an existing benchmark record can enter performance statistics."""

    if record.get("status") != "success":
        return False
    try:
        return int(float(record.get("n_test_positives", 0) or 0)) > 0
    except (TypeError, ValueError):
        return False


def undefined_single_class_metrics() -> dict[str, Any]:
    """Canonical representation for a test split without both classes."""

    return {
        "roc_auc": None,
        "pr_auc": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "mcc": None,
        "balanced_accuracy": None,
        "metric_status": "undefined_single_class",
        "performance_eligible": False,
    }
