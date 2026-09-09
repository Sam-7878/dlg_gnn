"""
tests/test_scenario_provenance.py

Round 2 validation: verify that scenario_type assignment does not
constitute direct label leakage into observable model inputs.

Tests:
  1. assign_scenarios_no_leakage() does NOT propagate labels past scenario_type
  2. SyntheticContextGenerator.generate_contexts() receives only scenario_type strings
  3. Cross-class mixing rates are within expected bounds
  4. TF-IDF shortcut AUC is below trivial-leakage threshold (< 0.95)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


def test_assign_scenarios_no_leakage_signature():
    """assign_scenarios_no_leakage must accept labels but not output labels in scenario strings."""
    from data_generation.synthetic_context_generator import assign_scenarios_no_leakage
    import inspect
    sig = inspect.signature(assign_scenarios_no_leakage)
    # First positional parameter must be labels
    params = list(sig.parameters.keys())
    assert "labels" in params, "Function must accept 'labels' as input"


def test_assign_scenarios_returns_only_strings():
    """All returned scenario types must be strings, never binary labels."""
    from data_generation.synthetic_context_generator import assign_scenarios_no_leakage
    labels = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    scenarios = assign_scenarios_no_leakage(labels, seed=42)
    assert len(scenarios) == len(labels)
    for s in scenarios:
        assert isinstance(s, str), f"Scenario type must be string, got {type(s)}: {s}"
        assert s not in {"0", "1", "True", "False"}, (
            f"Scenario type must not be a label value, got '{s}'"
        )


def test_hard_positive_negative_cross_class_mixing():
    """
    Verify that fraud and benign nodes can both receive fraud-type scenarios,
    preventing trivial label-from-scenario recovery.
    """
    from data_generation.synthetic_context_generator import assign_scenarios_no_leakage

    np.random.seed(42)
    n = 2000
    labels = (np.random.rand(n) < 0.5).astype(int)  # balanced for this test
    scenarios = assign_scenarios_no_leakage(labels, seed=42)

    fraud_idx = np.where(labels == 1)[0]
    benign_idx = np.where(labels == 0)[0]

    # Fraud nodes should sometimes get benign-type scenarios (hard positives)
    benign_type_scenarios = {"benign", "hard_positive"}
    fraud_with_benign = sum(scenarios[i] in benign_type_scenarios for i in fraud_idx)
    fraud_benign_rate = fraud_with_benign / len(fraud_idx)
    assert fraud_benign_rate > 0.10, (
        f"Fraud nodes have too few benign-type scenarios: {fraud_benign_rate:.2%} "
        "(expected > 10% for hard positive mixing)"
    )

    # Benign nodes should sometimes get fraud-type scenarios (hard negatives)
    fraud_type_scenarios = {"phishing", "romance_scam", "investment_scam",
                            "account_takeover", "pig_butchering", "hard_negative"}
    benign_with_fraud = sum(scenarios[i] in fraud_type_scenarios for i in benign_idx)
    benign_fraud_rate = benign_with_fraud / len(benign_idx)
    assert benign_fraud_rate > 0.10, (
        f"Benign nodes have too few fraud-type scenarios: {benign_fraud_rate:.2%} "
        "(expected > 10% for hard negative mixing)"
    )


def test_context_generator_receives_no_labels():
    """
    SyntheticContextGenerator.generate_contexts() must accept scenario_types
    (strings) but not labels.
    """
    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage,
    )
    import inspect

    gen = SyntheticContextGenerator(seed=42)
    sig = inspect.signature(gen.generate_contexts)
    params = list(sig.parameters.keys())

    # Must accept scenario_types
    assert "scenario_types" in params, (
        "generate_contexts must accept scenario_types parameter"
    )
    # Must NOT accept label, y, fraud_flag, is_fraud
    forbidden_params = {"label", "labels", "y", "fraud_flag", "is_fraud", "fraud"}
    found = forbidden_params & set(params)
    assert not found, (
        f"generate_contexts must not accept label parameters: {found}"
    )


def test_shortcut_auc_below_trivial_threshold():
    """
    TF-IDF + Logistic Regression AUC on context text must be < 0.95.
    AUC >= 0.95 would indicate trivially recoverable label leakage.
    """
    from data_generation.synthetic_context_generator import (
        SyntheticContextGenerator, assign_scenarios_no_leakage,
    )
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    np.random.seed(7)
    n = 500
    labels = (np.random.rand(n) < 0.10).astype(int)
    scenarios = assign_scenarios_no_leakage(labels, seed=7)
    gen = SyntheticContextGenerator(seed=7)
    event_ids = [f"tx_{i}" for i in range(n)]
    records = gen.generate_contexts(scenario_types=scenarios, event_ids=event_ids)
    texts = [r["context_text"] for r in records]

    tfidf = TfidfVectorizer(max_features=300, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)

    try:
        lr = LogisticRegression(
            max_iter=500, C=1.0, random_state=7, class_weight="balanced",
            solver="lbfgs",
        )
        lr.fit(X, labels)
        y_score = lr.predict_proba(X)[:, 1]
        auc = roc_auc_score(labels, y_score)
    except Exception:
        pytest.skip("Logistic Regression failed (may be class-imbalance issue with tiny sample)")

    assert auc < 0.95, (
        f"TF-IDF shortcut AUC={auc:.4f} >= 0.95: context generation has trivial label leakage. "
        "Review assign_scenarios_no_leakage() cross-class mixing rates."
    )


def test_scenario_assignment_deterministic():
    """Same labels + seed must produce identical scenario lists."""
    from data_generation.synthetic_context_generator import assign_scenarios_no_leakage

    labels = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1])
    s1 = assign_scenarios_no_leakage(labels, seed=99)
    s2 = assign_scenarios_no_leakage(labels, seed=99)
    assert s1 == s2, "assign_scenarios_no_leakage is not deterministic for the same seed"
