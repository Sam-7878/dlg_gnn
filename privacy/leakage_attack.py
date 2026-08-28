"""
privacy/leakage_attack.py

Attribute inference attack on risk vector representations.

Threat model:
    Attacker can observe the transmitted risk representation (r_t or variants)
    but NOT the raw context text. The attacker attempts to infer sensitive
    attributes from the representation alone.

Sensitive attributes:
    scam_category  : k_t — the fraud category (multi-class)
    urgency_state  : binary — whether the context has urgency cues
    intent         : binary — steal_funds vs. other intent

Attack model:
    Logistic Regression trained on the representation features.
    Intentionally weak (no deep model) — the goal is to measure INFORMATION
    leakage, not achieve SOTA attack accuracy.

Output (Privacy-Utility Table row):
    representation_name : str
    bytes               : int
    attack_accuracy     : float   (fraction of correctly inferred labels)
    attack_macro_f1     : float
    attack_auc          : float   (where applicable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    representation_name: str
    target_attribute: str
    bytes_per_sample: float
    accuracy: float
    macro_f1: float
    auc: float
    n_samples: int
    n_classes: int
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "representation": self.representation_name,
            "attribute": self.target_attribute,
            "bytes": round(self.bytes_per_sample, 1),
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "auc": round(self.auc, 4),
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "notes": self.notes,
        }


class LeakageAttack:
    """
    Attribute inference attack via Logistic Regression.

    Usage:
        attack = LeakageAttack()
        result = attack.run(
            features=feature_matrix,     # [N, D] numpy array (the risk representation)
            targets=target_labels,        # [N] int array (the sensitive attribute)
            representation_name="full_vector",
            target_attribute="scam_category",
            bytes_per_sample=14.0,
        )
    """

    def __init__(self, max_iter: int = 500, cv_folds: int = 5, random_state: int = 42):
        self.max_iter = max_iter
        self.cv_folds = cv_folds
        self.random_state = random_state

    def _build_features(self, risk_dicts: List[Dict[str, Any]]) -> np.ndarray:
        """Convert list of risk dicts to a numpy feature matrix [N, 5]."""
        rows = []
        for d in risk_dicts:
            rows.append([
                float(d.get("local_risk_score", 0.0)),
                float(d.get("confidence", 0.0)),
                float(d.get("risk_type_id", 0)),
                float(d.get("context_age_sec", 0)),
                float(d.get("relation_hint_id", 0)),
            ])
        return np.array(rows, dtype=np.float32)

    def run(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        representation_name: str,
        target_attribute: str,
        bytes_per_sample: float,
    ) -> AttackResult:
        """
        Run an attribute inference attack.

        Args:
            features            : [N, D] feature matrix
            targets             : [N] integer class labels
            representation_name : e.g. "full_vector", "quantized", "noisy", "minimal"
            target_attribute    : e.g. "scam_category", "urgency_state"
            bytes_per_sample    : actual serialized byte count for this representation
        """
        n_samples = len(targets)
        n_classes = int(np.unique(targets).size)

        if n_samples < self.cv_folds * 2:
            logger.warning(f"Too few samples ({n_samples}) for {self.cv_folds}-fold CV. Skipping.")
            return AttackResult(
                representation_name=representation_name,
                target_attribute=target_attribute,
                bytes_per_sample=bytes_per_sample,
                accuracy=float("nan"), macro_f1=float("nan"), auc=float("nan"),
                n_samples=n_samples, n_classes=n_classes,
                notes="insufficient_samples",
            )

        clf = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            C=1.0,
        )

        # Cross-validated accuracy
        cv_acc = cross_val_score(
            clf, features, targets,
            cv=self.cv_folds, scoring="accuracy",
        )
        accuracy = float(cv_acc.mean())

        # Cross-validated macro F1
        cv_f1 = cross_val_score(
            clf, features, targets,
            cv=self.cv_folds, scoring="f1_macro",
        )
        macro_f1 = float(cv_f1.mean())

        # AUC (binary or OvR for multiclass)
        try:
            if n_classes == 2:
                cv_auc = cross_val_score(
                    clf, features, targets,
                    cv=self.cv_folds, scoring="roc_auc",
                )
            else:
                cv_auc = cross_val_score(
                    clf, features, targets,
                    cv=self.cv_folds, scoring="roc_auc_ovr_weighted",
                )
            auc = float(cv_auc.mean())
        except Exception as e:
            logger.debug(f"AUC computation failed: {e}")
            auc = float("nan")

        logger.info(
            f"[LeakageAttack] {representation_name} → {target_attribute}: "
            f"Acc={accuracy:.4f} F1={macro_f1:.4f} AUC={auc:.4f} ({bytes_per_sample:.1f}B)"
        )

        return AttackResult(
            representation_name=representation_name,
            target_attribute=target_attribute,
            bytes_per_sample=bytes_per_sample,
            accuracy=accuracy,
            macro_f1=macro_f1,
            auc=auc,
            n_samples=n_samples,
            n_classes=n_classes,
        )

    def run_from_dicts(
        self,
        risk_dicts: List[Dict[str, Any]],
        targets: np.ndarray,
        representation_name: str,
        target_attribute: str,
        codec=None,  # VectorCodec instance for byte measurement
    ) -> AttackResult:
        """Convenience: build feature matrix from risk_dicts and run attack."""
        features = self._build_features(risk_dicts)
        if codec is not None:
            bytes_list = [codec.measure_bytes(d) for d in risk_dicts]
            bytes_per_sample = float(np.mean(bytes_list))
        else:
            bytes_per_sample = float(features.shape[1] * 4)  # default: D×float32
        return self.run(features, targets, representation_name, target_attribute, bytes_per_sample)

    def run_shortcut_test(
        self,
        context_texts: List[str],
        labels: np.ndarray,
        representation_name: str = "context_text_tfidf",
    ) -> AttackResult:
        """
        TASK 5.4: Text-only trivial shortcut test.
        Trains TF-IDF + LR on raw context text to detect label leakage.

        If this test returns near-perfect accuracy, the synthetic context
        directly encodes the fraud label (dataset construction flaw).
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline

        n_samples = len(labels)
        bytes_per_sample = float(
            np.mean([len(t.encode("utf-8")) for t in context_texts])
        )

        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
            ("clf", LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)),
        ])

        cv_acc = cross_val_score(pipe, context_texts, labels, cv=self.cv_folds, scoring="accuracy")
        cv_f1  = cross_val_score(pipe, context_texts, labels, cv=self.cv_folds, scoring="f1_macro")
        try:
            cv_auc = cross_val_score(pipe, context_texts, labels, cv=self.cv_folds, scoring="roc_auc")
            auc = float(cv_auc.mean())
        except Exception:
            auc = float("nan")

        accuracy = float(cv_acc.mean())
        macro_f1 = float(cv_f1.mean())

        logger.info(
            f"[ShortcutTest] TF-IDF+LR on context text: "
            f"Acc={accuracy:.4f} F1={macro_f1:.4f} AUC={auc:.4f}"
        )
        # For imbalanced datasets (e.g. 10% fraud), trivial majority classifier has Acc=0.90.
        # Thus, AUC > 0.85 or macro_f1 > 0.80 is the proper criterion for near-perfect shortcut.
        is_shortcut = (auc > 0.85) if not np.isnan(auc) else (macro_f1 > 0.80)
        if is_shortcut:
            logger.warning(
                "⚠️  SHORTCUT DETECTED: text-only classifier achieves AUC > 0.85 (near-perfect). "
                "The synthetic context may directly encode fraud labels. "
                "Review dataset generation protocol."
            )
        else:
            logger.info("  ✅ PASS: No trivial text shortcut detected (AUC <= 0.85).")

        return AttackResult(
            representation_name=representation_name,
            target_attribute="fraud_label",
            bytes_per_sample=bytes_per_sample,
            accuracy=accuracy,
            macro_f1=macro_f1,
            auc=auc,
            n_samples=n_samples,
            n_classes=2,
            notes="shortcut_test_tfidf_lr",
        )
