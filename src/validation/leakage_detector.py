"""
validation/leakage_detector.py

Detects potential label leakage in synthetic context datasets.

Checks:
    1. Direct literal leakage (label=1, is_fraud=true in text)
    2. Structural leakage (scenario_type selection correlated with label)
    3. Vocabulary imbalance (fraud-specific words concentrated in label==1)
    4. Shortcut test (TF-IDF + LR on context text — see TASK 5.4)
"""

import json
import os
import logging
from collections import Counter
from typing import List, Optional

logger = logging.getLogger(__name__)


class LeakageDetector:
    """
    Audits a synthetic context JSONL file for label leakage.

    Usage:
        detector = LeakageDetector(context_path)
        passed = detector.detect_leakage(report_md_path)
    """

    DIRECT_LEAK_PATTERNS = ["label=1", "is_fraud=true", "this_is_a_scam", "fraud=1"]
    FRAUD_VOCABULARY = [
        "scam", "phishing", "fraud", "steal", "hack", "malicious",
        "guaranteed return", "guaranteed profit", "send immediately",
    ]
    FRAUD_SCENARIO_TYPES = {
        "investment_scam", "romance_scam", "phishing_url_scam",
        "impersonation_scam", "urgent_transfer_request", "fake_customer_support",
        "crypto_wallet_migration_scam", "recovery_phrase_stealing_attempt",
        "high_yield_guaranteed_return_scam", "multi_stage_grooming_scam",
    }

    def __init__(self, context_path: str):
        self.context_path = context_path

    def _load_records(self) -> List[dict]:
        records = []
        with open(self.context_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def detect_leakage(self, report_md_path: str) -> bool:
        """
        Run all leakage checks and write a markdown report.

        Returns:
            True if ANY leakage is detected, False if dataset is clean.
        """
        logger.info(f"Running leakage audit on {self.context_path} ...")
        records = self._load_records()
        n = len(records)

        # ── Check 1: Direct literal leakage ───────────────────────────────
        check1_issues = []
        for rec in records:
            text = rec.get("context_text", "").lower()
            for pattern in self.DIRECT_LEAK_PATTERNS:
                if pattern in text:
                    check1_issues.append((rec.get("context_id"), pattern))

        # ── Check 2: Structural leakage — scenario_type vs label ──────────
        # If label==1 always maps to fraud scenario (no hard_positive),
        # the model receives label information via scenario_type.
        label_to_scenarios: dict = {0: Counter(), 1: Counter()}
        hard_positive_count = 0
        hard_negative_count = 0
        for rec in records:
            label = int(rec.get("label", -1))
            scenario = rec.get("scenario_type", "unknown")
            if label in (0, 1):
                label_to_scenarios[label][scenario] += 1
            if scenario == "hard_positive":
                hard_positive_count += 1
            if scenario == "hard_negative":
                hard_negative_count += 1

        # Check: fraud scenarios appear ONLY in label==1 (structural leakage)
        check2_issues = []
        if sum(label_to_scenarios.keys()) == -2:
            # No label field found (leakage-clean generation)
            check2_issues.append("NOTE: No 'label' field in records — label-free generation (clean).")
        else:
            fraud_in_benign = sum(
                label_to_scenarios[0].get(s, 0) for s in self.FRAUD_SCENARIO_TYPES
            )
            benign_in_fraud = sum(
                label_to_scenarios[1].get(s, 0)
                for s in ["benign"]
            )
            if fraud_in_benign == 0 and label_to_scenarios[0]:
                check2_issues.append(
                    "⚠️  STRUCTURAL LEAKAGE: No fraud-type scenarios in label==0 class "
                    "(hard_negative missing or zero). Shortcut risk: high."
                )
            if benign_in_fraud == 0 and label_to_scenarios[1]:
                check2_issues.append(
                    "⚠️  STRUCTURAL LEAKAGE: No benign-scenario entries in label==1 class "
                    "(hard_positive missing or zero). Shortcut risk: high."
                )

        # ── Check 3: Vocabulary imbalance ─────────────────────────────────
        vocab_counts = {0: Counter(), 1: Counter()}
        for rec in records:
            label = int(rec.get("label", -1))
            if label not in (0, 1):
                continue
            text = rec.get("context_text", "").lower()
            for word in self.FRAUD_VOCABULARY:
                if word in text:
                    vocab_counts[label][word] += 1

        vocab_issues = []
        n0 = max(sum(label_to_scenarios[0].values()), 1)
        n1 = max(sum(label_to_scenarios[1].values()), 1)
        for word in self.FRAUD_VOCABULARY:
            rate0 = vocab_counts[0][word] / n0
            rate1 = vocab_counts[1][word] / n1
            if rate1 > 0.5 and rate0 < 0.02:
                vocab_issues.append(
                    f"  '{word}': fraud rate={rate1:.3f}, benign rate={rate0:.3f} — "
                    f"near-perfect discriminator"
                )

        # ── Determine overall status ───────────────────────────────────────
        leakage_detected = bool(check1_issues or check2_issues or vocab_issues)

        # ── Write report ───────────────────────────────────────────────────
        os.makedirs(os.path.dirname(os.path.abspath(report_md_path)), exist_ok=True)
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write("# Dataset Leakage Audit Report\n\n")
            f.write(f"**Dataset**: `{self.context_path}`  \n")
            f.write(f"**Total Records**: {n}  \n")
            f.write(f"**Hard Positive count**: {hard_positive_count}  \n")
            f.write(f"**Hard Negative count**: {hard_negative_count}  \n\n")
            f.write(f"**Status**: {'⚠️ LEAKAGE DETECTED' if leakage_detected else '✅ CLEAN'}  \n\n")
            f.write("---\n\n")

            f.write("## Check 1: Direct Literal Leakage\n\n")
            if check1_issues:
                f.write(f"**FAIL** — {len(check1_issues)} instances found:\n\n")
                for cid, pat in check1_issues[:10]:
                    f.write(f"- `{cid}`: pattern `{pat}`\n")
            else:
                f.write("**PASS** — No direct literal leakage detected.\n")
            f.write("\n")

            f.write("## Check 2: Structural Leakage (scenario_type vs label)\n\n")
            f.write(f"Label==0 scenario distribution: {dict(label_to_scenarios[0].most_common(5))}  \n")
            f.write(f"Label==1 scenario distribution: {dict(label_to_scenarios[1].most_common(5))}  \n\n")
            if check2_issues:
                for issue in check2_issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("**PASS** — Hard positives and hard negatives present. No structural leakage.\n")
            f.write("\n")

            f.write("## Check 3: Fraud Vocabulary Imbalance\n\n")
            if vocab_issues:
                f.write("**WARNING** — Following words are near-perfect label discriminators:\n\n")
                for vi in vocab_issues:
                    f.write(f"{vi}\n")
            else:
                f.write("**PASS** — No extreme vocabulary imbalance detected.\n")
            f.write("\n")

            f.write("## Check 4: Shortcut Test (TF-IDF + LR)\n\n")
            f.write("Run separately via `experiments/run_leakage.py`.\n")
            f.write("Acceptance criterion: text-only TF-IDF+LR accuracy < 0.85\n\n")

            f.write("---\n\n")
            f.write("*Generated by `src/validation/leakage_detector.py`*\n")

        logger.info(f"Leakage audit report saved to {report_md_path}")
        return leakage_detected
