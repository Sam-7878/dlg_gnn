"""
Manuscript claim and number linter.
Validates exact wording, detector counts, dataset counts, and frozen numerical anchors.
"""

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEX_FILE = REPO_ROOT / "dlg_gnn/docs/papers/_42_Benchmark/DLG-Benchmark.tex"

FORBIDDEN_TERMS = [
    ("DARPA", "DARPA must be 0 occurrences in final manuscript"),
    ("THEIA", "THEIA must be 0 occurrences in final manuscript"),
    ("TC-E5", "TC-E5 must be 0 occurrences in final manuscript"),
    ("provenance stress test", "provenance stress test must be 0 occurrences in final manuscript"),
    ("six detectors", "Use eight detector configurations: six established baselines and two DLG variants"),
    ("eight baselines", "Use eight detector configurations: six established baselines and two DLG variants"),
    ("DLG plus eight detectors", "Use eight detector configurations: six established baselines and two DLG variants"),
]

REQUIRED_NUMERICAL_ANCHORS = [
    ("71", "71 supported model-dataset pairs"),
    ("9", "9 unsupported/restricted pairs"),
    ("355", "355 successful runs"),
    ("1.71", "DLG-Aug ROC-AUC / PR-AUC average rank 1.71"),
    ("1.86", "DLG-Aug validation F1 average rank 1.86"),
    ("0.0350", "Elliptic PR-AUC delta +0.0350"),
    ("0.0656", "Reddit-Syn PR-AUC delta -0.0656"),
]


def audit_manuscript():
    if not TEX_FILE.exists():
        print(f"Error: {TEX_FILE} does not exist.")
        sys.exit(1)

    text = TEX_FILE.read_text(encoding="utf-8")
    errors = []

    # Check forbidden terms
    for term, reason in FORBIDDEN_TERMS:
        matches = list(re.finditer(re.escape(term), text, re.IGNORECASE))
        # Special check: case sensitive vs case insensitive
        exact_matches = [m for m in matches if m.group(0).lower() == term.lower()]
        if exact_matches:
            errors.append(f"[FORBIDDEN] Found {len(exact_matches)} occurrence(s) of '{term}': {reason}")

    # Check required numerical anchors
    for anchor, desc in REQUIRED_NUMERICAL_ANCHORS:
        if anchor not in text:
            errors.append(f"[MISSING ANCHOR] Required numeric anchor '{anchor}' ({desc}) not found in manuscript.")

    # Check Conclusion first sentence
    conclusion_match = re.search(r"\\section\{Conclusions\}[\s\S]*?\\label\{sec:conclusion\}[\s\S]*?\n\n(.*?)\n\n", text)
    if conclusion_match:
        first_sentence = conclusion_match.group(1).strip()
        if "eight detector configurations" not in first_sentence:
            errors.append(f"[CONCLUSION WORDING] Conclusion first sentence must state 'eight detector configurations...': got '{first_sentence[:100]}...'")

    if errors:
        print("Manuscript claim audit FAILED with errors:")
        for e in errors:
            print("  *", e)
        return False
    else:
        print("Manuscript claim audit PASSED all checks cleanly!")
        return True


if __name__ == "__main__":
    ok = audit_manuscript()
    sys.exit(0 if ok else 1)
