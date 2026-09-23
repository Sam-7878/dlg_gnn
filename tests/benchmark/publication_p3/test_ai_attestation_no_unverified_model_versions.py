#!/usr/bin/env python3
"""
test_ai_attestation_no_unverified_model_versions.py

Round P3 Gate: Verifies that the Author Attestation Matrix in
publication/benchmark/ai_use_author_attestation.md uses product-level
designations (OpenAI ChatGPT, Anthropic Claude, Google Gemini) and does
not fabricate or cite unverified specific model version numbers.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
ATTESTATION_MD = REPO_ROOT / "publication" / "benchmark" / "ai_use_author_attestation.md"

UNVERIFIED_VERSIONS = [
    "GPT-4 / GPT-4o",
    "GPT-4o",
    "Claude 3.5 Sonnet",
    "Gemini 1.5 Pro",
]

PRODUCT_LEVEL_TOOLS = [
    "OpenAI ChatGPT",
    "Anthropic Claude",
    "Google Gemini",
]


def test_attestation_has_no_unverified_model_versions():
    assert ATTESTATION_MD.exists(), f"Missing file: {ATTESTATION_MD}"
    content = ATTESTATION_MD.read_text(encoding="utf-8")

    for unv in UNVERIFIED_VERSIONS:
        assert unv not in content, (
            f"Unverified model version string '{unv}' found in {ATTESTATION_MD.name}"
        )


def test_attestation_uses_product_level_tools():
    content = ATTESTATION_MD.read_text(encoding="utf-8")

    for prod in PRODUCT_LEVEL_TOOLS:
        assert prod in content, (
            f"Product-level name '{prod}' not found in {ATTESTATION_MD.name}"
        )
