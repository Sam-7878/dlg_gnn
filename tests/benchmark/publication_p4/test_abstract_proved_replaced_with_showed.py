#!/usr/bin/env python3
"""
test_abstract_proved_replaced_with_showed.py

Round P4 Editorial Gate:
Verifies that:
1. 'local augmentation proved strongly dataset-dependent' is completely eliminated.
2. 'local augmentation showed strong dataset dependence' is present in both manuscript TeX sources.
"""

from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MDPI_TEX = REPO_ROOT / "docs/papers/_42_Benchmark/DLG-Benchmark.tex"
PREPRINTS_TEX = REPO_ROOT / "publication/benchmark/preprints/DLG-Benchmark-Preprint.tex"

BANNED_PHRASE = "local augmentation proved strongly dataset-dependent"
REQUIRED_PHRASE = "local augmentation showed strong dataset dependence"


def test_mdpi_abstract_phrase():
    assert MDPI_TEX.exists()
    content = MDPI_TEX.read_text(encoding="utf-8")
    assert BANNED_PHRASE not in content, f"Found '{BANNED_PHRASE}' in {MDPI_TEX}"
    assert REQUIRED_PHRASE in content, f"Missing '{REQUIRED_PHRASE}' in {MDPI_TEX}"


def test_preprints_abstract_phrase():
    assert PREPRINTS_TEX.exists()
    content = PREPRINTS_TEX.read_text(encoding="utf-8")
    assert BANNED_PHRASE not in content, f"Found '{BANNED_PHRASE}' in {PREPRINTS_TEX}"
    assert REQUIRED_PHRASE in content, f"Missing '{REQUIRED_PHRASE}' in {PREPRINTS_TEX}"
