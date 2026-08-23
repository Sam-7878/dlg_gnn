#!/usr/bin/env python3
"""
Test: Verification that no synthetic fallback generator is used in the defense pipeline.
Defense Extension Round D3 hard gate requirement.
"""
import ast
import os
from pathlib import Path
import pytest

SCRIPTS_DIR = Path("scripts/defense_extension_real")

def test_no_synthetic_generators_in_real_pipeline():
    """Verify that real pipeline scripts do not contain mock or synthetic graph generators."""
    assert SCRIPTS_DIR.exists(), f"Scripts directory {SCRIPTS_DIR} must exist"
    
    python_files = list(SCRIPTS_DIR.glob("*.py"))
    assert len(python_files) > 0, "Real defense pipeline scripts must be present"

    forbidden_patterns = [
        "synthetic_graph_generator",
        "mock_graph_generator",
        "er_graph_generator",
        "barabasi_albert_generator",
    ]

    for py_file in python_files:
        content = py_file.read_text(encoding="utf-8")
        for pattern in forbidden_patterns:
            assert pattern not in content, (
                f"Forbidden synthetic generator pattern '{pattern}' found in {py_file}"
            )
