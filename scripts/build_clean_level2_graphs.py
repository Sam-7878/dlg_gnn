#!/usr/bin/env python3
"""A08 raw-only construction entrypoint. No graph cache or embedding fallback.

The approved config/specs and immutable output workspaces are managed by the
benchmark facade. Graph-level relation_builder targets remain unchanged.
"""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"projects/benchmark/scripts/a08_data_repair.py"),run_name="__main__")
