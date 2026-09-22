"""Canonical artifact locations for GraphRAG Round 3.

Round-specific outputs live below their domain/round directory so rerunning an
experiment does not repopulate loose files at the repository root.
"""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ROUND3_RESULTS = ROOT / "results" / "graphrag" / "round_3"
ROUND3_REPORTS = ROOT / "reports" / "graphrag" / "round_3"
ROUND3_FIGURES = ROOT / "figures" / "graphrag" / "round_3"

CHECKPOINT_DIR = ROUND3_RESULTS / "real_checkpoints"
CHECKPOINT_MANIFEST_DIR = ROUND3_RESULTS / "checkpoint_manifests"
RAW_PREDICTION_DIR = ROUND3_RESULTS / "real_raw_predictions"
TABLE_DIR = ROUND3_RESULTS / "tables"


def ensure_round3_artifact_dirs() -> None:
    for path in (
        ROUND3_RESULTS,
        ROUND3_REPORTS,
        ROUND3_FIGURES,
        CHECKPOINT_DIR,
        CHECKPOINT_MANIFEST_DIR,
        RAW_PREDICTION_DIR,
        TABLE_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
