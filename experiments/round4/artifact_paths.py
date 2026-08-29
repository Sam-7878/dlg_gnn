"""Canonical Round 4 artifact locations."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "data" / "benchmark" / "gog_scimain_v1"
RESULTS_DIR = ROOT / "results" / "graphrag" / "round_4"
REPORTS_DIR = ROOT / "reports" / "graphrag" / "round_4"
FIGURES_DIR = ROOT / "figures" / "graphrag" / "round_4"
TABLES_DIR = ROOT / "tables" / "graphrag" / "round_4"
CHECKPOINT_DIR = RESULTS_DIR / "real_checkpoints"
CHECKPOINT_MANIFEST_DIR = RESULTS_DIR / "checkpoint_manifests"
RISK_CHECKPOINT_DIR = RESULTS_DIR / "risk_encoder_checkpoints"
RAW_PREDICTION_DIR = RESULTS_DIR / "raw_predictions"


def ensure_dirs() -> None:
    for path in (
        RESULTS_DIR, REPORTS_DIR, FIGURES_DIR, TABLES_DIR, CHECKPOINT_DIR,
        CHECKPOINT_MANIFEST_DIR, RISK_CHECKPOINT_DIR, RAW_PREDICTION_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
