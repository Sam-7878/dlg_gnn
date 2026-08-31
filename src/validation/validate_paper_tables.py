#!/usr/bin/env python3
"""Completeness gate for generated paper tables and threshold provenance."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from validation.sci_v3_final_common import atomic_json


BAD = re.compile(r"(^|\W)(--|NA|N/A|TODO|placeholder|CSV reference only)($|\W)", re.IGNORECASE)


def run(canonical_dir: Path, table_dir: Path, report: Path) -> dict:
    errors: list[str] = []
    calibration = pd.read_csv(canonical_dir / "canonical_calibration.csv")
    metrics = pd.read_csv(canonical_dir / "canonical_metrics.csv")
    required_calibration = {"nll", "brier", "ece10", "ece20", "adaptive_ece", "classwise_ece"}
    missing = required_calibration - set(calibration.columns)
    if missing:
        errors.append(f"missing calibration columns: {sorted(missing)}")
    for column in required_calibration & set(calibration.columns):
        if calibration[column].isna().any():
            errors.append(f"undefined calibration cells: {column}")
    for chain in ("ethereum", "bsc", "polygon", "pooled"):
        for method in calibration.method.unique():
            seeds = set(calibration[(calibration.chain == chain) & (calibration.method == method)].seed)
            if seeds != {11, 22, 33, 44, 55}:
                errors.append(f"calibration seed coverage {chain}/{method}: {sorted(seeds)}")
    selective = metrics[metrics.evidence_family == "selective_risk"] if "evidence_family" in metrics else pd.DataFrame()
    for column in ("aurc", "e_aurc"):
        if column not in selective or selective[column].isna().any():
            errors.append(f"selective-risk metric incomplete: {column}")
    for path in sorted(table_dir.glob("*")):
        if path.suffix.lower() not in {".csv", ".tex"}:
            continue
        text = path.read_text(encoding="utf-8")
        if BAD.search(text):
            errors.append(f"placeholder token in {path}")
    payload = {"status": "PASS" if not errors else "FAIL", "errors": errors}
    atomic_json(report, payload)
    if errors:
        raise ValueError("; ".join(errors))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-dir", default="results/sci_v3_final/canonical")
    parser.add_argument("--table-dir", default="results/sci_v3_final/figures_and_tables/tables")
    parser.add_argument("--report", default="results/sci_v3_final/paper_table_validation.json")
    args = parser.parse_args()
    try:
        payload = run(Path(args.canonical_dir), Path(args.table_dir), Path(args.report))
    except ValueError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)})); return 1
    print(json.dumps(payload)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
