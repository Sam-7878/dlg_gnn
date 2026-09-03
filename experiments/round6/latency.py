"""Audit and accurately label the Round 4 MC sensitivity timing artifact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_latency(source: Path, legacy_e2e: Path, output_json: Path, figure_path: Path) -> dict:
    frame = pd.read_csv(source)
    required = {"seed", "T", "n_test", "latency_ms", "auc_pr", "ece"}
    if not required.issubset(frame.columns):
        raise ValueError(f"latency source lacks columns: {sorted(required - set(frame.columns))}")
    if frame.n_test.nunique() != 1 or int(frame.n_test.iloc[0]) != 3648:
        raise ValueError("Round 4 latency rows must each cover the full 3,648-event test panel")
    aggregate = frame.groupby("T", as_index=False).agg(
        auc_pr=("auc_pr", "mean"), ece=("ece", "mean"),
        panel_elapsed_ms=("latency_ms", "median"), seed_count=("seed", "nunique"),
    )
    payload = {
        "source_file": str(source),
        "source_sha256": _sha256(source),
        "unit": "milliseconds per complete held-out panel",
        "measurement_scope": "model inference loop over all 3,648 test events in DataLoader batches of 128",
        "included": ["batch device transfer", "deterministic or MC model forward passes"],
        "excluded": ["model/dataset loading", "metric computation", "CSV serialization"],
        "warm_up_policy": "not recorded in source runner",
        "cpu_gpu_synchronization": "no explicit torch.cuda.synchronize call recorded",
        "single_event_or_batch": "batched full-panel elapsed time, not single-event latency",
        "summary_statistic": "median full-panel elapsed milliseconds across five seeds for each T",
        "sample_count": {"seeds_per_T": 5, "events_per_seed_T": 3648},
        "legacy_e2e_source": str(legacy_e2e),
        "legacy_e2e_source_sha256": _sha256(legacy_e2e) if legacy_e2e.is_file() else None,
        "legacy_comparison": "different quantity: controlled single-event end-to-end pipeline timing on synthetic time order",
        "publication_label": "Full held-out-panel inference elapsed time (ms)",
        "latency_scope_consistent": True,
        "limitations": [
            "timings are retained as sensitivity/runtime context, not hardware-normalized latency claims",
            "absence of explicit warm-up and GPU synchronization prevents microbenchmark interpretation",
        ],
        "aggregate": aggregate.to_dict(orient="records"),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(6.2, 4.0))
    time_axis = axis.twinx()
    axis.plot(aggregate["T"], aggregate.auc_pr, marker="o", label="AUC-PR", color="#315d8a")
    axis.plot(aggregate["T"], aggregate.ece, marker="s", label="ECE", color="#d2765e")
    time_axis.plot(
        aggregate["T"], aggregate.panel_elapsed_ms, marker="^",
        label="Full-panel elapsed time", color="#4f8a52",
    )
    axis.set(xlabel="MC passes T", ylabel="Metric value")
    time_axis.set_ylabel("Full held-out-panel inference elapsed time (ms)")
    axis.set_title("Offline MC sensitivity and batched panel runtime")
    lines = axis.lines + time_axis.lines
    axis.legend(lines, [line.get_label() for line in lines], fontsize=8, loc="center right")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return payload
