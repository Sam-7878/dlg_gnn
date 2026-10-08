#!/usr/bin/env python3
"""Plot only selected raw-backed A04 memory measurements."""
from __future__ import annotations
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "evaluation/benchmark/v2/paper_ready_final/table_memory_selected_a04.csv"
OUT = ROOT / "evaluation/benchmark/v2/paper_ready_final"
ORDER = (("Cora-Syn", "DOMINANT"), ("DGraphFin", "DOMINANT"),
         ("Ethereum", "DLG-Aug"), ("Reddit-Syn", "DLG-Base"))


def main() -> None:
    records = list(csv.DictReader(SOURCE.open()))
    for row in records:
        raw = ROOT / row["raw_log_path"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row["raw_log_sha256"]
    grouped = {(r["dataset"], r["model"], r["envelope"]): r for r in records}
    assert len(grouped) == 8
    fig, ax = plt.subplots(figsize=(8.4, 4.9), constrained_layout=True)
    xs = np.arange(4)
    width = 0.34
    full = [int(grouped[d,m,"full"]["peak_allocated_bytes"]) / 2**30 for d,m in ORDER]
    cap = [int(grouped[d,m,"cap8g"]["peak_allocated_bytes"]) / 2**30 for d,m in ORDER]
    ax.bar(xs - width/2, full, width, color="#2865a5", label="Full 24 GiB envelope")
    for index, ((dataset, model), value) in enumerate(zip(ORDER, cap)):
        row = grouped[dataset, model, "cap8g"]
        if row["support_status"] == "SUPPORTED_EXACT":
            ax.bar(index + width/2, value, width, color="#55a592", label="8 GiB allocator cap" if index == 0 else None)
        else:
            ax.scatter(index + width/2, value, marker="x", s=90, linewidths=2,
                       color="#bb513c", zorder=4, label="8 GiB run: OOM")
            ax.annotate("OOM", (index + width/2, value), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=8, color="#9d3d30")
    ax.axhline(8, color="#555555", linestyle=":", linewidth=1, label="8 GiB cap")
    ax.set_xticks(xs, [f"{d}\n{m}" for d,m in ORDER])
    ax.set_ylabel("Measured peak CUDA allocated memory (GiB)")
    ax.set_ylim(0, 17)
    ax.grid(axis="y", alpha=.22)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Selected measured full-graph execution cases (seed 42)")
    fig.text(.5, -.015,
             "Selected measured cases, not a complete memory profile of all detectors. "
             "The X marks a run that ended in OOM; its height is the observed allocation before failure.",
             ha="center", va="top", fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure_memory_measured_a05.{ext}", dpi=220, bbox_inches="tight")
    print(OUT / "figure_memory_measured_a05.png")


if __name__ == "__main__": main()
