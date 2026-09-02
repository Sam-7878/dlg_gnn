"""Build repeated-timing and extended-MC manuscript artifacts."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from validation.sci_v3_final_common import atomic_csv


def save(path: Path, frame: pd.DataFrame) -> None:
    atomic_csv(path.with_suffix(".csv"), frame)
    path.with_suffix(".tex").write_text(frame.to_latex(index=False, float_format=lambda value: f"{value:.3f}", escape=True), encoding="utf-8")


def main() -> None:
    root = Path("results/sci_v3_submission_r2")
    table_root, figure_root = root / "manuscript/tables", root / "manuscript/figures"
    timing = pd.read_csv(root / "runtime/five_repeat_timing.csv")
    summary = timing.groupby(["seed", "scenario", "policy"]).agg(
        repeats=("repeat", "count"), deep_route_rate=("deep_route_rate", "mean"),
        mean_latency_ms=("mean_latency_ms", "mean"), mean_latency_sd_ms=("mean_latency_ms", "std"),
        p95_latency_ms=("p95_latency_ms", "mean"), p99_latency_ms=("p99_latency_ms", "mean"),
        throughput_events_s=("throughput_events_per_second", "mean"), rss_peak_bytes=("rss_peak_bytes", "max"),
        vram_peak_bytes=("vram_peak_bytes", "max")).reset_index()
    save(table_root / "table_calibrated_five_repeat_runtime", summary)
    mc = pd.read_csv(root / "runtime/mc_sensitivity_latency.csv")
    columns = ["seed", "mc_samples", "mc_samples" if "mc_samples_executed" not in mc else "mc_samples_executed",
               "mean_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput_events_per_second", "deep_route_rate"]
    columns = list(dict.fromkeys(columns))
    save(table_root / "table_mc_sensitivity", mc[columns])
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, left = plt.subplots(figsize=(6.3, 4.0)); right = left.twinx()
    left.plot(mc.mc_samples, mc.mean_latency_ms, marker="o", color="#4C78A8", label="Mean latency")
    right.plot(mc.mc_samples, mc.throughput_events_per_second, marker="s", color="#E45756", label="Throughput")
    left.set(xlabel="Requested MC passes (T)", ylabel="Mean latency (ms)"); right.set_ylabel("Events/s")
    handles = left.lines + right.lines; left.legend(handles, [line.get_label() for line in handles], loc="upper left")
    figure.tight_layout(); figure.savefig(figure_root / "figure_mc_sensitivity.pdf", bbox_inches="tight")
    figure.savefig(figure_root / "figure_mc_sensitivity.png", dpi=200, bbox_inches="tight"); plt.close(figure)
    print(f"wrote {len(summary)} repeated-timing rows and {len(mc)} MC rows")


if __name__ == "__main__":
    main()
