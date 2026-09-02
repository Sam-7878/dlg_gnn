"""Build the five-policy repeated runtime manuscript table."""
from pathlib import Path

import pandas as pd

from validation.sci_v3_final_common import atomic_csv


def main() -> None:
    root = Path("results/sci_v3_submission_r2")
    frame = pd.read_csv(root/"runtime/five_repeat_policy_summary.csv")
    columns = ["policy","seeds","repeats","deep_route_rate","mean_latency_ms","mean_latency_sd_ms",
               "p95_latency_ms","p99_latency_ms","throughput_events_s","latency_reduction_vs_full_deep_pct",
               "rss_peak_bytes","vram_peak_bytes"]
    table = frame[columns]
    destination = root/"manuscript/tables/table_calibrated_five_repeat_runtime"
    atomic_csv(destination.with_suffix(".csv"), table)
    destination.with_suffix(".tex").write_text(table.to_latex(index=False, float_format=lambda value:f"{value:.3f}", escape=True), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
