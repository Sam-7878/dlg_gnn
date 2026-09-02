"""Finalize manuscript runtime claims from the repeated five-policy evidence."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


MANUSCRIPT = Path(
    "docs/work_reports/110_stream_mc_sci_v3_submission_r2/"
    "_41_01_DLG_StreamMC/DLG-StreamMC.tex"
)
SUMMARY = Path(
    "results/sci_v3_submission_r2/runtime/five_repeat_policy_summary.csv"
)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def main() -> None:
    frame = pd.read_csv(SUMMARY).set_index("policy")
    calibrated = frame.loc["validation_calibrated_dual"]
    risk = frame.loc["legacy_risk_controlled"]
    dual = frame.loc["legacy_dual"]

    calibrated_route = pct(float(calibrated.deep_route_rate))
    calibrated_reduction = f"{float(calibrated.latency_reduction_vs_full_deep_pct):.2f}"
    calibrated_latency = f"{float(calibrated.mean_latency_ms):.3f}"
    full_latency = f"{float(frame.loc['full_deep'].mean_latency_ms):.3f}"
    risk_route = pct(float(risk.deep_route_rate))
    dual_route = pct(float(dual.deep_route_rate))

    text = MANUSCRIPT.read_text(encoding="utf-8")

    old_contribution = (
        "Resource claims are based on actual selective execution from chronologically ordered raw "
        "transactions. Against full-deep inference on identical 500-event prefixes, dual routing "
        "reduced mean end-to-end latency by 9.41\\%, while validation-constrained empirical "
        "risk-controlled routing reduced mean latency by 31.65\\%."
    )
    new_contribution = (
        "Resource claims are based on actual selective execution from chronologically ordered raw "
        f"transactions. In a warm-started five-seed, five-repeat policy comparison on identical "
        f"500-event prefixes, the validation-calibrated route sent {calibrated_route}\\% of events "
        f"to the deep stage and reduced mean end-to-end latency by {calibrated_reduction}\\% "
        "relative to full-deep execution."
    )
    text = text.replace(old_contribution, new_contribution)

    old_runtime = (
        "The raw-event profiler measures ingestion, state update, subgraph extraction, feature "
        "construction, GIN/MC execution, routing, optional relation construction and GATv2/fusion, "
        "queues, and trace construction. Direct exits skip all deep-stage operations. "
        "Table~\\ref{tab:runtime_complete} reports mean, P95, P99, throughput, route rate, RSS, and "
        "VRAM rather than only percentage reductions. Consequently, the previously reported "
        "9.41\\% and 31.65\\% reductions are interpreted as configuration-specific measured "
        "observations; the distinct route rates explain why the empirical-risk policy may be "
        "faster than the dual policy."
    )
    new_runtime = (
        "The raw-event profiler measures ingestion, state update, subgraph extraction, feature "
        "construction, GIN/MC execution, routing, optional relation construction and GATv2/fusion, "
        "queues, and trace construction. Direct exits skip all deep-stage operations. "
        "Table~\\ref{tab:runtime_complete} preserves the earlier single-trace frontier, including "
        "the configuration-specific 9.41\\% and 31.65\\% mean reductions. The repeated policy "
        f"audit in Table~\\ref{{tab:runtime_repeats}} explains the apparent inversion: legacy dual "
        f"routed {dual_route}\\% of events to the deep stage, whereas legacy empirical-risk routing "
        f"routed {risk_route}\\%. The validation-calibrated route reduced this rate to "
        f"{calibrated_route}\\%, with mean latency {calibrated_latency} ms versus {full_latency} ms "
        f"for full-deep execution ({calibrated_reduction}\\% reduction). Route frequency, rather "
        "than a universal property of the policy names, drives the measured ordering."
    )
    text = text.replace(old_runtime, new_runtime)

    old_repeat = (
        "The calibrated route was additionally timed in five measured repeats per fixed seed after "
        "a 25-event warmup. Table~\\ref{tab:runtime_repeats} reports the repeat mean and "
        "between-repeat standard deviation together with tail latency, throughput, route rate, RSS, "
        "and VRAM. This separates run-to-run timing variability from the earlier single-trace "
        "frontier.\n"
        "\\begin{table*}[t]\\centering\\caption{Warm-started five-repeat calibrated-route timing.}"
        "\\label{tab:runtime_repeats}\\resizebox{\\textwidth}{!}{\\input{generated_r2/tables/"
        "table_calibrated_five_repeat_runtime.tex}}\\end{table*}"
    )
    new_repeat = (
        "Five routing policies were additionally timed in five measured repeats per fixed seed "
        "after a 25-event warmup. Table~\\ref{tab:runtime_repeats} reports the aggregate mean and "
        "standard deviation across the 25 seed--repeat runs together with tail latency, throughput, "
        "route rate, RSS, and VRAM. The identical prefix and warm-start protocol isolate routing "
        "policy effects while exposing both seed and repeat variability.\n"
        "\\begin{table*}[t]\\centering\\caption{Warm-started five-policy timing: five repeats for "
        "each of five seeds.}\\label{tab:runtime_repeats}\\resizebox{\\textwidth}{!}{\\input{"
        "generated_r2/tables/table_calibrated_five_repeat_runtime.tex}}\\end{table*}"
    )
    text = text.replace(old_repeat, new_repeat)

    old_boundary = (
        "Measured raw-event E2E mean-latency reduction of $9.41\\%$ (dual) and $31.65\\%$ "
        "(empirical risk-controlled)"
    )
    new_boundary = (
        f"Warm-started 25-run policy audit: {calibrated_reduction}\\% mean-latency reduction at "
        f"{calibrated_route}\\% deep-route rate"
    )
    text = text.replace(old_boundary, new_boundary)

    old_limit = (
        "The reported mean end-to-end latency reductions of $9.41\\%$ and $31.65\\%$ are measured "
        "on the evaluated software stack, hardware, model implementation, and identical 500-event "
        "prefixes. They should not be interpreted as hardware-independent constants."
    )
    new_limit = (
        f"The warm-started repeated policy audit measured a {calibrated_reduction}\\% mean "
        f"end-to-end latency reduction for validation-calibrated routing at a {calibrated_route}\\% "
        "deep-route rate. The earlier 9.41\\% and 31.65\\% single-trace reductions are retained as "
        "configuration-specific observations. None of these values should be interpreted as a "
        "hardware-independent constant."
    )
    text = text.replace(old_limit, new_limit)

    MANUSCRIPT.write_text(text, encoding="utf-8")
    print(
        "finalized repeated policy claims: "
        f"route={calibrated_route}%, reduction={calibrated_reduction}%"
    )


if __name__ == "__main__":
    main()
