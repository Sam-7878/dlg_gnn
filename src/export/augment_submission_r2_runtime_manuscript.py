"""Add repeated timing and extended-MC artifacts to the integrated manuscript."""
from pathlib import Path


def main() -> None:
    path = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC/DLG-StreamMC.tex")
    text = path.read_text(encoding="utf-8")
    if "tab:runtime_repeats" in text:
        print("runtime augmentation already present"); return
    marker = "\\begin{table*}[t]\\centering\\caption{Measured raw-event runtime frontier.}\\label{tab:runtime_complete}\\resizebox{\\textwidth}{!}{\\input{generated_r2/tables/table_runtime_measured.tex}}\\end{table*}"
    addition = marker + r"""
The calibrated route was additionally timed in five measured repeats per fixed seed after a 25-event warmup. Table~\ref{tab:runtime_repeats} reports the repeat mean and between-repeat standard deviation together with tail latency, throughput, route rate, RSS, and VRAM. This separates run-to-run timing variability from the earlier single-trace frontier.
\begin{table*}[t]\centering\caption{Warm-started five-repeat calibrated-route timing.}\label{tab:runtime_repeats}\resizebox{\textwidth}{!}{\input{generated_r2/tables/table_calibrated_five_repeat_runtime.tex}}\end{table*}

\subsection{Monte Carlo Pass Sensitivity}
The representative seed-11 raw-event workload was measured at $T\in\{1,3,5,8,10,20,30\}$ after the same 25-event warmup. Figure~\ref{fig:mc_sensitivity} exposes the latency--throughput trade-off. The preserved identical 500-event prefix contains no fraud positives, so accuracy metrics from this runtime prefix are undefined; held-out graph predictions, not this sensitivity run, remain the accuracy evidence.
\begin{figure}[t]\centering\includegraphics[width=\columnwidth]{generated_r2/figures/figure_mc_sensitivity.pdf}\caption{Measured MC-pass latency and throughput sensitivity on the preserved raw-event prefix.}\label{fig:mc_sensitivity}\end{figure}
\begin{table*}[t]\centering\caption{Extended measured MC-pass sensitivity.}\label{tab:mc_sensitivity}\resizebox{\textwidth}{!}{\input{generated_r2/tables/table_mc_sensitivity.tex}}\end{table*}"""
    if marker not in text: raise RuntimeError("runtime insertion marker missing")
    path.write_text(text.replace(marker, addition, 1), encoding="utf-8")
    print("augmented repeated timing and MC sensitivity")


if __name__ == "__main__":
    main()
