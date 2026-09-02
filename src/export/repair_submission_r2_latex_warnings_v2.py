"""Idempotent warning repair for the authoritative LaTeX master."""
from pathlib import Path


def main() -> None:
    path = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC/DLG-StreamMC.tex")
    text = path.read_text(encoding="utf-8")
    replacements = [
        (r"""\begin{equation}
\text{GoG representation/data}
\rightarrow
\text{DLG-GNN local-to-global hierarchy}
\rightarrow
\text{DLG-StreamMC selective bounded streaming}.
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\text{GoG representation/data}&\rightarrow\text{DLG-GNN local-to-global hierarchy}\\
&\rightarrow\text{DLG-StreamMC selective bounded streaming}.
\end{aligned}
\end{equation}"""),
        (r"""\begin{equation}
E_c(t)
=
\left\{
e_\tau:
t-W < \tau \le t
\right\},
\qquad
|V_c(t)|\le N_{\max},
\qquad
|E_c(t)|\le E_{\max},
\label{eq:state_bounds}
\end{equation}""", r"""\begin{equation}
\begin{aligned}
E_c(t)&=\{e_\tau:t-W < \tau \le t\},\\
|V_c(t)|&\le N_{\max},\qquad |E_c(t)|\le E_{\max}.
\end{aligned}
\label{eq:state_bounds}
\end{equation}"""),
        (r"""\begin{equation}
\text{state update}
\rightarrow
\text{local graph build}
\rightarrow
\text{GIN/MC}
\rightarrow
\text{router}
\rightarrow
[\text{optional relation + GATv2 + fusion}],
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\text{state update}&\rightarrow\text{local graph build}\rightarrow\text{GIN/MC}\\
&\rightarrow\text{router}\rightarrow[\text{optional relation + GATv2 + fusion}].
\end{aligned}
\end{equation}"""),
        (r"""\begin{verbatim}
executed_backbone      = ProductionLevel1GIN
paper_method_backbone = ProductionLevel1GIN
profiler_backbone     = ProductionLevel1GIN
routing_backbone      = ProductionLevel1GIN
deep_stage            = ProductionLevel2GATv2
fusion                = WeightedLogitFusion
mc_scope              = ProductionLevel1GIN_dropout_only
\end{verbatim}""", r"""\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,breaklines=true]
executed_backbone      = ProductionLevel1GIN
paper_method_backbone = ProductionLevel1GIN
profiler_backbone     = ProductionLevel1GIN
routing_backbone      = ProductionLevel1GIN
deep_stage            = ProductionLevel2GATv2
fusion                = WeightedLogitFusion
mc_scope              = ProductionLevel1GIN_dropout_only
\end{lstlisting}"""),
        (r"""\begin{equation}
\text{claim}
\rightarrow
\text{table/figure}
\rightarrow
\text{canonical record}
\rightarrow
\text{raw evidence}
\rightarrow
\text{config/code/environment}.
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\text{claim}&\rightarrow\text{table/figure}\rightarrow\text{canonical record}\\
&\rightarrow\text{raw evidence}\rightarrow\text{config/code/environment}.
\end{aligned}
\end{equation}"""),
    ]
    changed = 0
    for old, new in replacements:
        if old in text: text = text.replace(old, new, 1); changed += 1
    path.write_text(text, encoding="utf-8")
    print(f"repaired {changed} overflow blocks")


if __name__ == "__main__":
    main()
