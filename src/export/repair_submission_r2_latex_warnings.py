"""Repair compile-discovered citations and the largest equation overflows."""
from pathlib import Path


def replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text: raise RuntimeError(f"missing replacement block: {label}")
    return text.replace(old, new, 1)


def main() -> None:
    root = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC")
    appendix = (root / "DLG_StreamMC_Appendices_Reader_Guide_v1.tex").read_text(encoding="utf-8")
    appendix = appendix.replace("\\cite{liu2021cola}", "\\cite{liu2022cola}")
    appendix = appendix.replace("\\cite{bandyopadhyay2020done}", "\\cite{bandyopadhyay2020outlier}")
    (root / "DLG_StreamMC_Appendices_Reader_Guide_v1.tex").write_text(appendix, encoding="utf-8")
    path = root / "DLG-StreamMC.tex"; text = path.read_text(encoding="utf-8")
    text = replace_required(text, r"""\begin{equation}
\text{GoG representation/data}
\rightarrow
\text{DLG-GNN local-to-global hierarchy}
\rightarrow
\text{DLG-StreamMC selective bounded streaming}.
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\text{GoG representation/data}
&\rightarrow \text{DLG-GNN local-to-global hierarchy}\\
&\rightarrow \text{DLG-StreamMC selective bounded streaming}.
\end{aligned}
\end{equation}""", "positioning equation")
    text = replace_required(text, r"""\begin{equation}
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
\end{equation}""", "state bounds")
    text = replace_required(text, r"""\begin{equation}
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
\end{equation}""", "profiler pipeline")
    text = replace_required(text, r"""\begin{verbatim}
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
\end{lstlisting}""", "identity listing")
    text = replace_required(text, r"""\begin{equation}
\text{table/figure}
\rightarrow
\text{canonical record}
\rightarrow
\text{raw evidence}
\rightarrow
\text{config/code/environment}.
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\text{table/figure}&\rightarrow\text{canonical record}\rightarrow\text{raw evidence}\\
&\rightarrow\text{config/code/environment}.
\end{aligned}
\end{equation}""", "provenance equation")
    path.write_text(text, encoding="utf-8")
    print("repaired appendix citation keys and largest overfull blocks")


if __name__ == "__main__":
    main()
