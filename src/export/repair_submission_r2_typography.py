"""Reduce remaining material overfull boxes in the LaTeX master."""
from pathlib import Path


def main() -> None:
    path = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC/DLG-StreamMC.tex")
    text = path.read_text(encoding="utf-8")
    text = text.replace(r"""\begin{equation}
\begin{aligned}
\text{GoG representation/data}&\rightarrow\text{DLG-GNN local-to-global hierarchy}\\
&\rightarrow\text{DLG-StreamMC selective bounded streaming}.
\end{aligned}
\end{equation}""", r"""\begin{equation}
\begin{gathered}
\text{GoG representation/data}\\[-1mm]
\downarrow\\[-1mm]
\text{DLG-GNN local-to-global hierarchy}\\[-1mm]
\downarrow\\[-1mm]
\text{DLG-StreamMC selective bounded streaming}.
\end{gathered}
\end{equation}""")
    text = text.replace(r"""\begin{equation}
\begin{aligned}
\text{raw event}
&\rightarrow
\text{bounded local graph state}
\rightarrow
\text{ProductionLevel1GIN}\\
&\rightarrow
\text{MC uncertainty}
\rightarrow
\text{Selective Router}\\
&\rightarrow
\begin{cases}
\text{direct decision},\\
\text{ProductionLevel2GATv2}
\rightarrow
\text{WeightedLogitFusion}.
\end{cases}
\end{aligned}
\label{eq:final_path}
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\text{raw event}&\rightarrow\text{bounded local state}\rightarrow\text{L1 GIN}\\
&\rightarrow\text{MC uncertainty}\rightarrow\text{router}\\
&\rightarrow\begin{cases}\text{direct decision},\\\text{L2 GATv2}\rightarrow\text{logit fusion}.\end{cases}
\end{aligned}
\label{eq:final_path}
\end{equation}""")
    text = text.replace(r"""\begin{equation}
\mathbb{E}[C_{\mathrm{selective}}]
=
C_{\mathrm{state}}
+
C_{\mathrm{L1+MC}}
+
C_{\mathrm{router}}
+
P(\textsc{Deep})
C_{\mathrm{deep}}
+
C_{\mathrm{overhead}}
\label{eq:expected_cost}
\end{equation}""", r"""\begin{equation}
\begin{aligned}
\mathbb{E}[C_{\mathrm{selective}}]
={}&C_{\mathrm{state}}+C_{\mathrm{L1+MC}}+C_{\mathrm{router}}\\
&+P(\textsc{Deep})C_{\mathrm{deep}}+C_{\mathrm{overhead}}.
\end{aligned}
\label{eq:expected_cost}
\end{equation}""")
    text = text.replace(r"""\begin{verbatim}
configs/sci_v3_submission/production_closure.yaml
\end{verbatim}""", r"""\begin{lstlisting}[basicstyle=\ttfamily\scriptsize,breaklines=true]
configs/sci_v3_submission/production_closure.yaml
\end{lstlisting}""")
    text = text.replace(r"""\begin{equation}
\text{contract-local reasoning}
\quad\text{vs.}\quad
\text{inter-contract relational reasoning}.
\end{equation}""", r"""\begin{equation}
\begin{gathered}
\text{contract-local reasoning}\\[-1mm]
\text{versus}\\[-1mm]
\text{inter-contract relational reasoning}.
\end{gathered}
\end{equation}""")
    path.write_text(text, encoding="utf-8")
    print("repaired remaining material typography overflows")


if __name__ == "__main__":
    main()
