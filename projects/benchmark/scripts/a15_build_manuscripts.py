#!/usr/bin/env python3
"""
A15 Manuscript Builder
Reconstructs Preprints and Applied Sciences manuscripts for A15.
Synchronizes Table 1, Table 2, Table 4, support paragraph, and tag identities.
"""
import shutil
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PAPER_A14 = ROOT / "projects/benchmark/paper/a14"
PAPER_A15 = ROOT / "projects/benchmark/paper/a15"
PUB_A15 = ROOT / "projects/benchmark/publication/a15"

# 1. Create clean paper/a15 from paper/a14
if PAPER_A15.exists():
    shutil.rmtree(PAPER_A15)
shutil.copytree(PAPER_A14, PAPER_A15)

# Copy newly generated tables
tables_src = PAPER_A15 / "tables"

# 2. Read verified Table 2 values from main_ap_mean_sd.csv
main_ap_df = pd.read_csv(PUB_A15 / "main_ap_mean_sd.csv").set_index("dataset")

def format_ap_cell(m_mean, m_sd, status, is_best=False):
    if status == "UNSUPPORTED" or pd.isna(m_mean):
        return "N/A"
    val_str = f"{m_mean:.4f}"
    if is_best:
        return f"\\mathbf{{{val_str}}}"
    return f"{val_str}"

# Define canonical order of datasets for Table 2
canonical_order = [
    "Elliptic", "DGraphFin", "BitcoinOTC", "Ethereum", "BSC", "Polygon",
    "Yelp-Syn", "Amazon-Syn", "Reddit-Syn", "Flickr-Syn", "Cora-Syn", "CiteSeer-Syn", "PubMed-Syn",
    "LANL-RedTeam"
]

models = ["DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]

# Build Table 2 LaTeX rows
table2_rows = []
for ds in canonical_order:
    row_data = main_ap_df.loc[ds]
    pi_test = row_data["pi_test"]
    pi_str = f"{pi_test:.4f}"
    
    # Find best model
    best_val = -1.0
    best_model = None
    for m in models:
        if row_data[f"{m}_status"] == "SUPPORTED":
            val = row_data[f"{m}_mean"]
            if val > best_val:
                best_val = val
                best_model = m
                
    cells = [ds, f"${pi_str}$"]
    for m in models:
        m_mean = row_data[f"{m}_mean"]
        m_sd = row_data[f"{m}_sd"]
        status = row_data[f"{m}_status"]
        is_best = (m == best_model)
        cells.append(format_ap_cell(m_mean, m_sd, status, is_best))
        
    line = " & ".join(cells) + r" \\"
    if ds == "PubMed-Syn":
        line += r"\midrule"
    elif ds == "LANL-RedTeam":
        line += r"\bottomrule"
    table2_rows.append(line)

table2_body = "\n".join(table2_rows)

table2_tex = r"""\begin{table}[htbp]
\centering\small\setlength{\tabcolsep}{3.5pt}
\caption{Average Precision (PR-AUC; mean across five seeds). Evaluated test prevalence ($\pi_{\mathrm{test}}$) defines the empirical no-skill random-ranking baseline on the evaluated test split. Unsupported cells are denoted as N/A.}
\label{tab:pr_auc_main}
\begin{tabularx}{\textwidth}{lrrrrrrrr}\toprule
\textbf{Dataset} & $\pi_{\mathrm{test}}$ & \textbf{DOMINANT} & \textbf{AnomalyDAE} & \textbf{CoLA} & \textbf{GADNR} & \textbf{OCGNN} & \textbf{DLG-Base} & \textbf{DLG-Aug} \\\midrule
""" + table2_body + r"""
\end{tabularx}
\begin{tablenotes}\footnotesize
\item Note: For datasets with random stratified cross-seed splits (Ethereum, BSC, Polygon, Cora), $\pi_{\mathrm{test}}$ reports the empirical mean test positive rate across seeds 42--46.
\end{tablenotes}
\end{table}"""

# Table 1 LaTeX content (A15-G1: Disambiguate Graph Target Prevalence vs Evaluated Test Prevalence)
t1_tex = r"""\begin{table}[htbp]
\centering\small
\caption{Benchmark graph dataset properties, evaluated model-input populations, target label provenance, and evaluation test prevalence. Model-input counts ($N, E, d$) reflect the exact tensors evaluated by all models; upstream raw source graph totals are reported separately. Graph Target Prevalence ($\pi_{\mathrm{graph}} = y_{\mathrm{pos}} / N$) reflects whole-graph label proportions; Evaluated Test Prevalence ($\pi_{\mathrm{test}}$) defines the empirical no-skill AP baseline on the evaluated test split (mean across seeds 42--46).}
\label{tab:dataset_properties}
\begin{tabularx}{\textwidth}{lrrrrrX}\toprule
\textbf{Dataset} & \textbf{Nodes ($N$)} & \textbf{Edges ($E$)} & \textbf{Features ($d$)} & \textbf{Graph Prev.} & \textbf{Test $\pi_{\mathrm{test}}$} & \textbf{Target Provenance and Evaluated Population} \\\midrule
\multicolumn{7}{l}{\textit{Native Financial / Blockchain Datasets ($N=5$)}} \\
Elliptic & $46,564$ & $83,188$ & $165$ & $0.0976$ & $0.0976$ & Bitcoin transaction graph; illicit entities (Weber et al., 2019). Evaluated induced graph: 46,564 nodes (raw source: 203,769 nodes). \\
DGraphFin & $1,225,601$ & $746,271$ & $17$ & $0.0127$ & $0.0127$ & FinTech loan user graph; fraudulent defaulters (Huang et al., 2022). Evaluated induced graph: 1,225,601 nodes (raw source: 3,700,550 nodes). \\
Ethereum & $14,385$ & $115,065$ & $8$ & $0.4184$ & $0.4089^*$ & Token-transfer feature-similarity graph; verified phishing/scam contracts (Luo et al., 2024). Test split: 2,893 nodes ($40.89\%$ fraud). \\
BSC & $7,481$ & $59,832$ & $8$ & $0.1476$ & $0.1473^*$ & Binance Smart Chain token transfer graph; fraud contracts (Luo et al., 2024). Test split: 1,500 nodes ($14.73\%$ fraud). \\
Polygon & $2,303$ & $18,411$ & $8$ & $0.0261$ & $0.0259^*$ & Polygon token transfer graph; verified fraud contracts (Luo et al., 2024). Test split: 471 nodes ($2.59\%$ fraud). \\
\midrule
\multicolumn{7}{l}{\textit{Injected Synthetic Benchmarks ($N=8$, including Real Trust Graph with Injected Targets)}} \\
BitcoinOTC & $6,005$ & $7,409$ & $67$ & $0.0300$ & $0.0300$ & SNAP OTC signed trust network (Kumar et al., 2016); controlled injected anomaly targets on real graph. Test split: 3,003 nodes. \\
Yelp-Syn & $716,847$ & $14,004,995$ & $300$ & $0.0100$ & $0.0100$ & Social review network; injected structural and contextual anomalies. (PyG/PyGOD evaluated tensor: 716,847 nodes). Test split: 358,424 nodes. \\
Amazon-Syn & $13,752$ & $507,378$ & $767$ & $0.0300$ & $0.0300$ & Co-purchasing graph; injected structural and contextual anomalies. (PyG/PyGOD evaluated tensor: 13,752 nodes). Test split: 6,876 nodes. \\
Reddit-Syn & $232,965$ & $114,869,737$ & $602$ & $0.0200$ & $0.0200$ & Reddit post interaction graph; injected structural anomalies. (PyG/PyGOD evaluated tensor: 232,965 nodes). Test split: 116,483 nodes. \\
Flickr-Syn & $89,250$ & $1,001,494$ & $500$ & $0.0200$ & $0.0200$ & Image sharing network; injected structural anomalies. (PyG/PyGOD evaluated tensor: 89,250 nodes). Test split: 44,625 nodes. \\
Cora-Syn & $2,708$ & $11,276$ & $1433$ & $0.0299$ & $0.0298^*$ & Citation network; injected structural anomalies (PyGOD generator). (PyG/PyGOD evaluated tensor: 2,708 nodes). Test split: 1,354 nodes. \\
CiteSeer-Syn & $3,327$ & $9,914$ & $3703$ & $0.0298$ & $0.0300$ & Citation network; injected structural anomalies (PyGOD generator). (PyG/PyGOD evaluated tensor: 3,327 nodes). Test split: 1,664 nodes. \\
PubMed-Syn & $19,717$ & $93,958$ & $500$ & $0.0300$ & $0.0300$ & Citation network; injected structural anomalies (PyGOD generator). (PyG/PyGOD evaluated tensor: 19,717 nodes). Test split: 9,859 nodes. \\
\midrule
\multicolumn{7}{l}{\textit{External Enterprise Cybersecurity Network ($N=1$)}} \\
LANL-RedTeam & $16,694$ & $323,897$ & $13$ & $0.0180$ & $0.0180$ & Enterprise authentication graph; red-team compromised computers (Kent, 2015). Evaluated destination computers: 16,694 nodes. Test split: 3,338 nodes. \\
\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize
\item $^*$ For datasets with random stratified cross-seed splits, $\pi_{\mathrm{test}}$ denotes the empirical mean test-set positive rate across seeds 42--46 (Ethereum: $0.3951$--$0.4307$, mean $0.4089$; BSC: $0.1353$--$0.1600$, mean $0.1473$; Polygon: $0.0149$--$0.0382$, mean $0.0259$; Cora: $0.0295$--$0.0303$, mean $0.0298$).
\end{tablenotes}
\end{table}"""
(PAPER_A15 / "tables/table_dataset_properties.tex").write_text(t1_tex)

# Table 4 LaTeX content (A15-G3: Observational characterization without causal leaps)
t4_tex = r"""\begin{table}[htbp]
\centering\footnotesize\setlength{\tabcolsep}{4.5pt}
\caption{Conditional Average Precision performance of DLG-Base versus DLG-Aug across all 14 evaluated datasets (13 primary graphs plus external LANL-RedTeam). Results report mean AP and seed-paired delta ($\Delta\mathrm{AP} = \mathrm{AP}_{\mathrm{Aug}} - \mathrm{AP}_{\mathrm{Base}}$) alongside paired standard errors ($n=5$ random seeds). Localized structural augmentation provides clear utility in sparse transaction graphs but exhibits performance trade-offs on dense social networks and bipartite cyber graphs.}
\label{tab:conditional_delta}
\begin{tabularx}{\textwidth}{lrrrr>{\raggedright\arraybackslash}X}\toprule
\textbf{Dataset} & \textbf{DLG-Base (AP)} & \textbf{DLG-Aug (AP)} & $\mathbf{\Delta\mathrm{AP}}$ & \textbf{Paired SE} & \textbf{Operational Characterization} \\\midrule
Elliptic & $0.0687$ & $0.1037$ & $\mathbf{+0.0350}$ & $\pm 0.0022$ & \textbf{Large Gain} ($+50.97\%$ rel.); positive $\Delta\mathrm{AP}$ observed; mechanism not isolated by this comparison. \\
DGraphFin & $0.0101$ & $0.0134$ & $+0.0034$ & $\pm 0.0008$ & \textbf{Gain} ($+33.23\%$ rel.); exact full-graph execution supported under the reported envelope. \\
BitcoinOTC & $0.8274$ & $0.8211$ & $-0.0063$ & $\pm 0.0033$ & \textbf{Slight Loss} ($-0.77\%$ rel.); high baseline precision ($0.827$) retains competitive performance. \\
Ethereum & $0.4106$ & $0.4359$ & $\mathbf{+0.0253}$ & $\pm 0.0144$ & \textbf{Gain} ($+6.15\%$ rel.); DLG-Aug has higher mean AP in this constructed similarity-graph evaluation. \\
BSC & $0.3390$ & $0.3411$ & $+0.0021$ & $\pm 0.0103$ & \textbf{Marginal Gain} ($+0.61\%$ rel.); slight positive mean difference on BSC token transfers. \\
Polygon & $0.0854$ & $0.0731$ & $-0.0124$ & $\pm 0.0054$ & \textbf{Loss} ($-14.48\%$ rel.); negative $\Delta\mathrm{AP}$ observed in this similarity-graph evaluation. \\
Yelp-Syn & $0.0826$ & $0.0832$ & $+0.0007$ & $\pm 0.0015$ & Stable / marginal gain on injected contextual/structural anomaly patterns. \\
Amazon-Syn & $0.1245$ & $0.1302$ & $+0.0057$ & $\pm 0.0016$ & Stable / marginal gain on injected contextual/structural anomaly patterns. \\
Reddit-Syn & $0.4044$ & $0.3388$ & $\mathbf{-0.0656}$ & $\pm 0.0031$ & \textbf{Substantial Loss} ($-16.22\%$ rel.); dense community structure exhibits negative delta. \\
Flickr-Syn & $0.4569$ & $0.4599$ & $+0.0030$ & $\pm 0.0076$ & Stable / marginal gain on injected contextual/structural anomaly patterns. \\
Cora-Syn & $0.4758$ & $0.4720$ & $-0.0038$ & $\pm 0.0031$ & Stable citation network performance; local features preserve structural fidelity. \\
CiteSeer-Syn & $0.6004$ & $0.5992$ & $-0.0013$ & $\pm 0.0014$ & Stable citation network performance; local features preserve structural fidelity. \\
PubMed-Syn & $0.1623$ & $0.1605$ & $-0.0018$ & $\pm 0.0003$ & Stable citation network performance; local features preserve structural fidelity. \\
LANL-RedTeam$^{\dagger}$ & $0.1367$ & $0.1114$ & $-0.0252$ & $\pm 0.0126$ & \textbf{External Loss} ($-18.47\%$ rel.); negative $\Delta\mathrm{AP}$ observed on bipartite cyber authentication graph. \\
\bottomrule
\end{tabularx}
\begin{tablenotes}\footnotesize
\item $^{\dagger}$ LANL-RedTeam is an external enterprise cybersecurity network evaluated under identical protocol but reported as single-graph descriptive evidence.
\end{tablenotes}
\end{table}"""
(PAPER_A15 / "tables/table_conditional_delta.tex").write_text(t4_tex)

# Text updates:
old_support_text = (
    r"Of the 91 primary model--dataset pairs ($13 \times 7$), exactly 84 pairs are supported across all five seeds ($84 \times 5 = 420$ successful primary runs). "
    r"GADNR fails due to OOM or numerical covariance determinant exceptions on seven graphs, while AnomalyDAE exhausts memory on Yelp-Syn and Reddit-Syn due to its $O(N^2)$ pairwise cross-attention. "
    r"In contrast, the exact Gram formulations developed in Section~\ref{sec:models} allow DOMINANT, DLG-Base, and DLG-Aug to complete all runs successfully without memory overflow, even on the 1.2-million-node DGraphFin graph."
)

new_support_text = (
    r"Of the 91 primary model--dataset pairs ($13 \times 7$), exactly 84 pairs are supported across all five seeds ($84 \times 5 = 420$ successful primary runs). "
    r"GADNR fails due to OOM or numerical covariance determinant exceptions on five graphs (Elliptic, DGraphFin, Yelp-Syn, Flickr-Syn, and Reddit-Syn), while AnomalyDAE exhausts memory on Yelp-Syn and Reddit-Syn due to its $O(N^2)$ pairwise cross-attention (seven unsupported primary pairs in total). "
    r"In contrast, the exact Gram formulations developed in Section~\ref{sec:models} allow DOMINANT, DLG-Base, and DLG-Aug to complete all runs successfully without memory overflow, even on the 1.2-million-node DGraphFin graph."
)

old_interp_text = r"""\textbf{Interpretation of Prevalence Baselines and Comparator Superiority:}
\begin{enumerate}
    \item On the Elliptic dataset, DLG-Aug is the only model achieving AP higher than the test prevalence ($0.1037 > 0.0976$), whereas DOMINANT ($0.0682$), AnomalyDAE ($0.0538$), and DLG-Base ($0.0687$) fall below random baseline precision.
    \item On DGraphFin, DLG-Aug reaches exact parity with prevalence ($0.0134$), while other baselines perform near or below prevalence.
    \item On the three cryptocurrency networks, established comparators outperform DLG-Aug: OCGNN achieves the highest AP on Ethereum ($0.4996$ vs.\ Aug $0.4359$), while AnomalyDAE leads on BSC ($0.3609$ vs.\ Aug $0.3411$) and Polygon ($0.1878$ vs.\ Aug $0.0731$). Thus, \textit{at least one established comparator exceeds DLG-Aug in mean AP on each of the three blockchain datasets.}
\end{enumerate}"""

new_interp_text = r"""\textbf{Interpretation of Prevalence Baselines and Comparator Superiority:}
\begin{enumerate}
    \item On the Elliptic dataset, DLG-Aug is the only model achieving AP higher than the test prevalence ($0.1037 > 0.0976$), whereas DOMINANT ($0.0682$), AnomalyDAE ($0.0538$), and DLG-Base ($0.0687$) fall below random baseline precision.
    \item On DGraphFin, DLG-Aug ($0.0134$) and CoLA ($0.0127$) perform at or above the verified evaluation prevalence baseline ($0.0127$), while other baselines perform near or below prevalence.
    \item On the three cryptocurrency networks, established comparators outperform DLG-Aug: OCGNN achieves the highest AP on Ethereum ($0.4996$ vs.\ Aug $0.4359$, both exceeding the $0.4089$ test prevalence), while AnomalyDAE leads on BSC ($0.3609$ vs.\ Aug $0.3411$) and Polygon ($0.1878$ vs.\ Aug $0.0731$). Thus, \textit{at least one established comparator exceeds DLG-Aug in mean AP on each of the three blockchain datasets.}
\end{enumerate}"""

def update_manuscript(tex_path, new_filename):
    text = tex_path.read_text(encoding="utf-8")
    
    # 1. Update tag strings
    text = text.replace("benchmark-a14-submission-2026-10-24", "benchmark-a15-submission-2026-10-25")
    text = text.replace("A14", "A15")
    text = text.replace("a14", "a15")
    
    # 2. Update support text
    if old_support_text in text:
        text = text.replace(old_support_text, new_support_text)
    else:
        print(f"Warning: old_support_text not found in {tex_path.name}")
        
    # 3. Update interpretation text
    if old_interp_text in text:
        text = text.replace(old_interp_text, new_interp_text)
    else:
        print(f"Warning: old_interp_text not found in {tex_path.name}")

    # 4. Replace Table 1
    pos_t1 = text.find(r"\label{tab:dataset_properties}")
    if pos_t1 != -1:
        start_t1 = text.rfind(r"\begin{table}", 0, pos_t1)
        end_t1 = text.find(r"\end{table}", pos_t1) + len(r"\end{table}")
        text = text[:start_t1] + t1_tex + text[end_t1:]
    else:
        print(f"Warning: tab:dataset_properties not found in {tex_path.name}")

    # 5. Replace Table 2
    pos_t2 = text.find(r"\label{tab:pr_auc_main}")
    if pos_t2 != -1:
        start_t2 = text.rfind(r"\begin{table}", 0, pos_t2)
        end_t2 = text.find(r"\end{table}", pos_t2) + len(r"\end{table}")
        text = text[:start_t2] + table2_tex + text[end_t2:]
    else:
        print(f"Warning: tab:pr_auc_main not found in {tex_path.name}")

    # 6. Replace Table 4
    pos_t4 = text.find(r"\label{tab:conditional_delta}")
    if pos_t4 != -1:
        start_t4 = text.rfind(r"\begin{table}", 0, pos_t4)
        end_t4 = text.find(r"\end{table}", pos_t4) + len(r"\end{table}")
        text = text[:start_t4] + t4_tex + text[end_t4:]
    else:
        print(f"Warning: tab:conditional_delta not found in {tex_path.name}")

    # Write out new file
    out_path = tex_path.parent / new_filename
    out_path.write_text(text, encoding="utf-8")
    if tex_path.name != new_filename and tex_path.exists():
        tex_path.unlink()
    print(f"Wrote updated manuscript: {out_path}")

# Update Preprints
prep_src = PAPER_A15 / "DLG-Benchmark_A14_Preprints.tex"
update_manuscript(prep_src, "DLG-Benchmark_A15_Preprints.tex")

# Update Applied Sciences
appl_src = PAPER_A15 / "mdpi/DLG-Benchmark_A14_AppliedSciences.tex"
update_manuscript(appl_src, "DLG-Benchmark_A15_AppliedSciences.tex")

print("A15 manuscript reconstruction complete.")
