#!/usr/bin/env python3
"""
A18 Gate G3, G4, G5: Manuscript Evidence Repair & Editorial Updates
Applies verified mathematical tables, code-faithful method descriptions,
bounded scientific language, and restored literature citations
to both DLG-Benchmark_A16_Preprints.tex and DLG-Benchmark_A16_AppliedSciences.tex.
"""

from pathlib import Path
import re

WORKSPACE_ROOT = Path("/mnt/d/_Work/goat_bank/dlg_gnn")
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = Path("D:/_Work/goat_bank/dlg_gnn")

PREPRINTS_TEX_FILES = [
    WORKSPACE_ROOT / "projects/benchmark/paper/a16/DLG-Benchmark_A16_Preprints.tex",
    WORKSPACE_ROOT / "projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.tex",
]
APPLSCI_TEX_FILES = [
    WORKSPACE_ROOT / "projects/benchmark/paper/a16/mdpi/DLG-Benchmark_A16_AppliedSciences.tex",
    WORKSPACE_ROOT / "projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.tex",
]

TABLE3_FILE = WORKSPACE_ROOT / "projects/benchmark/publication/a18/table_alert_budget_verified.tex"
TABLE_A2_FILE = WORKSPACE_ROOT / "projects/benchmark/publication/a18/f1_table_generated.tex"

for p in PREPRINTS_TEX_FILES + APPLSCI_TEX_FILES:
    assert p.exists(), f"Missing {p}"
assert TABLE3_FILE.exists(), f"Missing {TABLE3_FILE}"
assert TABLE_A2_FILE.exists(), f"Missing {TABLE_A2_FILE}"

with open(TABLE3_FILE, "r", encoding="utf-8") as f:
    table3_body = f.read().strip()

with open(TABLE_A2_FILE, "r", encoding="utf-8") as f:
    table_a2_rows = f.read().strip()


def update_manuscript(tex_path: Path, is_mdpi: bool = False):
    print(f"Updating manuscript: {tex_path.name} (is_mdpi={is_mdpi})")
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Fix Coummunications typo
    content = content.replace("Coummunications", "Communications")

    # 2. Fix Translation typo in Generative AI statement
    content = content.replace("Translation of the text", "translation of the text")

    # 3. Update Table 1 Citations in Table 1 target descriptions
    content = content.replace("illicit entities (Weber et al., 2019)", "illicit entities \\cite{weber2019elliptic}")
    content = content.replace("fraudulent defaulters (Huang et al., 2022)", "fraudulent defaulters \\cite{huang2022dgraph}")
    content = content.replace("verified phishing/scam contracts (Luo et al., 2024)", "verified phishing/scam contracts \\cite{luo2024multichain}")
    content = content.replace("fraud contracts (Luo et al., 2024)", "fraud contracts \\cite{luo2024multichain}")
    content = content.replace("verified fraud contracts (Luo et al., 2024)", "verified fraud contracts \\cite{luo2024multichain}")
    content = content.replace("SNAP OTC signed trust network (Kumar et al., 2016)", "SNAP OTC signed trust network \\cite{kumar2016edge,snapbitcoinotc}")
    content = content.replace("red-team compromised computers (Kent, 2015)", "red-team compromised computers \\cite{kent2015cyberdata}")

    # 4. Update Section 2 (Related Work) citations
    old_sec2_pygod = "Benchmark libraries such as PyGOD \\cite{liu2024pygod} and BOND \\cite{liu2022bond} have standardized implementations on small and medium benchmarks."
    new_sec2_pygod = (
        "Comprehensive surveys \\cite{ma2023survey,qiao2025survey} and benchmark initiatives such as PyGOD \\cite{liu2024pygod}, "
        "BOND \\cite{liu2022bond}, and GADBench \\cite{tang2023gadbench} have standardized implementations across small and medium graphs. "
        "Recent specialized architectures, including PREM \\cite{pan2023prem}, TAM \\cite{qiao2023tam}, GADAM \\cite{chen2024gadam}, "
        "and ADAG \\cite{he2024adag}, explore graph prompt learning, truncation, and adaptive augmentation."
    )
    if old_sec2_pygod in content:
        content = content.replace(old_sec2_pygod, new_sec2_pygod)

    # Update CONAD diagnostic-only description in Section 2.1
    old_conad_text = "Contrastive methods such as CoLA \\cite{liu2022cola} and CONAD \\cite{xu2022conad} train local-to-subgraph mutual information estimators, scoring nodes by contrastive disagreement."
    new_conad_text = (
        "Contrastive methods such as CoLA \\cite{liu2022cola} and CONAD \\cite{xu2022conad} train local-to-subgraph mutual information estimators, "
        "scoring nodes by contrastive disagreement. However, due to PyGOD's CONAD implementation requiring full perturbed graph generation whose "
        "reproducibility was archived only as a preliminary diagnostic baseline without approved multi-seed canonical runs, CONAD is evaluated "
        "strictly as a diagnostic comparator rather than a primary benchmark model."
    )
    if old_conad_text in content:
        content = content.replace(old_conad_text, new_conad_text)

    # Update dynamic graph citation in Limitations (Section 7.2 / Section 6.2)
    old_dyn_text = "Dynamic graph models (e.g., TGN, JODIE) remain valuable future benchmarks."
    new_dyn_text = (
        "Dynamic graph models and continuous-time temporal architectures (e.g., TGN \\cite{rossi2020tgn}, JODIE \\cite{kumar2019jodie}, "
        "TGAT \\cite{xu2020tgat}, EvolveGCN \\cite{pareja2020evolvegcn}, DyRep \\cite{trivedi2019dyrep}) remain valuable future benchmarks."
    )
    if old_dyn_text in content:
        content = content.replace(old_dyn_text, new_dyn_text)

    # 5. Update Section 3.1 & Figure 1 Caption to match Code Trace
    old_dlg_aug_item = "\\item \\textbf{DLG-Aug} \\cite{park2026dlg}: Decoupled Local-to-Global model combining local ego-net pretraining with global propagation."
    new_dlg_aug_item = (
        "\\item \\textbf{DLG-Aug} \\cite{park2026dlg}: Decoupled Local-to-Global model combining finite-receptive local GCN pre-training "
        "with augmented global relational propagation."
    )
    if old_dlg_aug_item in content:
        content = content.replace(old_dlg_aug_item, new_dlg_aug_item)

    old_fig1_cap = (
        "\\caption{Decoupled Local-to-Global (DLG) anomaly detection architecture. In Level 1, a local ego-net encoder pretrains on localized "
        "neighborhoods and freezes representation $\\mathbf{Z}_{\\mathrm{local}}$. In Level 2, the global relational encoder propagates over the "
        "full graph using concatenated features $[\\mathbf{X} \\parallel \\mathbf{Z}_{\\mathrm{local}}]$. The final anomaly score is evaluated "
        "using the exact sparse Gram decoder without materializing dense $N \\times N$ matrices.}"
    )
    new_fig1_cap = (
        "\\caption{Decoupled Local-to-Global (DLG) anomaly detection architecture. In Level 1, a local GCN feature autoencoder pretrains on "
        "full-graph attributes using a finite 2-hop message-passing receptive field (without explicit ego-subgraph induction) and extracts frozen "
        "local representation $\\mathbf{Z}_{\\mathrm{local}}$. In Level 2, the global relational encoder processes the augmented representation "
        "$[\\mathbf{X} \\parallel \\mathbf{Z}_{\\mathrm{local}}]$ across graph topology $\\mathcal{E}$. The attribute decoder reconstructs "
        "original attributes $\\mathbf{X}$, while the structure decoder computes the unsquashed Gram matrix $\\hat{S} = \\mathbf{Z} \\mathbf{Z}^\\top$. "
        "Per-node anomaly scores combine Euclidean attribute and structural reconstruction residuals.}"
    )
    if old_fig1_cap in content:
        content = content.replace(old_fig1_cap, new_fig1_cap)

    # Add code-faithful Section 3.1 & 3.2 text detailing DLG-Base and DLG-Aug
    old_sec32_head = "\\subsection{Decoupled Local-to-Global Architecture}\nThe overall workflow of the Decoupled Local-to-Global (DLG) architecture is illustrated in Figure~\\ref{fig:dlg_architecture}."
    new_sec32_head = (
        "\\subsection{Decoupled Local-to-Global Architecture}\n"
        "The overall workflow of the Decoupled Local-to-Global (DLG) architecture is illustrated in Figure~\\ref{fig:dlg_architecture}. "
        "The framework encompasses two distinct variants: \\textit{DLG-Base} and \\textit{DLG-Aug} (DLG-Full).\n\n"
        "\\textbf{DLG-Base Module:} DLGBase couples a Level 1 local encoder and a Level 2 global encoder sequentially. "
        "Given node attributes $X \\in \\mathbb{R}^{N \\times D}$ and edge index $\\mathcal{E}$, the local encoder generates neighborhood-aware representation "
        "$Z_{\\mathrm{local}} = \\mathrm{GCN}_1(X, \\mathcal{E})$, which directly feeds the global encoder $Z_{\\mathrm{global}} = \\mathrm{GCN}_2(Z_{\\mathrm{local}}, \\mathcal{E})$. "
        "A learnable scalar parameter $\\alpha$ fuses both representations: "
        "$Z_{\\mathrm{fused}} = \\sigma(\\alpha) Z_{\\mathrm{local}} + (1 - \\sigma(\\alpha)) Z_{\\mathrm{global}}$. "
        "The attribute decoder reconstructs $\\hat{X} = \\mathrm{GCN}_{\\mathrm{dec}}(Z_{\\mathrm{fused}}, \\mathcal{E})$, and the structure decoder reconstructs adjacency "
        "via dot product $\\hat{S} = Z_{\\mathrm{fused}} Z_{\\mathrm{fused}}^\\top$.\n\n"
        "\\textbf{DLG-Aug (DLG-Full Pipeline):} In contrast to DLG-Base, DLG-Aug decouples the local and global stages into a two-phase training protocol. "
        "In Stage 1 (Local Pre-training), a separate 2-layer GCN autoencoder ($k=2$ message-passing receptive field) is trained on input attributes $X$ "
        "by minimizing mean squared feature reconstruction error $\\|X - \\hat{X}_{\\mathrm{L1}}\\|_F^2$. Crucially, this pre-training operates in full-graph forward "
        "passes without explicit ego-net subgraph extraction. Once trained, its parameters are discarded, and the extracted per-node embeddings "
        "$Z_{\\mathrm{local}} \\in \\mathbb{R}^{N \\times d}$ are detached and frozen. "
        "In Stage 2 (Global Encoding), the input feature matrix is augmented via concatenation $\\tilde{X} = [X \\parallel Z_{\\mathrm{local}}] \\in \\mathbb{R}^{N \\times (D + d)}$. "
        "The global GCN encoder processes $\\tilde{X}$ over $\\mathcal{E}$ to produce latent embedding $Z \\in \\mathbb{R}^{N \\times d}$. "
        "The attribute decoder then reconstructs the \\textit{original} attributes $\\hat{X} \\in \\mathbb{R}^{N \\times D}$ (not the augmented $\\tilde{X}$), "
        "while the structure decoder computes $\\hat{S} = Z Z^\\top$."
    )
    if old_sec32_head in content:
        content = content.replace(old_sec32_head, new_sec32_head)

    # 6. Update Section 3.3 regarding Training Objective vs Anomaly Scoring Formula
    old_scoring_text = "where $\\mathrm{deg}(i) = \\sum_j A_{ij}^2 = \\sum_j A_{ij}$ for unweighted binary graphs. Summing over all $N$ nodes yields:"
    new_scoring_text = (
        "where $\\mathrm{deg}(i) = \\sum_j A_{ij}^2 = \\sum_j A_{ij}$ for unweighted binary graphs. "
        "At inference, following standard PyGOD semantics, the per-node structural anomaly score $\\mathcal{A}_{\\mathrm{struct}, i}$ is the Euclidean $L_2$ norm "
        "of the reconstruction residual vector: $\\mathcal{A}_{\\mathrm{struct}, i} = \\|A_{i,:} - \\hat{S}_{i,:}\\|_2 = \\sqrt{\\mathrm{deg}(i) - 2 \\mathbf{z}_i^\\top (A \\mathbf{Z})_i + \\mathbf{z}_i^\\top (\\mathbf{Z}^\\top \\mathbf{Z}) \\mathbf{z}_i}$. "
        "The composite node anomaly score combines attribute and structural residuals: $\\mathcal{A}_i = w \\|X_i - \\hat{X}_i\\|_2 + (1 - w) \\mathcal{A}_{\\mathrm{struct}, i}$ ($w=0.5$). "
        "During model training, summing squared errors over all $N$ nodes yields the global objective:"
    )
    if old_scoring_text in content:
        content = content.replace(old_scoring_text, new_scoring_text)

    # 7. Update Section 4.2 with Wilcoxon sample size property
    old_wilcoxon = "For pairwise comparisons against DLG-Aug, we report two-sided Wilcoxon signed-rank tests adjusted via the Holm step-down procedure \\cite{holm1979}."
    new_wilcoxon = (
        "For pairwise comparisons against DLG-Aug, we report two-sided Wilcoxon signed-rank tests adjusted via the Holm step-down procedure \\cite{holm1979}. "
        "We note that for within-dataset comparisons across $N_{\\mathrm{seeds}} = 5$, the minimum achievable two-sided Wilcoxon signed-rank $p$-value is "
        "$2^{-(5-1)} = 0.0625$ when all five differences share the same nonzero sign. Consequently, no single dataset comparison can attain conventional "
        "$\\alpha = 0.05$ significance under five seeds alone; statistical hypothesis testing is therefore evaluated across the benchmark suite via the Friedman permutation framework."
    )
    if old_wilcoxon in content:
        content = content.replace(old_wilcoxon, new_wilcoxon)

    # 8. Replace Table 3 with verified Table 3
    # Look for \label{tab:alert_budget} block
    t3_pattern = re.compile(
        r"(\\begin\{table\}\[htbp\].*?\\caption\{Operational Alert-Budget Performance.*?\\label\{tab:alert_budget\}\s*)(.*?)(%?.*?\\end\{tabularx\}\s*\\end\{table\})",
        re.DOTALL
    )
    match_t3 = t3_pattern.search(content)
    if match_t3:
        header_part = match_t3.group(1)
        # Extract \caption and \label and wrap table3_body
        new_t3_full = f"{header_part}{table3_body}\n\\end{{table}}"
        # Replace the entire matched block
        content = content[:match_t3.start()] + new_t3_full + content[match_t3.end():]
        print("  -> Successfully replaced Table 3 with verified data.")
    else:
        print("  WARNING: Could not match Table 3 pattern!")

    # 9. Update Section 5.2 narrative and Table 7 based on verified Table 3
    # Replace obsolete Table 7 alert-budget row and Elliptic +50.97% row
    old_t7_row_elliptic = "Clustered transaction networks with extreme imbalance (e.g., Elliptic) & \\textbf{DLG-Aug} (achieves highest AP on clustered transaction topology; $+50.97\\%$ relative over DLG-Base)"
    new_t7_row_elliptic = "Transaction networks with extreme imbalance (e.g., Elliptic) & \\textbf{DLG-Aug} (achieves highest AP, $\\Delta\\mathrm{AP} = +0.0350$ absolute uplift over DLG-Base, surpassing test prevalence $0.1037 > 0.0976$)"
    if old_t7_row_elliptic in content:
        content = content.replace(old_t7_row_elliptic, new_t7_row_elliptic)

    old_t7_row_alert = "Strict Alert Budgets ($P@1\\%$ prioritization in AML triage) & \\textbf{AnomalyDAE / DLG-Aug} & \\textbf{DOMINANT} & OCGNN, CoLA (insufficient precision at head of alert list) \\\\"
    new_t7_row_alert = (
        "Strict Alert Budgets ($P@1\\%$ prioritization in AML triage) & "
        "\\textbf{AnomalyDAE} (leads Precision@1\\% on BSC and Polygon) & "
        "\\textbf{DOMINANT / DLG-Aug} (DOMINANT leads on Ethereum; DLG-Aug competitive across all three) & "
        "CoLA, GADNR (insufficient top-rank precision on low-prevalence chains; descriptive of tested graph settings) \\\\"
    )
    if old_t7_row_alert in content:
        content = content.replace(old_t7_row_alert, new_t7_row_alert)

    # 10. Soften causal phrases
    content = content.replace("dense social networks and bipartite cyber graphs", "dense social networks and cyber authentication graphs")
    content = content.replace("dense community structure exhibits negative delta.", "negative delta observed on dense social interaction graph.")
    content = content.replace("bipartite cyber authentication graph.", "cyber authentication event graph.")
    content = content.replace(
        "While DLG-Aug yields valuable improvements on graphs with pronounced local clustering (Elliptic $+0.0350$, Ethereum $+0.0253$), it causes severe degradation on dense social networks (Reddit-Syn $-0.0656$) and cyber authentication bipartite graphs (LANL-RedTeam $-0.0252$).",
        "While DLG-Aug yields valuable improvements on transaction graphs like Elliptic ($\\Delta\\mathrm{AP} = +0.0350$) and Ethereum ($\\Delta\\mathrm{AP} = +0.0253$), it exhibits lower performance on Reddit-Syn ($\\Delta\\mathrm{AP} = -0.0656$) and LANL-RedTeam ($\\Delta\\mathrm{AP} = -0.0252$)."
    )
    content = content.replace("Dense interaction graphs with strong community structure (e.g., Reddit)", "Dense interaction graphs (e.g., Reddit-Syn)")
    content = content.replace("Bipartite cyber authentication logs (e.g., LANL)", "Cyber authentication logs (e.g., LANL-RedTeam)")
    content = content.replace("negative delta on bipartite authentication graphs", "negative delta on authentication event graphs")
    content = content.replace("clustered transaction graphs such as Elliptic", "transaction graphs such as Elliptic")
    content = content.replace("bipartite cyber graphs", "cyber authentication graphs")

    # 11. Replace Table A2 with verified Table A2
    # Look for \label{tab:app_f1_full} block
    ta2_pattern = re.compile(
        r"(\\begin\{table\}\[H\].*?\\label\{tab:app_f1_full\}\s*\\begin\{tabular\*\}\{\\textwidth\}\{@\{\\extracolsep\{\\fill\}\}lrrrrrrr\}\\toprule\s*\\textbf\{Dataset\}.*?\\midrule\s*)(.*?)(\\bottomrule\s*\\end\{tabular\*\}\s*\\end\{table\})",
        re.DOTALL
    )
    match_ta2 = ta2_pattern.search(content)
    if match_ta2:
        top_part = match_ta2.group(1)
        bottom_part = match_ta2.group(3)
        # table_a2_rows already has \midrule and \bottomrule
        # Let's inspect format:
        new_ta2_full = f"{top_part}{table_a2_rows}\n\\end{{tabular*}}\n\\end{{table}}"
        content = content[:match_ta2.start()] + new_ta2_full + content[match_ta2.end():]
        print("  -> Successfully replaced Table A2 with verified data.")
    else:
        print("  WARNING: Could not match Table A2 pattern!")

    # Write back
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  -> Finished updating {tex_path.name}")


if __name__ == "__main__":
    for p in PREPRINTS_TEX_FILES:
        if p.exists():
            update_manuscript(p, is_mdpi=False)
    for p in APPLSCI_TEX_FILES:
        if p.exists():
            update_manuscript(p, is_mdpi=True)
    print("=== Manuscript Update Completed ===")
