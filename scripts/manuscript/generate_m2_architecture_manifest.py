#!/usr/bin/env python3
"""
generate_m2_architecture_manifest.py

Verifies PyTorch model parameters directly from source code:
- DLGBase (gog_fraud.models.pygod.dlg_base)
- DLGFull / DLGFullBase (gog_fraud.models.pygod.dlg_full / dlg_full_base)
- ExactDLGBase / ExactDLGFullBase (gog_fraud.models.pygod.shared_reconstruction)

Generates:
1. dlg_architecture_source_trace.json
2. table_dlg_architecture_budget_m2.tex
3. dlg_architecture_equations_m2.tex
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
import torch.nn as nn
from torch_geometric.nn import GCN

from gog_fraud.models.pygod.dlg_base import DLGBase
from gog_fraud.models.pygod.dlg_full_base import DLGFullBase

DATASET_CONFIGS = [
    {"name": "Elliptic", "F": 165, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "DGraphFin", "F": 17, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "BitcoinOTC", "F": 67, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "Yelp-Syn", "F": 300, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "Amazon-Syn", "F": 767, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "Reddit-Syn", "F": 602, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "Flickr-Syn", "F": 500, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
    {"name": "Cora-Syn", "F": 1433, "epochs_base": "0 + 100 = 100", "epochs_aug": "20 + 100 = 120"},
    {"name": "CiteSeer-Syn", "F": 3703, "epochs_base": "0 + 100 = 100", "epochs_aug": "20 + 100 = 120"},
    {"name": "PubMed-Syn", "F": 500, "epochs_base": "0 + 100 = 100", "epochs_aug": "20 + 100 = 120"},
    {"name": "LANL-RedTeam", "F": 13, "epochs_base": "0 + 50 = 50", "epochs_aug": "20 + 50 = 70"},
]

def analyze_models():
    records = []
    
    for cfg in DATASET_CONFIGS:
        F_dim = cfg["F"]
        name = cfg["name"]
        
        # 1. DLGBase
        base_model = DLGBase(in_dim=F_dim, hid_dim=64, num_layers=4)
        base_active = sum(p.numel() for p in base_model.parameters())
        base_total = base_active
        
        # Component breakdown for DLGBase
        local_enc_p = sum(p.numel() for p in base_model.local_encoder.parameters())
        global_enc_p = sum(p.numel() for p in base_model.global_encoder.parameters())
        gate_p = sum(p.numel() for p in [base_model.alpha])
        decoder_p = sum(p.numel() for p in base_model.attr_decoder.parameters())
        
        # 2. DLG-Aug
        # Stage 1: Local Pretraining
        l1_enc = GCN(in_channels=F_dim, hidden_channels=64, num_layers=2, out_channels=64)
        l1_dec = nn.Linear(64, F_dim)
        l1_p = sum(p.numel() for p in l1_enc.parameters()) + sum(p.numel() for p in l1_dec.parameters())
        
        # Stage 2: Global GCN on augmented input (F + 64)
        l2_model = DLGFullBase(in_dim=F_dim + 64, orig_dim=F_dim, hid_dim=64, num_layers=4)
        aug_active = sum(p.numel() for p in l2_model.parameters())
        aug_total = l1_p + aug_active
        
        # Component breakdown for DLG-Aug
        l2_enc_p = sum(p.numel() for p in l2_model.encoder.parameters())
        l2_dec_p = sum(p.numel() for p in l2_model.attr_decoder.parameters())
        
        records.append({
            "dataset": name,
            "F": F_dim,
            "base_active_params": base_active,
            "base_total_params": base_total,
            "base_epochs": cfg["epochs_base"],
            "base_components": {
                "local_encoder_params": local_enc_p,
                "global_encoder_params": global_enc_p,
                "gate_params": gate_p,
                "attr_decoder_params": decoder_p,
                "struct_decoder_params": 0
            },
            "aug_l1_pretrain_params": l1_p,
            "aug_active_params": aug_active,
            "aug_total_params": aug_total,
            "aug_epochs": cfg["epochs_aug"],
            "aug_components": {
                "l1_encoder_params": sum(p.numel() for p in l1_enc.parameters()),
                "l1_linear_decoder_params": sum(p.numel() for p in l1_dec.parameters()),
                "l2_global_encoder_params": l2_enc_p,
                "l2_attr_decoder_params": l2_dec_p,
                "l2_struct_decoder_params": 0
            }
        })
    
    return records

def generate_latex_table(records):
    lines = [
        r"% Auto-generated DLG architecture, parameter, and training budget audit (M2 Canonical)",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Parameter Count and Optimization Budget Comparison Between DLG-Base and DLG-Aug Across Primary and External Datasets.}",
        r"\label{tab:dlg_architecture_budget}",
        r"\footnotesize",
        r"\begin{tabular}{lcrccr}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Dim ($F$)} & \textbf{Model} & \textbf{Active Params} & \textbf{Total Params} & \textbf{Epochs (Local+Glob)} \\",
        r"\midrule"
    ]
    
    for rec in records:
        name = rec["dataset"]
        F_dim = rec["F"]
        b_act = f"{rec['base_active_params']:,}"
        b_tot = f"{rec['base_total_params']:,}"
        b_ep = rec["base_epochs"]
        
        a_act = f"{rec['aug_active_params']:,}"
        a_tot = f"{rec['aug_total_params']:,}"
        a_ep = rec["aug_epochs"]
        
        lines.append(f"\\multirow{{2}}{{*}}{{{name}}} & \\multirow{{2}}{{*}}{{{F_dim}}} & DLG-Base & {b_act} & {b_tot} & {b_ep} \\\\")
        lines.append(f" & & DLG-Aug & {a_act} & {a_tot} & {a_ep} \\\\")
        lines.append(r"\midrule" if rec != records[-1] else r"\bottomrule")
    
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)

def generate_latex_equations():
    doc = r"""% DLG Architecture Equations (Exact Source Code Truth: Sequential 2-Level GCN, Scalar Gate, 2-Layer Decoders)
% Model 1: DLG-Base (gog_fraud.models.pygod.dlg_base.DLGBase)
\begin{align}
    H_{\mathrm{loc}} &= \mathrm{GCN}_{\mathrm{loc}}(X, A) \in \mathbb{R}^{N\times d}, \label{eq:dlg_base_local} \\
    H_{\mathrm{glob}} &= \mathrm{GCN}_{\mathrm{glob}}(H_{\mathrm{loc}}, A) \in \mathbb{R}^{N\times d}, \label{eq:dlg_base_global} \\
    \alpha &= \sigma(\alpha_{\mathrm{param}}) \in (0, 1), \quad \alpha_{\mathrm{param}} \in \mathbb{R}, \label{eq:dlg_base_gate} \\
    Z &= \alpha H_{\mathrm{loc}} + (1 - \alpha) H_{\mathrm{glob}} \in \mathbb{R}^{N\times d}, \label{eq:dlg_base_fused} \\
    \hat{X} &= \mathrm{GCN}_{\mathrm{dec}}(Z, A) \in \mathbb{R}^{N\times F}, \label{eq:dlg_base_attr_dec} \\
    \hat{A} &= Z Z^\top \in \mathbb{R}^{N\times N}. \label{eq:dlg_base_struct_dec}
\end{align}

% Model 2: DLG-Aug (gog_fraud.models.pygod.dlg_full.DLGFull)
% Stage 1: Local Pretraining (20 epochs, then detached and frozen)
\begin{align}
    H^{\mathrm{loc}} &= \mathrm{GCN}_{\mathrm{loc}}(X, A) \in \mathbb{R}^{N\times D}, \label{eq:dlg_aug_local_enc} \\
    \hat{X}^{\mathrm{loc}} &= H^{\mathrm{loc}} W_{\mathrm{loc}} + b_{\mathrm{loc}} \in \mathbb{R}^{N\times F}, \label{eq:dlg_aug_local_dec} \\
    \mathcal{L}_{\mathrm{loc}} &= \frac{1}{NF} \left\|X - \hat{X}^{\mathrm{loc}}\right\|_F^2. \label{eq:dlg_aug_local_loss}
\end{align}

% Stage 2: Global Training on Augmented Attributes (50 or 100 epochs)
\begin{align}
    X^{\mathrm{aug}} &= \left[X \,\|\, H^{\mathrm{loc}}\right] \in \mathbb{R}^{N\times(F+D)}, \label{eq:dlg_aug_concat} \\
    Z &= \mathrm{GCN}_{\mathrm{glob}}(X^{\mathrm{aug}}, A) \in \mathbb{R}^{N\times d}, \label{eq:dlg_aug_global_enc} \\
    \hat{X} &= \mathrm{GCN}_{\mathrm{dec}}(Z, A) \in \mathbb{R}^{N\times F}, \label{eq:dlg_aug_attr_dec} \\
    \hat{A} &= Z Z^\top \in \mathbb{R}^{N\times N}. \label{eq:dlg_aug_struct_dec}
\end{align}

% Anomaly Scoring Formula (Shared by both models)
\begin{align}
    e_i^{x} &= \left\|X_i - \hat{X}_i\right\|_2, \quad e_i^{a} = \left\|A_i - \hat{A}_i\right\|_2, \label{eq:dlg_residuals} \\
    s_i &= \lambda e_i^{x} + (1 - \lambda) e_i^{a}, \quad \lambda = 0.5. \label{eq:dlg_anomaly_score}
\end{align}
"""
    return doc

def main():
    records = analyze_models()
    
    out_dir = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m2" / "architecture"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Source trace JSON
    trace_path = out_dir / "dlg_architecture_source_trace.json"
    trace_data = {
        "description": "Exact architecture source trace and parameter budget across DLG models",
        "formulas": {
            "DLG-Base": {
                "local_encoder": "GCN(in=F, hid=64, layers=2, out=64) -> 64*F + 4224 params",
                "global_encoder": "GCN(in=64, hid=64, layers=2, out=64) -> 8320 params",
                "gate": "nn.Parameter(scalar) -> 1 param",
                "attr_decoder": "GCN(in=64, hid=64, layers=2, out=F) -> 65*F + 4160 params",
                "struct_decoder": "DotProduct -> 0 params",
                "total_active_formula": "129*F + 16705"
            },
            "DLG-Aug": {
                "stage1_l1_pretrain": {
                    "l1_encoder": "GCN(in=F, hid=64, layers=2, out=64) -> 64*F + 4224 params",
                    "l1_decoder": "Linear(in=64, out=F) -> 65*F params",
                    "subtotal": "129*F + 4224 params (trained 20 epochs, then frozen)"
                },
                "stage2_l2_global": {
                    "encoder": "GCN(in=F+64, hid=64, layers=2, out=64) -> 64*F + 8320 params",
                    "attr_decoder": "GCN(in=64, hid=64, layers=2, out=F) -> 65*F + 4160 params",
                    "struct_decoder": "DotProduct -> 0 params",
                    "active_formula": "129*F + 12480"
                },
                "total_pipeline_formula": "258*F + 16704"
            }
        },
        "datasets": records
    }
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump(trace_data, f, indent=2)
    print(f"Wrote {trace_path}")
    
    # 2. LaTeX Table
    tex_table = generate_latex_table(records)
    table_path = out_dir / "table_dlg_architecture_budget_m2.tex"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(tex_table)
    print(f"Wrote {table_path}")
    
    # Also update generated/ in manuscript folder
    paper_gen_path = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "generated" / "table_dlg_architecture_budget.tex"
    paper_gen_path.parent.mkdir(parents=True, exist_ok=True)
    with open(paper_gen_path, "w", encoding="utf-8") as f:
        f.write(tex_table)
    print(f"Updated manuscript table at {paper_gen_path}")
    
    # 3. LaTeX Equations snippet
    eq_path = out_dir / "dlg_architecture_equations_m2.tex"
    with open(eq_path, "w", encoding="utf-8") as f:
        f.write(generate_latex_equations())
    print(f"Wrote {eq_path}")

if __name__ == "__main__":
    main()
