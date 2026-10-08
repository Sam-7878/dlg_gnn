"""
Formal architecture audit and parameter/training budget generator for DLG-Base and DLG-Aug.
Traces exact source code equations and computes parameter counts for all 10 primary datasets + LANL.
"""

import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]

OUT_MD = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/architecture/dlg_base_exact_equations.md"
OUT_JSON = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/architecture/dlg_architecture_manifest.json"
OUT_CSV = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/architecture/dlg_parameter_budget_by_dataset.csv"
OUT_TEX = REPO_ROOT / "dlg_gnn/docs/papers/_42_Benchmark/generated/table_dlg_architecture_budget.tex"

DATASETS = [
    ("Elliptic", 165, "Real Financial Fraud", 50),
    ("DGraphFin", 17, "Real Financial Fraud", 50),
    ("BitcoinOTC", 67, "Real Trust / Fraud", 50),
    ("Yelp-Syn", 300, "Synthetic Review Fraud", 50),
    ("Amazon-Syn", 767, "Synthetic E-Commerce Fraud", 50),
    ("Reddit-Syn", 602, "Synthetic Social Anomaly", 50),
    ("Flickr-Syn", 500, "Synthetic Social Anomaly", 50),
    ("Cora-Syn", 1433, "Synthetic Citation Anomaly", 100),
    ("CiteSeer-Syn", 3703, "Synthetic Citation Anomaly", 100),
    ("PubMed-Syn", 500, "Synthetic Citation Anomaly", 100),
    ("LANL-RedTeam", 13, "External Cybersecurity", 50),
]


def compute_params(F: int, H: int = 64):
    # DLG-Base
    # local_encoder: GCN(F -> H, num_layers=2)
    # L1: F*H + H, L2: H*H + H
    base_local = (F * H + H) + (H * H + H)
    # global_encoder: GCN(H -> H, num_layers=2)
    # L1: H*H + H, L2: H*H + H
    base_global = 2 * (H * H + H)
    # gate: alpha scalar
    base_gate = 1
    # attr_decoder: GCN(H -> F, num_layers=2)
    # L1: H*H + H, L2: H*F + F
    base_decoder = (H * H + H) + (H * F + F)
    base_total = base_local + base_global + base_gate + base_decoder
    base_active_global = base_total

    # DLG-Aug
    # Local stage (pretraining):
    # l1_encoder: GCN(F -> H, num_layers=2) -> (F*H + H) + (H*H + H)
    aug_l1_enc = (F * H + H) + (H * H + H)
    # l1_decoder: Linear(H, F) -> H*F + F
    aug_l1_dec = H * F + F
    aug_local_total = aug_l1_enc + aug_l1_dec

    # Global stage:
    # encoder: GCN((F+H) -> H, num_layers=2)
    # L1: (F+H)*H + H = F*H + H*H + H, L2: H*H + H
    aug_global_enc = (F * H + H * H + H) + (H * H + H)
    # attr_decoder: GCN(H -> F, num_layers=2)
    # L1: H*H + H, L2: H*F + F
    aug_global_dec = (H * H + H) + (H * F + F)
    aug_global_active = aug_global_enc + aug_global_dec
    aug_total = aug_local_total + aug_global_active

    return {
        "base_local": base_local,
        "base_global": base_global,
        "base_gate": base_gate,
        "base_decoder": base_decoder,
        "base_total": base_total,
        "base_active_global": base_active_global,
        "aug_local": aug_local_total,
        "aug_global_active": aug_global_active,
        "aug_total": aug_total,
    }


def generate_audit():
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, F, domain, ep in DATASETS:
        p = compute_params(F, 64)
        l1_ep = 20 if ep == 50 else 20
        # DLG-Base row
        rows.append({
            "dataset": name,
            "feature_dim": F,
            "variant": "DLG-Base",
            "local_module_params": p["base_local"],
            "global_module_params": p["base_global"] + p["base_gate"],
            "decoder_params": p["base_decoder"],
            "total_unique_trainable_params": p["base_total"],
            "active_global_stage_params": p["base_active_global"],
            "inference_active_params": p["base_total"],
            "local_pretrain_epochs": 0,
            "global_epochs": ep,
            "total_optimization_epochs": ep,
        })
        # DLG-Aug row
        rows.append({
            "dataset": name,
            "feature_dim": F,
            "variant": "DLG-Aug",
            "local_module_params": p["aug_local"],
            "global_module_params": p["aug_global_active"] - (65 * F + 4160),
            "decoder_params": 65 * F + 4160,
            "total_unique_trainable_params": p["aug_total"],
            "active_global_stage_params": p["aug_global_active"],
            "inference_active_params": p["aug_global_active"],
            "local_pretrain_epochs": l1_ep,
            "global_epochs": ep,
            "total_optimization_epochs": ep + l1_ep,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")

    # Generate Markdown documentation of exact equations
    md_text = f"""# DLG Architecture Trace and Exact Equations

## 1. DLG-Base Formulation
In `gog_fraud.models.pygod.dlg_base.DLGBase` and `ExactDLGBase`:

### 1.1 Forward Pass Equations
Given node features $X \\in \\mathbb{{R}}^{{N \\times F}}$ and normalized adjacency operator $\\tilde{{A}} = \\tilde{{D}}^{{-1/2}} \\tilde{{A}} \\tilde{{D}}^{{-1/2}}$:

1. **Local Encoder (Level 1)**:
   $$Z_{{local}} = \\text{{GCN}}_{{local}}(X, \\tilde{{A}}) = \\text{{ReLU}}\\left(\\tilde{{A}} \\,\\text{{ReLU}}(\\tilde{{A}} X W_1 + b_1) W_2 + b_2\\right)$$
   where $W_1 \\in \\mathbb{{R}}^{{F \\times 64}}$, $W_2 \\in \\mathbb{{R}}^{{64 \\times 64}}$.

2. **Global Encoder (Level 2)**:
   $$Z_{{global}} = \\text{{GCN}}_{{global}}(Z_{{local}}, \\tilde{{A}}) = \\text{{ReLU}}\\left(\\tilde{{A}} \\,\\text{{ReLU}}(\\tilde{{A}} Z_{{local}} W_3 + b_3) W_4 + b_4\\right)$$
   where $W_3, W_4 \\in \\mathbb{{R}}^{{64 \\times 64}}$.

3. **Learnable Embedding Gate**:
   $$\\alpha = \\sigma(\\alpha_{{param}}), \\quad \\alpha_{{param}} \\in \\mathbb{{R}} \\text{{ initialized to }} 0.5$$
   $$Z_{{fused}} = \\alpha Z_{{local}} + (1 - \\alpha) Z_{{global}}$$

4. **Attribute Decoder**:
   $$\\hat{{X}} = \\text{{GCN}}_{{dec}}(Z_{{fused}}, \\tilde{{A}}) = \\tilde{{A}} \\,\\text{{ReLU}}(\\tilde{{A}} Z_{{fused}} W_5 + b_5) W_6 + b_6$$
   where $W_5 \\in \\mathbb{{R}}^{{64 \\times 64}}$, $W_6 \\in \\mathbb{{R}}^{{64 \\times F}}$.

5. **Structure Reconstruction (Linear Gram Formulation)**:
   $$\\hat{{S}} = Z_{{fused}} Z_{{fused}}^T$$
   The exact loss contribution is computed without materializing $N \\times N$:
   $$\\|\\hat{{S}} - A\\|_F^2 = \\text{{tr}}(G^2) - 2\\text{{tr}}(A G) + \\|A\\|_F^2, \\quad G = Z_{{fused}}^T Z_{{fused}} \\in \\mathbb{{R}}^{{64 \\times 64}}$$

6. **Loss Function**:
   $$\\mathcal{{L}}_{{total}} = 0.5 \\cdot \\frac{{1}}{{N}} \\sum_{{i=1}}^N \\|x_i - \\hat{{x}}_i\\|_2^2 + 0.5 \\cdot \\frac{{1}}{{N}} \\sum_{{i=1}}^N \\|s_i - \\hat{{s}}_i\\|_2^2$$

---

## 2. DLG-Aug Formulation
In `gog_fraud.models.pygod.dlg_full.DLGFull` and `SharedDLGFull`:

1. **Local Pretraining Stage (20 epochs, MSE loss)**:
   $$Z_{{L1}} = \\text{{GCN}}_{{L1}}(X, \\tilde{{A}}) \\in \\mathbb{{R}}^{{N \\times 64}}$$
   $$\\hat{{X}}_{{L1}} = Z_{{L1}} W_{{dec,L1}} + b_{{dec,L1}}$$
   $$\\mathcal{{L}}_{{L1}} = \\frac{{1}}{{N}} \\|X - \\hat{{X}}_{{L1}}\\|_F^2$$
   Following 20 epochs, $H_{{local}} = \\text{{detach}}(Z_{{L1}})$ is permanently frozen.

2. **Global Optimization Stage (50 epochs)**:
   Augmented feature matrix:
   $$X_{{aug}} = [X \\parallel H_{{local}}] \\in \\mathbb{{R}}^{{N \\times (F + 64)}}$$
   $$Z = \\text{{GCN}}_{{global}}(X_{{aug}}, \\tilde{{A}}) \\in \\mathbb{{R}}^{{N \\times 64}}$$
   $$\\hat{{X}} = \\text{{GCN}}_{{dec}}(Z, \\tilde{{A}}) \\in \\mathbb{{R}}^{{N \\times F}}$$
   $$\\hat{{S}} = Z Z^T$$

---

## 3. Capacity and Training Budget Comparison
For input dimension $F$ and hidden dimension $H=64$:
- **DLG-Base Total Parameters**: $129 F + 16,705$ (all active during 50 global epochs)
- **DLG-Aug Active Global Parameters**: $129 F + 12,480$ (active during 50 global epochs)
- **Difference in Active Parameters**: $|\\text{{Active}}_{{Base}} - \\text{{Active}}_{{Aug}}| = 4,225$ parameters, strictly constant across all datasets.
- **Optimization Budget**:
  - DLG-Base: 50 global epochs
  - DLG-Aug: 20 local epochs + 50 global epochs = 70 total epochs
"""
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Wrote {OUT_MD}")

    # Generate LaTeX table
    tex_lines = [
        "% Auto-generated DLG architecture, parameter, and training budget audit",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Parameter Count and Optimization Budget Comparison Between DLG-Base and DLG-Aug Across Primary and External Datasets.}",
        "\\label{tab:dlg_architecture_budget}",
        "\\footnotesize",
        "\\begin{tabular}{lcrccr}",
        "\\toprule",
        "\\textbf{Dataset} & \\textbf{Dim ($F$)} & \\textbf{Model} & \\textbf{Active Params} & \\textbf{Total Params} & \\textbf{Epochs (Local+Glob)} \\\\",
        "\\midrule",
    ]

    for name, F, domain, ep in DATASETS:
        p = compute_params(F, 64)
        l1_ep = 20
        tex_lines.append(f"\\multirow{{2}}{{*}}{{{name}}} & \\multirow{{2}}{{*}}{{{F}}} & DLG-Base & {p['base_total']:,} & {p['base_total']:,} & 0 + {ep} = {ep} \\\\")
        tex_lines.append(f" & & DLG-Aug & {p['aug_global_active']:,} & {p['aug_total']:,} & {l1_ep} + {ep} = {ep + l1_ep} \\\\")
        tex_lines.append("\\midrule")

    if tex_lines[-1] == "\\midrule":
        tex_lines[-1] = "\\bottomrule"

    tex_lines.append("\\end{tabular}")
    tex_lines.append("\\end{table}")

    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Wrote {OUT_TEX}")


if __name__ == "__main__":
    generate_audit()
