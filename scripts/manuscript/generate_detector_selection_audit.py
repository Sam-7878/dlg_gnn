"""
Generate official PyGOD 1.1.0 detector candidate universe and selection audit table.
Evaluates all 16 official detectors against objective-family coverage, task compatibility,
and benchmark suitability.
"""

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_CSV = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/detector_selection/pygod11_detector_selection_audit.csv"

# Official PyGOD 1.1.0 detector universe
DETECTORS = [
    {
        "detector": "SCAN",
        "objective_family": "structural_clustering",
        "is_deep_gnn": False,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Non-deep structural clustering heuristic; does not utilize node attributes or neural representation learning.",
    },
    {
        "detector": "Radar",
        "objective_family": "matrix_factorization",
        "is_deep_gnn": False,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Shallow matrix factorization framework; lacks multi-layer GNN propagation and relational message passing.",
    },
    {
        "detector": "ANOMALOUS",
        "objective_family": "matrix_factorization_joint",
        "is_deep_gnn": False,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Joint attribute-structure matrix regression; non-deep method superseded by graph autoencoders on large graphs.",
    },
    {
        "detector": "ONE",
        "objective_family": "matrix_factorization_outlier",
        "is_deep_gnn": False,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Outlier-aware matrix factorization heuristic; does not evaluate GNN inductive message passing.",
    },
    {
        "detector": "GAE",
        "objective_family": "vanilla_reconstruction",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Reconstructs structure only; strictly subsumed by DOMINANT which performs joint attribute and structure reconstruction.",
    },
    {
        "detector": "DONE",
        "objective_family": "deep_autoencoder_mlp",
        "is_deep_gnn": False,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Deep MLP autoencoders for structure and attributes without graph convolutional message passing.",
    },
    {
        "detector": "AdONE",
        "objective_family": "adversarial_deep_autoencoder",
        "is_deep_gnn": False,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Adversarial MLP autoencoder variation of DONE; lacks relational neighborhood aggregation.",
    },
    {
        "detector": "DOMINANT",
        "objective_family": "reconstruction_gae",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "INCLUDED",
        "primary_reason": "Primary canonical representative of joint GCN attribute and structure reconstruction GAE lineage (Ding et al., SDM 2019).",
    },
    {
        "detector": "AnomalyDAE",
        "objective_family": "dual_nonlinear_reconstruction",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "INCLUDED",
        "primary_reason": "Canonical representative of dual nonlinear architecture (structure-guided GAT + attribute-guided dense decoder) (Fan et al., ICASSP 2020).",
    },
    {
        "detector": "GAAN",
        "objective_family": "generative_adversarial",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "High training instability on million-edge sparse transaction graphs; redundant with reconstruction and contrastive objectives.",
    },
    {
        "detector": "DMGD",
        "objective_family": "density_manifold",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Evaluates multi-view density estimation; computationally intractable full-graph manifold distance computation.",
    },
    {
        "detector": "OCGNN",
        "objective_family": "one_class_boundary",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "INCLUDED",
        "primary_reason": "Canonical representative of one-class hypersphere boundary learning in graph representation space (Wang et al., NCA 2021).",
    },
    {
        "detector": "CoLA",
        "objective_family": "contrastive_ssl",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "INCLUDED",
        "primary_reason": "Canonical representative of contrastive self-supervised learning between target nodes and ego-net subgraphs (Liu et al., TNNLS 2022).",
    },
    {
        "detector": "GUIDE",
        "objective_family": "structure_motif_autoencoder",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "EXCLUDED",
        "primary_reason": "Requires higher-order motif/graphlet pre-computation which does not scale to million-edge graphs like Reddit-Syn or DGraphFin.",
    },
    {
        "detector": "CONAD",
        "objective_family": "contrastive_reconstruction_hybrid",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "INCLUDED",
        "primary_reason": "Representative of hybrid contrastive data augmentation combined with GAE reconstruction loss (Xu et al., PAKDD 2022).",
    },
    {
        "detector": "GADNR",
        "objective_family": "neighborhood_distribution_reconstruction",
        "is_deep_gnn": True,
        "unsupervised": True,
        "static_attributed_compatible": True,
        "in_pygod_1_1": True,
        "selected_status": "INCLUDED",
        "primary_reason": "State-of-the-art representative of neighborhood feature-distribution and degree reconstruction (Roy et al., WSDM 2024).",
    },
]

# Non-PyGOD 1.1.0 models mentioned in review
LITERATURE_VERIFICATIONS = [
    {
        "name": "GWAE",
        "full_name": "Graph Wasserstein Autoencoder",
        "venue_year": "IEEE TKDE 2023",
        "in_pygod_1_1": False,
        "status_note": "Wasserstein distance optimal transport autoencoder; not implemented in official PyGOD 1.1.0 release.",
    },
    {
        "name": "SL-GAD",
        "full_name": "Self-Supervised Learning for Graph Anomaly Detection",
        "venue_year": "IJCAI 2021 (Zheng et al.)",
        "in_pygod_1_1": False,
        "status_note": "Generative/contrastive GAD framework; not part of PyGOD 1.1.0 core detectors.",
    },
    {
        "name": "TAM",
        "full_name": "Time-Aware Memory / Temporal Graph Anomaly Modeling",
        "venue_year": "ACM SIGKDD / WSDM 2023",
        "in_pygod_1_1": False,
        "status_note": "Dynamic/temporal graph method; outside the scope of static attributed node anomaly detection.",
    },
    {
        "name": "CARE-GNN",
        "full_name": "Camouflage-Resistant GNN for Fraud Detection",
        "venue_year": "ACM CIKM 2020 (Dou et al.)",
        "in_pygod_1_1": False,
        "status_note": "Semi-supervised fraud detector with label-guided RL neighbor selection; not unsupervised.",
    },
    {
        "name": "SemiGNN",
        "full_name": "Semi-Supervised Graph Neural Network for Fraud Detection",
        "venue_year": "IEEE ICDM 2019 (Wang et al.)",
        "in_pygod_1_1": False,
        "status_note": "Semi-supervised multi-relation financial detector; requires labeled fraud ground truth during training.",
    },
]


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(DETECTORS)
    df.to_csv(OUT_CSV, index=False)
    print(f"Generated {OUT_CSV} with {len(df)} candidate detectors.")
    
    # Also write Markdown report in reports/
    report_path = REPO_ROOT / "dlg_gnn/outputs/benchmark/manuscript_m1/reports/05_detector_selection_and_literature_audit.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    md = ["# PyGOD 1.1.0 Official Detector Selection and Literature Audit\n"]
    md.append("## 1. Official PyGOD 1.1.0 Candidate Universe\n")
    md.append("| Detector | Objective Family | Deep GNN? | Unsupervised? | In PyGOD 1.1.0? | Status | Selection / Exclusion Rationale |")
    md.append("| :--- | :--- | :---: | :---: | :---: | :---: | :--- |")
    for d in DETECTORS:
        deep_s = "Yes" if d["is_deep_gnn"] else "No"
        unsup_s = "Yes" if d["unsupervised"] else "No"
        pyg_s = "Yes" if d["in_pygod_1_1"] else "No"
        status_s = f"**{d['selected_status']}**"
        md.append(f"| {d['detector']} | {d['objective_family']} | {deep_s} | {unsup_s} | {pyg_s} | {status_s} | {d['primary_reason']} |")

    md.append("\n## 2. Representation of Core GAD Objective Families\n")
    md.append("The 6 established baselines selected from PyGOD 1.1.0 uniquely cover the 6 foundational GAD objective paradigms without redundancy:\n")
    md.append("1. **Reconstruction GAE**: `DOMINANT` (Joint feature and structure reconstruction)\n")
    md.append("2. **Dual Nonlinear Reconstruction**: `AnomalyDAE` (Decoupled structure-guided GAT and attribute-guided dense decoder)\n")
    md.append("3. **Contrastive SSL**: `CoLA` (Target node vs local ego-net mutual information maximization)\n")
    md.append("4. **Hybrid Contrastive/Reconstruction**: `CONAD` (Siamese contrastive perturbation + graph autoencoder)\n")
    md.append("5. **Neighborhood Distribution Reconstruction**: `GADNR` (Gaussian/Wasserstein neighborhood aggregation and degree reconstruction)\n")
    md.append("6. **One-Class Hypersphere**: `OCGNN` (Minimum enclosing hypersphere boundary in GNN representation space)\n")

    md.append("\n## 3. Bibliographic Verification of External/Literature Models\n")
    md.append("| Name | Full Name / Architecture | Reference Venue | PyGOD 1.1.0? | Scope & Applicability Analysis |")
    md.append("| :--- | :--- | :--- | :---: | :--- |")
    for lit in LITERATURE_VERIFICATIONS:
        in_p = "Yes" if lit["in_pygod_1_1"] else "No"
        md.append(f"| {lit['name']} | {lit['full_name']} | {lit['venue_year']} | {in_p} | {lit['status_note']} |")

    report_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Generated {report_path}")


if __name__ == "__main__":
    main()
