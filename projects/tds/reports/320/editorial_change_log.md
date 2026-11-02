# Editorial Change Log — R320

This document records the exact editorial enhancements applied in R320 on top of the R319 scientific baseline.

## 1. Section 3 Architecture Overview Figure Added
- **Figure Added**: `fig_tds_architecture.pdf` inserted into Section 3 (Materials and Methods).
- **Structure**: 4-phase overview diagram illustrating:
  1. *Phase 1*: Multi-chain transfer stream ingestion (Ethereum, BSC, Polygon) and temporal cutoff windowing ($t_e \leq t_i$) for Full-Archive (Cohort A) and 7-Day Horizon (Cohort B).
  2. *Phase 2*: Dual-track representation extraction: Retained 128-event graph topology $\mathcal{E}$ with 3-dim node degree features vs. 24-dimensional tabular degree summary vector.
  3. *Phase 3*: Controlled model architectures: Standard GIN-0 (lacks count scale), Size-augmented GNNs (GIN-size, GIN-sum), and tabular baselines (Summary MLP, LightGBM, Retained-node LR, Isolation Forest).
  4. *Phase 4*: Post-inference auditable evaluation: Validation-only temperature scaling, retrospective observed-span diagnostics ($L_i = l_i - f_i$), chain-stratified alert budgets, and decision checklist.
- **Textual Reference**: Added contextual framing paragraph at the opening of Section 3 cross-referencing `Figure~\ref{fig:tds-architecture}`.

## 2. Related Work Section Major Expansion
- **Motivation**: Section 2 was previously brief (~4 paragraphs). As instructed, we cross-referenced and integrated key references from the accompanying `dlg_gnn/projects/benchmark/submission/preprints` manuscript (planned for submission prior to this work).
- **Enrichment Structure**: Expanded into 4 structured subsections:
  - *2.1 Unsupervised and Supervised Graph Anomaly Detection*: Incorporates comprehensive surveys (Ma et al. TKDE 2023, Qiao et al. TKDE 2025), benchmarks (PyGOD, BOND, GADBench, GAD-in-the-Wild), classical autoencoder/contrastive detectors (DOMINANT, AnomalyDAE, CoLA, CONAD, GAD-NR, OCGNN), and recent *Applied Sciences* reviews (Dilini et al. 2026).
  - *2.2 Blockchain Transaction Networks, Smart Contracts, and Label Shift*: Integrates transaction network forensics (Elliptic, DGraph, Bellei et al. 2024), token lifecycle studies (Cernera et al. 2023), class imbalance methods (GraphSMOTE, PC-GNN, Dual-Augment GNN), smart Ponzi scheme detection in *Applied Sciences* (Jiang & Tsai 2025), and formal dataset shift taxonomies (Moreno-Torres et al. 2012, Dal Pozzolo et al. 2015).
  - *2.3 Graph Neural Networks versus Tabular Baselines*: Introduces empirical benchmarking rigor (Shchur et al. 2018, Errica et al. ICLR 2020, Dwivedi et al. 2023), systematic graph-derived signal evaluation in *Applied Sciences* (Heidrich et al. 2026), GADBench tabular findings (Tang et al. NeurIPS 2023), and decoupled GNN frameworks (Park & Kim 2026).
  - *2.4 Dynamic and Temporal Graph Learning Limitations*: Highlights continuous-time dynamic models (TGAT, TGN) and discrete-time models (EvolveGCN, DyRep, JODIE), temporal evaluation pitfalls (TGB, DyGFormer), and contrasts dynamic link prediction with cutoff-admissible contract classification.
- **Reference Count**: Total BibTeX entries increased from 34 to 61 (27 high-impact references imported).

## 3. Preservation of Scientific Claims & Rigor
- Preserved strict delineation between representation advantage (explicit size information) and architectural superiority. No claim that sum pooling alone caused the recovery.
- Maintained strict warnings regarding retrospective label semantics (no fraud-discovery timestamps).
- Preserved exact 100% byte-level parity between `preprints` and `mdpi_applsci_anomaly` shared science files (`manuscript_body.tex`, `references.bib`, `declarations.tex`).
