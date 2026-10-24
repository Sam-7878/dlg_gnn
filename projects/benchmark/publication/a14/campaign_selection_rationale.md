# Benchmark Campaign Selection Rationale (A14)

**Document ID:** DLG-BENCH-RATIONALE-A14  
**Date:** 2026-10-24  
**Authority:** A14 Final Campaign Coherence Work Order (`DLG-BENCH-WO-A14-CAMPAIGN-COHERENCE`)  
**Canonical Registry SHA-256:** `45d4ca7b4a9b010aa779e7cb60a1816e0f1c08e98039a119d008a1d1cd5db907`  

---

## 1. Executive Summary

During the A13 independent audit, a significant discrepancy was identified between the conditional delta Table 4 in the manuscript and the plotted values in Figure 3 (`fig_conditional_delta.pdf`). Specifically:
- In the manuscript Table 4, the conditional deltas ($\Delta\mathrm{AP} = \mathrm{AP}_{\mathrm{Aug}} - \mathrm{AP}_{\mathrm{Base}}$) for the three cryptocurrency networks were:
  - **Ethereum:** $+0.0253$ ($0.4106 \to 0.4359$)
  - **BSC:** $+0.0021$ ($0.3390 \to 0.3411$)
  - **Polygon:** $-0.0123$ ($0.0854 \to 0.0731$)
- In contrast, the figure generation script in A13 mistakenly ingested an older, uncorrected historical run registry (`evidence/astra_revision/approved_run_registry.csv`), which plotted obsolete runs from `outputs/benchmark/a03_production`:
  - **Ethereum (Obsolete A03):** $-0.0015$ ($0.9803 \to 0.9788$)
  - **BSC (Obsolete A03):** $-0.1983$ ($0.6161 \to 0.4179$)
  - **Polygon (Obsolete A03):** $+0.0055$ ($0.0330 \to 0.0385$)

This caused opposite sign assertions between the text and Figure 3, and clipped the obsolete $-0.1983$ BSC bar against the fixed figure axis.

## 2. Canonical Campaign Resolution

To enforce complete coherence across all publication artifacts without altering experimental outcomes or conducting new training, A14 formally adopts **one single, authoritative canonical execution registry**:

1. **Cryptocurrency Networks (Ethereum, BSC, Polygon):**
   - **Canonical Campaign:** `benchmark_a08_crypto_clean_v1` (105 runs across 7 models $\times$ 5 seeds).
   - **Rationale for Adoption:** In campaign A08 (`evidence/a08_data_repair/`), the smart-contract feature similarity graphs were rigorously audited and repaired to eliminate lookahead data leakage in temporal transaction features, and evaluation thresholds were locked to validation-only tuning. Full node-level prediction arrays (`.npz`) were author-locally archived.
   - **Superseded Campaign:** The historical `a03_production` runs are formally designated as **SUPERSEDED** and excluded from all publication tables, figures, and manuscripts. All 105 superseded runs are preserved in `campaign_conflict_report.csv` for auditability.

2. **Standard Synthetic Benchmarks & DGraphFin (8 Primary Datasets):**
   - **Canonical Campaign:** `benchmark_sci_round5_final` (Elliptic, DGraphFin, Yelp-Syn, Amazon-Syn, Reddit-Syn, Flickr-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn; 280 runs across supported pairs).

3. **Bitcoin Trust Network (1 Primary Dataset):**
   - **Canonical Campaign:** `benchmark_a05_bitcoinotc_v1` (35 runs across 7 models $\times$ 5 seeds).

4. **Enterprise Authentication Network (1 External Evaluation):**
   - **Canonical Campaign:** `benchmark_lanl_redteam_v1` (35 runs across 7 models $\times$ 5 seeds).

## 3. Totals and Support Verification

- **Total Primary Model-Dataset Pairs:** $13 \times 7 = 91$ pairs.
- **Primary Supported Pairs:** 84 pairs ($84 \times 5 = 420$ successful runs).
- **Primary Unsupported Pairs:** 7 pairs (GADNR OOM on DGraphFin, Reddit-Syn, Yelp-Syn, Amazon-Syn, Flickr-Syn, Cora-Syn, CiteSeer-Syn).
- **External Supported Pairs:** 7 pairs ($7 \times 5 = 35$ successful runs on LANL-RedTeam).
- **Grand Total Executions in Canonical Registry:** Exactly **455 successful runs**.

Every number in Table 1, Table 2, Table 3, Table 4, Figure 2, Figure 3, and the Abstract is strictly derived from this single registry.
