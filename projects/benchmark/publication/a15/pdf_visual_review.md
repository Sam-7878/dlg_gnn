# PDF Visual Review & Layout Verification Report (A15)

**Document ID:** DLG-BENCH-VISUAL-A15  
**Date:** 2026-10-25  
**Audit Target:** Preprints.org PDF (18 pages) & MDPI Applied Sciences PDF (16 pages)  

---

## 1. Visual Inspection of Key Figures and Tables

1. **Figure 1 (DLG Architecture):**
   - Clean, balanced 4-stage diagram with clear hierarchical arrows from Graph Input -> Level 1 Local Ego-Net -> Level 2 Global Relational GNN -> Closed-Form Exact Gram Decoder.
   - Text boxes, mathematical formulas, and bounds are razor sharp and fully aligned.

2. **Figure 2 (Hardware Support Matrix):**
   - Complete 14-dataset $\times$ 7-model grid strictly reflects canonical registry.
   - 84 primary supported cells (5/5 seeds) colored soft green (`#d4edda`); 7 external LANL cells colored soft blue (`#cce5ff`); exactly 7 unsupportable cells distinctly labeled as OOM (5 cells for GADNR) or Err (2 cells for AnomalyDAE).
   - Zero fail-open default shading.

3. **Figure 3 (Conditional Representation Delta Bar Chart):**
   - Horizontal layout with generous bounds (`[-0.112, 0.065]`) eliminates axis compression.
   - `Reddit-Syn` ($-0.0656 \pm 0.0031$) has ample whitespace to the right of the y-axis label with zero text overlap.
   - `BSC` ($+0.0021 \pm 0.0103$) is correctly rendered as a positive gain with visible error bounds and zero bar clipping.
   - `Ethereum` ($+0.0253 \pm 0.0144$) and `Elliptic` ($+0.0350 \pm 0.0022$) are clearly rendered with non-colliding labels.
   - `Polygon` ($-0.0124 \pm 0.0054$) and `LANL-RedTeam` ($-0.0252 \pm 0.0126$) are unambiguously styled in red and blue respectively.
   - Descriptive SE note at the bottom is fully visible.

4. **Table 1 (Evaluated Dataset Properties):**
   - Explicitly separates Graph Target Prevalence ($y_{pos}/N$) from Evaluated Test Set Positive Rate $\pi_{\text{test}}$.
   - Denominators clearly footnoted.

5. **Table 2 (Main AP Benchmark Performance):**
   - Features verified test positive rate $\pi_{\text{test}}$ baselines for each dataset (Ethereum 0.4089, BSC 0.1473, Polygon 0.0259, etc.).
   - Model support counts and footnoted failure etiologies match the canonical support matrix.

6. **Table 4 (Conditional Representation Performance Delta):**
   - Replaces unproven causal hub mechanisms with observational phrasing.
   - Exact numerical match with Figure 3 and `conditional_ap_14.csv`.
