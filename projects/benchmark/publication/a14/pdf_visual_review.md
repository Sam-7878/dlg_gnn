# PDF Visual Review & Layout Verification Report (A14)

**Document ID:** DLG-BENCH-VISUAL-A14  
**Date:** 2026-10-24  
**Audit Target:** Preprints.org PDF (18 pages) & MDPI Applied Sciences PDF (16 pages)  

---

## 1. Visual Inspection of Key Figures and Tables

1. **Figure 1 (DLG Architecture):**
   - Clean, balanced 4-stage diagram with clear hierarchical arrows from Graph Input -> Level 1 Local Ego-Net -> Level 2 Global Relational GNN -> Closed-Form Exact Gram Decoder.
   - Text boxes, mathematical formulas, and bounds are razor sharp and fully aligned.

2. **Figure 2 (Hardware Support Matrix):**
   - Complete 14-dataset $	imes$ 7-model grid strictly reflects canonical registry.
   - 84 primary supported cells (5/5 seeds) colored soft green (`#d4edda`); 7 external LANL cells colored soft blue (`#cce5ff`); 7 unsupportable cells distinctly labeled as OOM (5 cells) or Err (2 cells).
   - Zero fail-open default shading.

3. **Figure 3 (Conditional Representation Delta Bar Chart):**
   - **Resolution of A13 Layout Defects:**
     - Horizontal layout with generous bounds (`[-0.112, 0.065]`) eliminates previous axis compression.
     - `Reddit-Syn` ($-0.0656 \pm 0.0031$) has ample whitespace to the right of the y-axis label with zero text overlap.
     - `BSC` ($+0.0021 \pm 0.0103$) is correctly rendered as a positive gain with visible error bounds and zero bar clipping.
     - `Ethereum` ($+0.0253 \pm 0.0144$) and `Elliptic` ($+0.0350 \pm 0.0022$) are clearly rendered with non-colliding labels.
     - `Polygon` ($-0.0124 \pm 0.0054$) and `LANL-RedTeam` ($-0.0252 \pm 0.0126$) are unambiguously styled in red and blue respectively.
     - Descriptive SE note at the bottom is fully visible.

4. **Table 4 (Conditional Delta Performance):**
   - Derived directly from `conditional_ap_14.csv`.
   - All 14 rows match Figure 3 deltas and signs with zero discrepancy.
   - Clear operational interpretations provided for all datasets.

5. **Table 1 (Dataset Properties) & Table 5 (Friedman Omnibus Statistics):**
   - Correctly positioned and styled with `tabularx` and `booktabs`.
   - S1--S4 statistics match frozen benchmarks (S1 Q=29.0357, S2 Q=23.7538, S3 Q=6.1333, S4 Q=21.8286).
