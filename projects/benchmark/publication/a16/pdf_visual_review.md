# PDF Visual Review & Layout Verification Report (A16 Closure)

**Document ID:** DLG-BENCH-VISUAL-A16  
**Date:** 2026-10-28  
**Audit Target:** Preprints.org PDF (18 pages) & MDPI Applied Sciences PDF (16 pages)  

---

## 1. Visual Inspection of Key Figures, Abstract, and Tables

1. **Abstract (Single-Paragraph 172 Words):**
   - Renders cleanly as a single concise 172-word paragraph on Page 1 of both Preprints.org (18 pages) and MDPI Applied Sciences (16 pages).
   - Numerically aligned with all tables (`84 of 91` primary pairs, `420 primary and 35 external runs`, Elliptic `+0.0350`, Reddit-Syn `-0.0656`, `1%` and `5%` alert budgets).

2. **Figure 1 (DLG Architecture):**
   - Clean, balanced 4-stage diagram with clear hierarchical arrows from Graph Input -> Level 1 Local Ego-Net -> Level 2 Global Relational GNN -> Closed-Form Exact Gram Decoder.

3. **Figure 2 (Hardware Support Matrix):**
   - Complete 14-dataset x 7-model grid strictly reflects `support_14x7.csv`.
   - 84 primary supported cells (`5/5` seeds) colored soft green (`#d4edda`); 7 external LANL cells (`Ext`) colored soft blue (`#cce5ff`); 7 unsupported primary cells labeled `Fail` (4 GADNR `NO_APPROVED_RAW_RUN` cells: DGraphFin, Elliptic, Reddit-Syn, Yelp-Syn) or `OOM` (1 GADNR `OBSERVED_HISTORICAL_CUDA_OOM` cell on Flickr-Syn + 2 AnomalyDAE `OBSERVED_CUDA_OOM` cells on Reddit-Syn and Yelp-Syn).

4. **Figure 3 (Conditional Representation Delta Bar Chart):**
   - Horizontal layout (`[-0.112, 0.065]`) with zero text overlap; `Elliptic` (`+0.0350`), `Ethereum` (`+0.0253`), `BSC` (`+0.0021`), `Polygon` (`-0.0124`), `Reddit-Syn` (`-0.0656`), and `LANL-RedTeam` (`-0.0252`) clearly labeled.

5. **Tables 1--6 and Appendix Tables A1--A2:**
   - Zero overfull/clipped tables, zero orphan headings, zero undefined references or citations.
