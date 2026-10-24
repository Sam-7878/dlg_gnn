# DLG-GNN Benchmark Campaign & Provenance Map (A11)

**Document ID:** DLG-BENCH-CAMP-MAP-A11  
**Date:** 2026-10-24  
**Scope:** 13 Primary Evaluation Datasets + 1 External Cybersecurity Evaluation Dataset (LANL-RedTeam), 7 Models, 5 Seeds (42~46)

---

## 1. Multi-Campaign Generation Structure

The DLG-GNN benchmark evidence is organized into distinct execution tiers and campaigns. Each tier reflects frozen, verified execution contracts rather than a single homogeneous training cycle.

| Campaign Tier | Target Datasets | Models Evaluated | Execution Characteristics | Preservation & Reproducibility Tier |
|---|---|---|---|---|
| **Historical Round 5** | Elliptic, DGraphFin, Yelp-Syn, Amazon-Syn, Flickr-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn (8 datasets) | DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug | Frozen historical run outputs; inductive/transductive split per dataset specification; 5 seeds each (40 pairs, 200 runs). | Preserved scalar/metric tier (`public_numeric_evidence.zip`); approved run registry hashes locked. |
| **Canonical BitcoinOTC** | BitcoinOTC (1 dataset) | DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug | Transaction trust graph; signed edge weights mapped to feature attributes; 5 seeds (7 pairs, 35 runs). | Preserved canonical graph tier; explicit $N=5,881, E=35,592$ structure documented. |
| **Corrected Crypto A08** | Ethereum, BSC, Polygon (3 datasets) | DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug | Repaired smart-contract feature similarity graphs; clean non-leaking transaction temporal features; strictly transductive validation-only threshold tuning. | Fully raw-score and metric recomputable tier (`a08_data_repair`); 105 runs with exact NPZ array hashes. |
| **Targeted AnomalyDAE / Exact CPU** | Elliptic, Reddit-Syn, large graphs | AnomalyDAE, DLG exact reconstruction | Chunked/row-block reconstruction avoiding dense $N \times N$ matrix instantiation. | Verified exact mathematical equivalence; explicit memory and runtime documentation. |
| **External Cybersecurity** | LANL-RedTeam (1 dataset) | DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug | Enterprise authentication graph with red-team penetration compromised accounts; 5 seeds (7 pairs, 35 runs). | Single-graph external descriptive evaluation; excluded from omnibus inferential Friedman views (S1–S4). |

---

## 2. Model & Dataset Population Overview

- **Primary Datasets ($N=13$):**
  - **Financial / Blockchain Native (5):** Elliptic, DGraphFin, Ethereum, BSC, Polygon
  - **Constructed / Injected Anomaly Graphs (8):** BitcoinOTC, Yelp-Syn, Amazon-Syn, Reddit-Syn, Flickr-Syn, Cora-Syn, CiteSeer-Syn, PubMed-Syn
- **External Evaluation ($N=1$):** LANL-RedTeam
- **Primary Support Contract:**
  - Total primary cells: $13 \times 7 = 91$ pairs.
  - Fully supported (`SUPPORTED_EXACT`): 84 pairs ($84 \times 5 = 420$ successful runs).
  - Unsupported (OOM / Timeout on massive graphs): 7 pairs (GADNR on DGraphFin, Reddit-Syn, Flickr-Syn, Amazon-Syn, Yelp-Syn, Cora-Syn, CiteSeer-Syn; CoLA/OCGNN memory limits on specific configurations).
- **Evaluation Prevalences:**
  - Elliptic: $\sim 0.0976$ (transductive test split)
  - DGraphFin: $\sim 0.0134$
  - Ethereum: $\sim 0.0195$
  - BSC: $\sim 0.0210$
  - Polygon: $\sim 0.0052$

---

## 3. Cryptographic Asset & Integrity Binding

- **GitHub Release Tag:** `benchmark-a09-2026-10-10`
- **Release Commit:** `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d`
- **Public Evidence Archives:**
  - `a08_public_numeric_evidence.zip`: SHA256 `b1ae519a058784ed1d8306d2986748de6832e142209502e984030ff35b18e100`
  - `a09_submission_closure_companion.zip`: SHA256 `336c7bcdaa50c68582be78161ccea3c769cf214d1b71a5d327e98709d1de94f5`
- **Number Registry Verification:**
  - SHA256: `532651b28105ddd83194ee51e06084fd692d2f8555f671fbe2ef147296ff7cbf`
