# Data Availability & Reproducibility Statement (A11)

**Manuscript Title:** Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs  
**Journal:** *Applied Sciences* (MDPI), Special Issue "Machine Learning and Its Application for Anomaly Detection"  
**Date:** 2026-10-24  

---

## 1. Public Numeric Reproduction Artifacts

In compliance with MDPI Open Science and Research Data Policies, all numerical results, statistical tables, aggregation scripts, and reproduction routines are openly archived:

- **Public Repository:** `https://github.com/Sam-7878/dlg_gnn`
- **Immutable Release Tag:** `benchmark-a09-2026-10-10`
- **Scientific Commit:** `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d`
- **Public Evidence Digests:**
  - `a08_public_numeric_evidence.zip`: SHA256 `b1ae519a058784ed1d8306d2986748de6832e142209502e984030ff35b18e100`
  - `a09_submission_closure_companion.zip`: SHA256 `336c7bcdaa50c68582be78161ccea3c769cf214d1b71a5d327e98709d1de94f5`
- **Reproduction Command:**
  ```bash
  python projects/benchmark/scripts/a08_aggregate.py --public-tables
  ```
  This command regenerates all 11 primary publication tables, S1--S4 complete-case statistics, and Wilcoxon-Holm pairwise contrasts directly from the verified numeric registry.

---

## 2. Underlying Raw Datasets

All raw graph datasets evaluated in this study are publicly accessible through their original institutional repositories:
- **Ethereum, BSC, Polygon:** Multi-Chain Graphs of Graphs (Luo et al., 2024), available at `https://github.com/Xtra-Computing/Cryptocurrency-Graphs-of-graphs`.
- **Elliptic Bitcoin Transaction Dataset:** Weber et al. (2019), available via Kaggle and official GitHub repository.
- **DGraphFin:** FinTech loan network (Huang et al., 2022), available at `https://dgraph.xinye.com`.
- **BitcoinOTC:** Stanford Network Analysis Project (SNAP; Kumar et al., 2016), available at `https://snap.stanford.edu/data/soc-sign-bitcoinotc.html`.
- **Benchmark Graphs (Yelp, Amazon, Reddit, Flickr, Cora, CiteSeer, PubMed):** Standard PyG / PyGOD library datasets.
- **LANL-RedTeam:** Los Alamos National Laboratory Cyber Security Data (Kent, 2015), available at `https://csr.lanl.gov/data/cyber1/`.

---

## 3. Reviewer Access to Author-Local Checkpoints & Node Scores

To protect transaction counterparty privacy and comply with repository size constraints, raw high-dimensional node score vectors and intermediate neural model weights are retained in author-local secure storage:
- **Available Records:** 105 verified raw-score NPZ arrays for cryptocurrency evaluations and corresponding PyTorch model checkpoints.
- **Access Protocol:** Reviewers and editors may request access to any preserved score array or checkpoint via the corresponding author (**Prof. Ki-Hyung Kim**, `kkim86@ajou.ac.kr`). Files will be delivered via a secure, logged transfer channel.
