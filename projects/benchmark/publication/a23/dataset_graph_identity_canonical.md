# Canonical Dataset Graph Identity, Topology, and Split Provenance (`A23-G1`)

## 1. Reconciliation of Upstream Objects, Model-Input Graphs, Labeled Pools, and Test Splits

In the initial A22 diagnostic script (`a22_g4_dataset_diagnostics.py`), two provenance mapping bugs occurred:
1. **DGraphFin Node Count (`367,702` vs. `1,225,601`):** `dataset_manifest_canonical.json` records three distinct quantities for `DGraphFin`: `source_graph_nodes = 3,700,550` (raw upstream nodes), `graph_nodes = 1,225,601` (the constructed/model-input induced graph reported in Manuscript Table 1), and `evaluation_nodes = 367,702` (the combined validation + test labeled borrower population, from which each seed draws `183,840` test nodes containing `2,326` positives). The A22 script mistakenly read `evaluation_nodes` (`367,702`) into `model_input_N`.
2. **Mismatched Upstream Denominators (`node_retention_ratio > 1`):** The A22 script divided `model_input_N` by literature node counts from different dataset variants (`11,944`-node FraudAmazon vs. `13,752`-node PyG Amazon-Computers; `45,954`-node YelpChi vs. `716,847`-node PyG Yelp; raw `5,881` Bitcoin-OTC user IDs vs. `6,005` indexed tensor slots; early transfer-only contract subsets vs. the canonical `14,385` / `7,481` / `2,303` smart-contract kNN/hybrid graphs). In **A23**, the misleading `node_retention_ratio` column is removed and replaced with explicit multi-stage counts (`raw_upstream_object_count`, `constructed_graph_N`, `model_input_N`, `labeled_evaluation_population_N`, `test_population_N_by_seed`, and `positives_by_seed`).

## 2. Canonical 14-Dataset Identity & Split Table

| Dataset | Raw Upstream Count | Model-Input $(N, E, F)$ | Labeled Pool $N$ | Test Split $N$ (Seeds 42–46) | Test Positives (Seeds 42–46) | $E/N$ (Dir.) | $2E/N$ (Undir.) |
| :--- | ---: | :--- | ---: | :--- | :--- | ---: | ---: |
| `Elliptic` | `203,769` | `(46,564, 83,188, 165)` | `46,564` | `42..46:23282` | `42:2272;43:2272;44:2272;45:2273;46:2273` | `1.7865` | `3.5731` |
| `DGraphFin` | `3,700,550` | `(1,225,601, 746,271, 17)` | `367,702` | `42..46:183840` | `42..46:2326` | `0.6089` | `1.2178` |
| `BitcoinOTC` | `5,881` | `(6,005, 7,409, 67)` | `6,005` | `42..46:3003` | `42..46:90` | `1.2338` | `2.4676` |
| `Ethereum` | `14,385` | `(14,385, 115,065, 8)` | `14,385` | `42..46:2893` | `42:1143;43:1246;44:1153;45:1184;46:1189` | `7.9990` | `15.9979` |
| `BSC` | `7,481` | `(7,481, 59,832, 8)` | `7,481` | `42..46:1500` | `42:215;43:240;44:232;45:215;46:203` | `7.9979` | `15.9957` |
| `Polygon` | `2,303` | `(2,303, 18,411, 8)` | `2,303` | `42..46:471` | `42:16;43:11;44:7;45:18;46:9` | `7.9944` | `15.9887` |
| `Amazon-Syn` | `13,752` | `(13,752, 507,378, 767)` | `13,752` | `42..46:6876` | `42..46:206` | `36.8949` | `73.7897` |
| `Yelp-Syn` | `716,847` | `(716,847, 14,004,995, 300)` | `716,847` | `42..46:358424` | `42..46:3584` | `19.5369` | `39.0739` |
| `Reddit-Syn` | `232,965` | `(232,965, 114,869,737, 602)` | `232,965` | `42..46:116483` | `42..46:2330` | `493.0772` | `986.1545` |
| `Flickr-Syn` | `89,250` | `(89,250, 1,001,494, 500)` | `89,250` | `42..46:44625` | `42:892;43:892;44:892;45:893;46:893` | `11.2212` | `22.4424` |
| `Cora-Syn` | `2,708` | `(2,708, 11,276, 1433)` | `2,708` | `42..46:1354` | `42:40;43:40;44:40;45:41;46:41` | `4.1640` | `8.3279` |
| `CiteSeer-Syn` | `3,327` | `(3,327, 9,914, 3703)` | `3,327` | `42..46:1664` | `42..46:50` | `2.9799` | `5.9597` |
| `PubMed-Syn` | `19,717` | `(19,717, 93,958, 500)` | `19,717` | `42..46:9859` | `42..46:296` | `4.7653` | `9.5307` |
| `LANL-RedTeam` | `156,117` | `(16,694, 323,897, 13)` | `16,694` | `42..46:3338` | `42..46:60` | `19.4020` | `38.8040` |

## 3. Degree Convention Note

- All `graph_edges` ($E$) in `dataset_manifest_canonical.json` and Manuscript Table 1 represent stored directed edge entries in PyTorch Geometric `edge_index` (`shape = [2, E]`).
- Accordingly, `directed_out_degree_E_over_N` is $E / N$ (for `DGraphFin`, $746,271 / 1,225,601 = 0.6089$), while `undirected_degree_equiv_2E_over_N` is $2E / N$ ($1.2178$ for `DGraphFin`).
- For `Ethereum`, `BSC`, and `Polygon`, the canonical A08 graphs are $k=5$ feature-similarity + hybrid transfer graphs over the labeled smart-contract pools (`14,385`, `7,481`, and `2,303` contracts), yielding $E/N \approx 8.00$ directed edges per node ($2E/N \approx 16.00$).
