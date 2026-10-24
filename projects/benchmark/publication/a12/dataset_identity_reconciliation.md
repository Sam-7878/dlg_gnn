# A12 Dataset Identity Reconciliation Report (P0-01)

**Date:** 2026-10-24  
**Gate:** **A12-G1 PASSED — Canonical Model-Input Identity Verified**  

## 1. Summary of Changes from A11

In A11 Table 1, datasets were improperly assigned raw/upstream repository totals rather than the exact model-input tensor counts evaluated by the algorithms. A12 restores the authoritative contract:

- **Evaluated Model-Input Graph:** The exact tensor passed to PyTorch/PyGOD models (`N`, `E`, `d`, $\pi_{\mathrm{test}}$).
- **Upstream Source Graph:** Separate columns explicitly detailing the raw repository graph before filtering, component extraction, or similarity graph construction.
- **Target Provenance Clarity:** Explicitly identifies **five source/native-label** datasets (Elliptic, DGraphFin, Ethereum, BSC, Polygon), **eight injected-target** datasets (including BitcoinOTC as a real trust graph with injected targets), and **one external** cybersecurity network (LANL-RedTeam).

## 2. Reconciled Contract Table

| Dataset | Model Nodes ($N$) | Model Edges ($E$) | Feat ($d$) | Test Preval. ($\pi$) | Source Nodes | Source Edges | Target Provenance |
|---|---:|---:|---:|---:|---:|---:|---|
| **Elliptic** | 46,564 | 83,188 | 165 | 0.0976 | 203769 | 234355 | real_external_label |
| **DGraphFin** | 1,225,601 | 746,271 | 17 | 0.0127 | 3700550 | 4300999 | real_external_label |
| **Ethereum** | 14,385 | 115,065 | 8 | 0.4184 | - | - | real_external_label |
| **BSC** | 7,481 | 59,832 | 8 | 0.1476 | - | - | real_external_label |
| **Polygon** | 2,303 | 18,411 | 8 | 0.0261 | - | - | real_external_label |
| **BitcoinOTC** | 6,005 | 7,409 | 67 | 0.0300 | 5881 | 35592 | synthetic_node_injection_on_real_trust_graph |
| **Yelp-Syn** | 716,847 | 14,004,995 | 300 | 0.0100 | 45954 | 3846979 | synthetic_node_injection |
| **Amazon-Syn** | 13,752 | 507,378 | 767 | 0.0300 | 11944 | 4397792 | synthetic_node_injection |
| **Reddit-Syn** | 232,965 | 114,869,737 | 602 | 0.0200 | 232965 | 11606919 | synthetic_node_injection |
| **Flickr-Syn** | 89,250 | 1,001,494 | 500 | 0.0200 | 89250 | 899756 | synthetic_node_injection |
| **Cora-Syn** | 2,708 | 11,276 | 1433 | 0.0299 | 2708 | 5429 | synthetic_node_injection |
| **CiteSeer-Syn** | 3,327 | 9,914 | 3703 | 0.0298 | 3327 | 4732 | synthetic_node_injection |
| **PubMed-Syn** | 19,717 | 93,958 | 500 | 0.0300 | 19717 | 44338 | synthetic_node_injection |
| **LANL-RedTeam** | 16,694 | 323,897 | 13 | 0.0180 | 156117 | 1496194 | real_external_label |

