# 07. Cross-Project Integration & Reuse Guide

This document provides practical, plug-and-play recipes for integrating and reusing modules from **DLG-GNN** in other external projects (e.g., fraud detection systems, AML engines, academic GNN benchmarking, or large-scale graph autoencoders).

---

## 1. Integration Patterns Overview

| Pattern | Goal | Key Source Files to Reuse | Target Environment |
|---|---|---|---|
| [**Pattern A**](#pattern-a-standalone-pygod-anomaly-detector) | Drop-in PyGOD anomaly detector | `src/gog_fraud/models/pygod/dlg.py`<br>`src/gog_fraud/models/pygod/exact_reconstruction.py` | Any PyTorch Geometric / PyGOD project |
| [**Pattern B**](#pattern-b-reusing-exact-sparse-reconstruction-backend) | Eliminate $O(N^2)$ memory wall in custom GNN autoencoders | `src/gog_fraud/models/pygod/exact_reconstruction.py` | PyTorch / PyG autoencoder architectures |
| [**Pattern C**](#pattern-c-reusing-the-benchmark--statistical-engine) | Benchmark a custom model across 10 datasets with Friedman/Wilcoxon tests | `src/analysis/add_benchmark_analysis.py`<br>`experiments/benchmark/run_sci_round5_final.py` | Academic GNN benchmark projects |
| [**Pattern D**](#pattern-d-adopting-the-streaming-monte-carlo-pipeline) | Real-time sliding window graph AML engine | `src/gog_fraud/streaming/`<br>`src/gog_fraud/selection/` | Production stream processing (Kafka / Flink) |
| [**Pattern E**](#pattern-e-reusing-the-anomaly-injection-protocol) | Inject synthetic anomalies into arbitrary graphs | `src/gog_fraud/data/transforms/` | Graph ML data augmentation pipelines |

---

## Pattern A: Standalone PyGOD Anomaly Detector

### Use Case
You have an existing PyTorch Geometric `Data` object and want to train and evaluate the Decoupled Local-to-Global (DLG) detector using the standard PyGOD `fit()` / `decision_function()` interface.

### Minimal Working Example
```python
import torch
from torch_geometric.data import Data
from gog_fraud.models.pygod.dlg import DLG

# 1. Prepare your PyG Data object
# x: [num_nodes, in_channels]
# edge_index: [2, num_edges]
x = torch.randn(1000, 32)
edge_index = torch.randint(0, 1000, (2, 5000))
data = Data(x=x, edge_index=edge_index)

# 2. Instantiate DLG with Exact Sparse Backend
model = DLG(
    hid_dim=64,
    num_layers=4,
    alpha=0.5,                             # Learnable fusion between local & global
    reconstruction_backend="exact_sparse", # O(|E|d + Nd^2) arithmetic, NO dense O(N^2) RAM
    epoch=50,
    lr=0.004,
    weight_decay=1e-4,
    contamination=0.05,                    # Expected outlier ratio (5%)
    gpu=0 if torch.cuda.is_available() else -1,
)

# 3. Fit model on the graph
model.fit(data)

# 4. Predict anomaly scores and binary outlier labels
outlier_scores = model.decision_function(data) # Continuous scores: [num_nodes]
binary_preds = model.predict(data)             # Binary labels: 0 (inlier), 1 (outlier)

print(f"Top 5 anomalous nodes: {torch.topk(torch.tensor(outlier_scores), k=5).indices.tolist()}")
```

---

## Pattern B: Reusing Exact Sparse Reconstruction Backend

### Use Case
You are building your own custom graph autoencoder (e.g., GAE, VGAE, DOMINANT, or GAD-NR) and your GPU runs out of memory because you are trying to compute $\|\mathbf{A} - \mathbf{Z}\mathbf{Z}^\top\|_F^2$ on graphs with $> 50{,}000$ nodes.

### Solution
Drop in [`exact_reconstruction.py`](file:///d:/_Work/goat_bank/dlg_gnn/src/gog_fraud/models/pygod/exact_reconstruction.py). It computes the exact scalar Frobenius loss without creating the dense $N \times N$ adjacency matrix.

### Minimal Working Example
```python
import torch
from gog_fraud.models.pygod.exact_reconstruction import resolve_backend

# Latent node embeddings from your GNN encoder: [N, d]
N, d = 100000, 64
Z = torch.randn(N, d, requires_grad=True, device="cuda")

# Sparse edge index: [2, |E|]
edge_index = torch.randint(0, N, (2, 500000), device="cuda")

# 1. Resolve exact sparse backend
backend = resolve_backend("exact_sparse")

# 2. Compute exact Frobenius structure loss in closed form:
# ||A - Z Z^T||_F^2 = ||Z^T Z||_F^2 - 2 * sum_{(u,v) in E} (z_u . z_v) + |E|
# Memory used: only O(|E| + Nd) instead of 40 GB for an N x N dense matrix!
gram = torch.matmul(Z.t(), Z)                 # [d, d] -> 64x64 matrix!
term_gram = torch.sum(gram ** 2)              # ||Z^T Z||_F^2
term_edges = torch.sum(Z[edge_index[0]] * Z[edge_index[1]]) # sum_{E} z_u^T z_v
term_ones = float(edge_index.size(1))         # |E|

loss_struct = (term_gram - 2.0 * term_edges + term_ones) / (N * N)

# 3. Backpropagate directly
loss_struct.backward()
print("Backprop succeeded! Peak memory:", torch.cuda.max_memory_allocated() / (1024 ** 2), "MB")
```

---

## Pattern C: Reusing the Benchmark & Statistical Engine

### Use Case
You are developing a novel GNN detector and want to rigorously compare it against the established 10-dataset benchmark portfolio using Friedman rank tests and Holm-adjusted Wilcoxon post-hoc tests.

### Integration Steps
1. **Define Your Model Wrapper:** Implement the PyGOD `fit(data)` and `decision_function(data)` API.
2. **Register Model in Benchmark Matrix:** Add your model name to `model_dataset_support_matrix.csv`.
3. **Run Experiments Across 5 Seeds:** Save output metric CSVs (ROC-AUC, PR-AUC, Macro-F1).
4. **Generate Publication-Grade Statistics:**
```python
from analysis.add_benchmark_analysis import (
    compute_friedman_test,
    compute_wilcoxon_holm_tests,
    generate_latex_table
)

# Load your raw multi-seed benchmark results dataframe
# df columns: ['dataset', 'model', 'seed', 'roc_auc', 'pr_auc', 'f1']
friedman_stat, p_val, rank_df = compute_friedman_test(df, metric="roc_auc")
print(f"Friedman Test: Chi2={friedman_stat:.4f}, p={p_val:.4e}")

# Pairwise Wilcoxon tests against baseline with Holm FWER correction
pairwise_df = compute_wilcoxon_holm_tests(df, reference_model="DLG-Aug", metric="roc_auc")
print(pairwise_df)
```

---

## Pattern D: Adopting the Streaming Monte Carlo AML Pipeline

### Use Case
You are building an AML fraud monitoring microservice that consumes incoming financial transactions from Apache Kafka and must maintain bounded graph state while scoring risks in real time.

### Minimal Working Example
```python
from gog_fraud.streaming.engine import StreamingEngine
from gog_fraud.streaming.subgraph_store import BoundedSubgraphStore
from gog_fraud.streaming.relation_state import RelationStateManager
from gog_fraud.selection.router import AMLSelectionRouter

# 1. Initialize Bounded Storage (e.g. 1-hour temporal window, max 2-hop ego-nets)
subgraph_store = BoundedSubgraphStore(window_seconds=3600, max_hops=2)
relation_state = RelationStateManager(decay_lambda=0.01)
router = AMLSelectionRouter(high_risk_threshold=0.85)

# 2. Instantiate Streaming Engine
engine = StreamingEngine(
    subgraph_store=subgraph_store,
    relation_state=relation_state,
    router=router,
)

# 3. Process Live Transaction Stream
def on_transaction_received(tx_event):
    # tx_event: dict with 'src', 'dst', 'amount', 'timestamp', 'features'
    risk_score, alert_payload = engine.process_transaction(
        src=tx_event["src"],
        dst=tx_event["dst"],
        amount=tx_event["amount"],
        timestamp=tx_event["timestamp"],
        features=tx_event["features"],
    )
    
    if alert_payload is not None:
        print(f"[AML ALERT] High-risk transaction detected: {alert_payload}")
    return risk_score
```

---

## Pattern E: Reusing the Anomaly Injection Protocol

### Use Case
You want to evaluate graph models on your own proprietary graphs (e.g. internal bank transaction networks) and need a reproducible protocol to inject realistic contextual and structural anomalies without leaking test labels.

### Code Recipe
```python
from gog_fraud.data.transforms import (
    inject_contextual_anomalies,
    inject_structural_anomalies,
)

# Ingest your clean base graph
clean_data = ... # PyG Data object

# 1. Inject contextual anomalies (perturbing node attributes to match distant clusters)
data_with_context = inject_contextual_anomalies(
    clean_data,
    anomaly_ratio=0.05, # 5% contextual outliers
    seed=42,
)

# 2. Inject structural anomalies (dense deceptive cliques)
data_synthetic = inject_structural_anomalies(
    data_with_context,
    anomaly_ratio=0.05, # 5% structural outliers
    clique_size=15,
    seed=42,
)

print(f"Synthetic benchmark created with {data_synthetic.y.sum().item()} anomalous nodes!")
```

---

## 2. Summary of Reusable Assets

| Asset Name | Location | Dependency | Standalone Usable? |
|---|---|---|:---:|
| `DLG` Detector | `src/gog_fraud/models/pygod/dlg.py` | PyTorch, PyG, PyGOD | **Yes** |
| `exact_sparse` Backend | `src/gog_fraud/models/pygod/exact_reconstruction.py` | PyTorch, torch_sparse | **Yes (Zero DLG dependencies)** |
| `BoundedSubgraphStore` | `src/gog_fraud/streaming/subgraph_store.py` | Python stdlib, PyG | **Yes** |
| `RelationStateManager` | `src/gog_fraud/streaming/relation_state.py` | NumPy, PyTorch | **Yes** |
| Statistical Analyzers | `src/analysis/add_benchmark_analysis.py` | SciPy, Pandas | **Yes** |
| Graph Transforms | `src/gog_fraud/data/transforms/` | PyTorch Geometric | **Yes** |
