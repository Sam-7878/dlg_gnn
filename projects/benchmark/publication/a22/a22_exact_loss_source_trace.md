# Canonical Exact-Loss Source Trace & Mathematical Orientation Audit (`A23-G4`)

## 1. Authoritative Repository Paths & SHA-256 Digests

In the initial A22 audit note (`a22_exact_loss_source_trace.md`), shorthand paths (`dlg_gnn/model/...`) were used in prose. The exact repository paths under `src/gog_fraud/models/pygod/` and their verified SHA-256 digests are listed below:

| Repository / Environment Path | Role | SHA-256 |
| :--- | :--- | :--- |
| `/mnt/d/_work/goat_bank/.venv_cuda/lib/python3.14/site-packages/pygod/nn/functional.py` | Upstream PyGOD `double_recon_loss` (`pygod==1.1.0`) | `5294db5d415953e2aed7b52c545fd81a2b98868652649f24614e0fc7950fc1b9` |
| `src/gog_fraud/models/pygod/__init__.py` | Production / verification module | `666e7e6e5878957049d9061e9d29d32740c2acf92ed12805a2cafead8dbb964e` |
| `src/gog_fraud/models/pygod/data_adapter.py` | Production / verification module | `6cf20817145063b5fbb5fef610d26ac81d576c251b05c252bf8f07ab7d2da294` |
| `src/gog_fraud/models/pygod/dlg.py` | Production / verification module | `a869ade9da1e8d52bbb2ffaee37bcf847542915b87f8748adc4450f3389b3a78` |
| `src/gog_fraud/models/pygod/dlg_base.py` | Production / verification module | `fdd964011b67cd1f9180f7e2374417e66da82c13c305acb0644eaac4bca9c888` |
| `src/gog_fraud/models/pygod/dlg_full.py` | Production / verification module | `1d862bf780ab83ba4eb008eecd94cca23258c71d4b87289c268d8beb5482f37b` |
| `src/gog_fraud/models/pygod/dlg_full_base.py` | Production / verification module | `82c357d49ecff6ea572129957687a969df3f26cb248895164a2d8f59d3698aaf` |
| `src/gog_fraud/models/pygod/exact_reconstruction.py` | Production / verification module | `c490d2b1fb43d2ec8247cc0ecbb0d92bbc7b3a831de00ed46733ee21a9bc1037` |
| `src/gog_fraud/models/pygod/gadnr.py` | Production / verification module | `a4253d1530d7957cc22f0e736c7b4625037a14206e333f2288dbd00e2a7264a9` |
| `src/gog_fraud/models/pygod/shared_reconstruction.py` | Production / verification module | `1919079c4e806a79ae817eb170afc3472fc7f0ca0960dbeb12a0383fdbbace8f` |
| `src/gog_fraud/models/pygod/sparse_message.py` | Production / verification module | `5d4b00cbcc56ece5057b432a2a7ce64bcac5accde45620a260465a9113b9c012` |
| `src/gog_fraud/models/pygod/stable_reconstruction.py` | Production / verification module | `713be1fdad8ad77cb30ab2c6ea845ade88bcd9a1d2cea12a7abb792411e9861a` |
| `projects/benchmark/scripts/a22_g1_dense_exact_training_equivalence.py` | Production / verification module | `1e93238fa524163d2808dafb5d121d2c2b51911758d65f7ff1e2de0f35bad5c9` |

---

## 2. Directed vs. Undirected Row-Residual Orientation Identity (`A22-F6` Fix)

For any binary adjacency matrix $A \in \{0, 1\}^{N \times N}$ (whether symmetric/undirected or asymmetric/directed) and latent node embedding matrix $\mathbf{Z} \in \mathbb{R}^{N \times d}$ with row vectors $\mathbf{z}_i \in \mathbb{R}^d$, PyGOD's `double_recon_loss` computes the per-node structural residual as the Euclidean $L_2$ norm of row $i$ of $(A - \mathbf{Z}\mathbf{Z}^\top)$:

$$
\mathcal{A}_{\mathrm{struct}, i}^2 = \|A_{i,:} - (\mathbf{Z}\mathbf{Z}^\top)_{i,:}\|_2^2 = \sum_{j=1}^N \left(A_{ij} - \mathbf{z}_i^\top \mathbf{z}_j\right)^2.
$$

Expanding the quadratic over outgoing neighbor index $j \in \{1, \dots, N\}$ for fixed source row $i$:

$$
\sum_{j=1}^N \left(A_{ij} - \mathbf{z}_i^\top \mathbf{z}_j\right)^2 = \underbrace{\sum_{j=1}^N A_{ij}^2}_{\mathrm{deg}_{\mathrm{out}}(i)} - 2 \mathbf{z}_i^\top \underbrace{\sum_{j=1}^N A_{ij} \mathbf{z}_j}_{(A\mathbf{Z})_i} + \mathbf{z}_i^\top \left(\mathbf{Z}^\top \mathbf{Z}\right) \mathbf{z}_i.
$$

- **Crucial Orientation Note for Directed Graphs:** The cross-term sum over column index $j$ in row $i$ is $\sum_{j=1}^N A_{ij} \mathbf{z}_j = (A\mathbf{Z})_i$ (row $i$ of the matrix product $A\mathbf{Z}$). In the earlier A22 note (`a22_exact_loss_source_trace.md`), an intermediate expression wrote $(\mathbf{Z}^\top A_{:,i})$, which equals $(A^\top \mathbf{Z})_i = \sum_{j=1}^N A_{ji} \mathbf{z}_j$ (the incoming-edge sum) and coincides with $(A\mathbf{Z})_i$ **only** when $A = A^\top$.
- **Production Code & Manuscript Verification:** Both the manuscript (Equation 3: $\mathrm{deg}(i) - 2\mathbf{z}_i^\top (A\mathbf{Z})_i + \mathbf{z}_i^\top (\mathbf{Z}^\top \mathbf{Z})\mathbf{z}_i$) and the production sparse implementation (`exact_reconstruction.py`, which aggregates target embeddings $z_j$ for each directed edge $(i, j)$ into source node $i$ via `neighbor_sum.index_add_(0, row, z[col])`) compute $(A\mathbf{Z})_i$ and `deg_out(i)` (`row` degree), matching `torch.norm(A - Z @ Z.T, dim=1)` identically on both directed and undirected graphs.
- **Asymmetric Directed Graph Unit Test (`a23_g4_asymmetric_orientation_test.json`):** Verified on a strictly asymmetric 12-node directed graph ($A \neq A^\top$):
  - $\max_i \|\sum_j A_{ij} \mathbf{z}_j - (A\mathbf{Z})_i\|_\infty = 0.00e+00$
  - $\max_i |\|A_{i,:} - (\mathbf{Z}\mathbf{Z}^\top)_{i,:}\|_2 - \mathcal{A}_{\mathrm{struct}, i}^{(A\mathbf{Z})_i}| = 1.78e-15$
  - By contrast, substituting $\mathbf{Z}^\top A_{:,i} = (A^\top \mathbf{Z})_i$ produces an orientation error of $4.4777$ on the same asymmetric graph.

---

## 3. Explicit Scope of the 18 CPU Equivalence Unit Tests

The 18 automated unit tests in `projects/benchmark/scripts/a22_g1_dense_exact_training_equivalence.py` (recorded in `projects/benchmark/publication/a22/a22_loss_equivalence_report.json`) verify:
1. Forward per-node structural residual scores $\mathcal{A}_{\mathrm{struct}, i}$ and composite scalar loss $\mathcal{L}$ between PyGOD's dense `double_recon_loss` and our sparse Gram implementation (`exact_reconstruction.py` / `stable_reconstruction.py`) across undirected, directed, isolated-node, self-loop, and $w \in \{0.0, 0.2, 0.5, 0.8, 1.0\}$ synthetic graph fixtures ($N=40$, float32 and float64).
2. Backward parameter gradients $\nabla_\theta \mathcal{L}$ and 10-step Adam optimization trajectories under deterministic initialization on CPU synthetic fixtures ($N=40$).
3. **Boundary of Claim:** These controlled unit tests establish mathematical and algorithmic equivalence of the objective function and gradient step on synthetic test fixtures; they do **not** claim bitwise GPU replay of all 455 historical benchmark runs (which were executed across mixed GPU hardware and cannot be bit-identically replayed due to standard CUDA atomic reduction non-determinism).
