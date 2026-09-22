# Exact Sparse Graph Reconstruction Formulation

This document provides the canonical mathematical derivation, implementation identity, and complexity analysis for exact sparse graph reconstruction in the DLG-GNN benchmark suite.

---

## 1. Problem Formulation and Dense Reference

Let $G = (V, E)$ be an attributed graph with $N = |V|$ nodes and $E = |E|$ edges (with adjacency matrix $A \in \{0, 1\}^{N \times N}$, binary and undirected, with optional self-loops).
Let $Z \in \mathbb{R}^{N \times d}$ denote the learned latent node representation matrix, where row $z_i \in \mathbb{R}^{1 \times d}$ is the embedding for node $i$.

In dot-product graph autoencoders (such as DOMINANT, CONAD, DLG-Base, and DLG-Aug), the reconstructed adjacency matrix is:
$$\hat{A} = Z Z^\top \in \mathbb{R}^{N \times N}$$
and the reconstructed $i$-th row is:
$$\hat{A}_{i,:} = z_i Z^\top \in \mathbb{R}^{1 \times N}$$

The per-node structural reconstruction loss under mean squared error is defined as the squared Euclidean distance between the ground-truth adjacency row $A_{i,:}$ and its reconstruction $\hat{A}_{i,:}$:
$$L_{\mathrm{struct}}^{(i)} = \|A_{i,:} - \hat{A}_{i,:}\|_2^2 = \|A_{i,:} - z_i Z^\top\|_2^2$$

Direct materialization of $\hat{A} = Z Z^\top$ requires an $N \times N$ dense matrix, consuming $O(N^2)$ memory (e.g., $1.2M \times 1.2M \times 4\text{ bytes} \approx 5.76\text{ TB}$ on DGraphFin), leading to immediate out-of-memory (OOM) failures.

---

## 2. Canonical Exact Sparse Identity

Expanding the squared $L_2$ norm yields:
$$\|A_{i,:} - z_i Z^\top\|_2^2 = \|A_{i,:}\|_2^2 - 2 A_{i,:} (z_i Z^\top)^\top + \|z_i Z^\top\|_2^2$$

We analyze the three terms individually:

### Term 1: Ground-Truth Degree Norm
For binary adjacency graphs where $A_{ij} \in \{0, 1\}$:
$$\|A_{i,:}\|_2^2 = \sum_{j=1}^N A_{ij}^2 = \sum_{j=1}^N A_{ij} = d_i$$
where $d_i$ is the node degree (or row sum).

### Term 2: Sparse-Dense Cross Term
The inner product simplifies using the sparsity pattern of row $i$:
$$A_{i,:} (z_i Z^\top)^\top = A_{i,:} (Z z_i^\top) = (A_{i,:} Z) z_i^\top = \sum_{j: A_{ij} \neq 0} A_{ij} (z_i z_j^\top)$$
Since $A_{i,:} Z \in \mathbb{R}^{1 \times d}$ is the sparse neighborhood aggregation of embeddings into node $i$, it is computed across all nodes in a single sparse-dense matrix multiplication:
$$S = A Z \in \mathbb{R}^{N \times d}$$
Taking the row-wise inner product with $z_i$ gives:
$$A_{i,:} (z_i Z^\top)^\top = \sum_{k=1}^d S_{ik} Z_{ik} = (S \odot Z) \mathbf{1}_d$$

### Term 3: Gram Quadratic Form
The squared norm of the reconstructed row is:
$$\|z_i Z^\top\|_2^2 = (z_i Z^\top)(z_i Z^\top)^\top = z_i (Z^\top Z) z_i^\top$$

Let $G = Z^\top Z \in \mathbb{R}^{d \times d}$ be the feature Gram matrix. Once $G$ is computed (cost $O(N d^2)$), the $i$-th quadratic form is evaluated in $O(d^2)$:
$$\|z_i Z^\top\|_2^2 = z_i G z_i^\top = \sum_{a=1}^d \sum_{b=1}^d Z_{ia} G_{ab} Z_{ib} = ((Z G) \odot Z) \mathbf{1}_d$$

### Exact Closed-Form Identity:
$$\|A_{i,:} - z_i Z^\top\|_2^2 = d_i - 2 \sum_{j: A_{ij} \neq 0} A_{ij} (z_i z_j^\top) + z_i (Z^\top Z) z_i^\top$$

---

## 3. Computational Complexity and Memory Guarantees

- **Arithmetic Complexity**:
  1. Gram matrix computation $G = Z^\top Z$: $O(N d^2)$ operations.
  2. Sparse neighborhood aggregation $S = A Z$: $O(E d)$ operations.
  3. Row quadratic forms and elementwise reductions: $O(N d^2 + N d)$ operations.
  - **Total Arithmetic Complexity**: $O(E d + N d^2)$.

- **Memory Consumption**:
  - **Dense intermediate storage avoided**: $O(N^2)$ (never allocated).
  - **Core stored graph and embedding memory**: $O(E + N d + d^2)$.
  - **Peak temporary working memory**: $O(N d + d^2)$.

---

## 4. Distinction Against Erroneous Approximations

Notice that:
$$z_i (Z^\top Z) z_i^\top \neq \|Z\|_F^2 \|z_i\|_2^2$$
in general. The identity $\|z_i Z^\top\|_2^2 = \|Z\|_F^2 \|z_i\|_2^2$ would only hold under the pathological condition that $Z^\top Z = \frac{\|Z\|_F^2}{d} I$, which never occurs in learned neural representations. The benchmark implementation strictly uses the exact Gram quadratic form $z_i (Z^\top Z) z_i^\top$.
