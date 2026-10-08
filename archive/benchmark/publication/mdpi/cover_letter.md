# Cover Letter for Manuscript Submission to MDPI *Applied Sciences*

**Date:** September 23, 2026  

**To:**  
The Editor-in-Chief and Guest Editors  
Special Issue: *"Graph Neural Networks: Theory, Methods and Applications"*  
Section: Computing and Artificial Intelligence  
*Applied Sciences* (MDPI)  

**Subject:** Submission of Research Article: *"A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs"*  

Dear Editors,

We are pleased to submit our research article titled **"A Reproducible and Scalability-Aware Benchmark for Graph Anomaly Detection with Decoupled Local-to-Global GNNs"** for consideration as an original research article in the Special Issue **"Graph Neural Networks: Theory, Methods and Applications"** within the *Computing and Artificial Intelligence* section of *Applied Sciences*.

### 1. Alignment with the Special Issue Scope
Graph neural networks have emerged as indispensable architectures for analyzing complex relational dependencies in financial systems, cybersecurity telemetry, and large-scale networks. The Special Issue explicitly calls for contributions in:
- *Scalable and efficient GNN architectures*
- *Self-supervised and unsupervised graph learning*
- *Graph representation learning and embeddings*
- *Financial modeling and other applied GNN settings*

Our study directly aligns with these core themes by developing mathematically exact, scalable sparse GNN backends for unsupervised graph anomaly detection, evaluating them across financial transaction networks, and providing complete reproducible benchmarks on attributed graphs.

### 2. Key Scientific Contributions
This manuscript makes four principal contributions to the field:
1. **Multi-Dataset Rigorous Benchmark**: We establish a frozen, reproducible benchmark across ten primary graphs, including three real-label financial/blockchain datasets (Elliptic, DGraphFin, and BitcoinOTC) and seven controlled synthetic-injection datasets (Yelp-Syn, Amazon-Syn, Flickr-Syn, Reddit-Syn, Cora-Syn, CiteSeer-Syn, and PubMed-Syn; constructed from public base graphs Yelp, Amazon, Flickr, Reddit, Cora, CiteSeer, and PubMed), evaluating eight detector configurations (DOMINANT, AnomalyDAE, CoLA, CONAD, GADNR, OCGNN, DLG-Base, and DLG-Aug) over five random seeds (355 successful primary runs).
2. **Exact Semantics-Preserving Scalability Backends**: To overcome the memory and computation bottlenecks of large graphs without altering historical detector loss functions, we derive, implement, and numerically verify a mathematically equivalent sparse reformulation of the linear dot-product reconstruction objective and implement a fused sparse message-passing backend that exactly matches reference forward passes, embedding gradients, and optimizer updates.
3. **Decomposition and Dataset-Dependent Augmentation**: We decompose the decoupled local-to-global architecture into a historical non-augmentation reconstruction baseline (DLG-Base) and a feature-augmented detector (DLG-Aug). While DLG-Aug achieves the top average rank in the fraud-oriented common subset, the primary benchmark and targeted sensitivity analyses show that the utility of local neighborhood augmentation is dataset dependent (improving PR-AUC on Elliptic by $+0.0350$, but reducing it on dense graphs such as Reddit-Syn by $-0.0656$).
4. **External Cybersecurity Validation & Sensitivity Controls**: We extend our findings to enterprise cybersecurity through external validation on the 16,694-computer LANL-RedTeam authentication graph (where DLG-Base also outperforms DLG-Aug) and execute 45 targeted sensitivity-control experiments across auxiliary zero-information inputs, feature permutation, and training budgets.

### 3. Preprint Declaration and Reproducibility
In accordance with MDPI editorial guidelines regarding open science and preprints:
- A publisher-neutral preprint version of this manuscript is deposited on Preprints.org under a CC BY 4.0 license.
  - Preprint Status: *Preprints.org deposit initiated; DOI will be declared upon official deposit confirmation.*
  - Companion Repository: `https://github.com/Sam-7878/dlg_gnn` (Release tag: `v1.0.0-preprint`).
- Mode 1 reproduction scripts operate entirely on packaged frozen artifacts and require no raw graph data downloads. Mode 2 experimental runners, complete source code, exact sparse backends, and verification suites are made openly accessible.

### 4. Author Declarations
- This manuscript represents original research and is not under consideration for publication elsewhere.
- All authors have reviewed and approved the manuscript for submission to *Applied Sciences*.
- The authors declare no conflicts of interest regarding this research.
- All financial support and IITP grant funding have been fully disclosed.

Thank you very much for your time and consideration of our manuscript.

Sincerely,

**Prof. Ki-Hyung Kim** (Corresponding Author)  
Department of Cyber Security, Ajou University  
Worldcup-ro 206, Yeongtong-gu, Suwon 16499, Republic of Korea  
Email: `kkim86@ajou.ac.kr` | Tel: +82-31-219-2433  
ORCID: `0000-0002-2321-4475`  

**SeongSu Park**  
Department of Computer Engineering, Ajou University  
Email: `parky@ajou.ac.kr`  
ORCID: `0009-0008-4056-3875`  
