# Cover Letter for MDPI Applied Sciences (A13)

**Date:** 2026-10-24  
**To:**  
Guest Editors, Special Issue **"Machine Learning and Its Application for Anomaly Detection"**  
Dr. Naeem Ayoub and Dr. Amira Mouakher  
Editorial Office, *Applied Sciences*, MDPI  

**Subject:** Submission of Manuscript: *"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"*

Dear Guest Editors and Editorial Board,

We are pleased to submit our original research manuscript titled **"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"** for consideration as a Research Article in the Special Issue **"Machine Learning and Its Application for Anomaly Detection"** of *Applied Sciences* (Special Issue ID: `C67H3U9QGT`).

### Special Issue Fit
Your Special Issue invites rigorous, application-oriented research on unsupervised anomaly detection, financial fraud investigation, cybersecurity defense, and scalable machine learning implementations. Our manuscript directly addresses these core priorities:
1. **Multi-Domain Support-Aware Benchmark:** Evaluates seven functioning unsupervised graph anomaly detectors across 13 primary graph benchmarks (five native financial, eight synthetic injected) and one external enterprise cybersecurity network (LANL-RedTeam) over five random seeds (455 total runs), documenting hardware execution boundaries (84 of 91 primary pairs supported) without heuristic score imputation.
2. **Evaluated Population & Dataset Reconciliation:** Explicitly reconciles evaluated model-input graph tensors ($N, E, d$) against raw source graphs, resolving historical ambiguities across financial and synthetic benchmarks.
3. **Operational Alert-Budget Analysis:** Evaluates precision and recall under strict forensic triage budgets ($P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$) and pairs Average Precision with evaluation prevalence baselines.
4. **Empirical Low-AUC Audit:** Empirically documents below-random ($<0.50$) ROC-AUC values under standard detector orientation on financial transaction graphs, presenting informed topological hypotheses while preserving authentic empirical measurements without test-driven sign inversion.
5. **Exact Closed-Form Decoupling:** Mathematically derives closed-form Gram-matrix reconstructions that eliminate dense $O(N^2)$ memory allocation for unsquashed dot-product decoders, enabling exact full-graph training without sampling approximations.
6. **Actionable Decision Matrix:** Synthesizes topological density, feature availability, and alert budgets into practical guidelines for deploying graph anomaly detection systems.

### Disclosure of Closely Related Preceding Work & Preprint Policy
To ensure transparency and ethical compliance:
- An earlier architectural manuscript introducing the foundational Decoupled Local-to-Global (DLG-GNN) concept for blockchain fraud detection was previously posted as a preprint (Park et al., 2026).
- The present submission is a distinct, comprehensive benchmark study. It evaluates broader comparative GAD methods (DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, and DLG-Aug), investigates multi-domain topologies (financial, social, citation, cyber), contributes exact closed-form algebraic decoders, and provides critical practitioner decision guidelines.
- Prior to journal submission, a publisher-neutral version of this benchmark study has been prepared for deposit on Preprints.org to secure a distinct benchmark preprint DOI. There is no concurrent submission to any other journal.

### Data Availability & Reviewer Access
All aggregate numerical evidence, scripts, and reproduction materials are openly archived under immutable GitHub release tag `benchmark-a13-submission-2026-10-24` at https://github.com/Sam-7878/dlg_gnn. High-resolution node-level score arrays for the 105 cryptocurrency evaluations are author-locally preserved and will be promptly provided to peer reviewers upon request. Historical baseline runs that did not record node score arrays are truthfully disclosed.

### Declarations
- **Conflict of Interest:** The authors declare no conflicts of interest.
- **Use of Generative AI:** During manuscript preparation, Claude Opus 5.5 and OpenAI ChatGPT/Codex were utilized for editorial formatting, grammar refinement, and LaTeX table structuring. All scientific calculations, derivations, empirical results, and scientific interpretations were conceived, verified, and finalized by the authors.

Sincerely,

**Ki-Hyung Kim**, Ph.D. (Corresponding Author)  
Professor, Department of Cyber Security  
Ajou University, Suwon 16499, Republic of Korea  
Email: `kkim86@ajou.ac.kr` | Tel: +82-31-219-2433  

**Seong-Su Park**  
Department of Computer Engineering  
Ajou University, Suwon 16499, Republic of Korea  
Email: `parky@ajou.ac.kr`
