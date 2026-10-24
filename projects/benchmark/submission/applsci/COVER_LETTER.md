# Cover Letter for MDPI Applied Sciences

**Date:** 2026-10-24  
**To:**  
Guest Editors, Special Issue **"Machine Learning and Its Application for Anomaly Detection"**  
Dr. Naeem Ayoub and Dr. Amira Mouakher  
Editorial Office, *Applied Sciences*, MDPI  

**Subject:** Submission of Manuscript: *"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"*

Dear Guest Editors and Editorial Board,

We are pleased to submit our original research manuscript titled **"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"** for consideration as a Research Article in the Special Issue **"Machine Learning and Its Application for Anomaly Detection"** of *Applied Sciences*.

### Special Issue Fit
Your Special Issue invites cutting-edge research on unsupervised anomaly detection, financial fraud investigation, cybersecurity defense, and scalable machine learning implementations. Our manuscript directly addresses these core themes by providing:
1. An empirical benchmark evaluating seven functioning unsupervised graph anomaly detectors across 13 primary benchmark graphs (five native financial, eight synthetic injected) and one external enterprise cybersecurity network (LANL-RedTeam) over five random seeds (455 total runs).
2. A support-aware evaluation protocol that documents hardware feasibility and out-of-memory boundaries (84/91 primary pairs supported) without heuristic score imputation.
3. An operational alert-budget analysis (evaluating precision and recall at top-1\% and top-5\% investigation workloads) and a diagnostic audit explaining why unsupervised reconstruction baselines suffer below-random ($<0.50$) ROC-AUC on financial transaction graphs.
4. Closed-form exact Gram-matrix decoders that reduce memory complexity from $O(N^2)$ to linear in edge density without sampling approximations, enabling full-graph training on graphs with millions of nodes.
5. An actionable practitioner decision matrix guiding model selection based on topological density, feature availability, and investigative capacity.

### Disclosure of Closely Related Preceding Work
To maintain transparency and ethical compliance:
- An earlier architectural manuscript introducing the foundational Decoupled Local-to-Global (DLG-GNN) concept for blockchain fraud detection was previously posted as a preprint (Park et al., 2026).
- The present submission is a distinct, comprehensive benchmark study. It evaluates broader comparative GAD methods (DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, and DLG-Aug), investigates multi-domain topologies (financial, social, citation, cyber), contributes exact closed-form algebraic decoders, and provides critical practitioner decision guidelines.
- Prior to journal submission, a publisher-neutral version of this benchmark study has been prepared for deposit on Preprints.org to secure a distinct benchmark preprint DOI. There is no concurrent submission to any other journal.

### Data Availability & Reviewer Access
All aggregate numerical evidence, scripts, and reproduction materials are openly archived under immutable GitHub release tag `benchmark-a09-2026-10-10` (commit `2cc6f85afc44830eb7eee92d709c6eb8d567ba2d`). High-resolution node-level score arrays and execution checkpoints are preserved author-locally and will be promptly provided to peer reviewers upon request.

All authors have approved the manuscript and agree with its submission to *Applied Sciences*. We look forward to your feedback and thank you for your consideration.

Sincerely,

**Ki-Hyung Kim**, Ph.D. (Corresponding Author)  
Professor, Department of Cyber Security  
Ajou University, Suwon 16499, Republic of Korea  
Email: `kkim86@ajou.ac.kr` | Tel: +82-31-219-2433  

**Seong-Su Park**  
Department of Computer Engineering  
Ajou University, Suwon 16499, Republic of Korea  
Email: `parky@ajou.ac.kr`
