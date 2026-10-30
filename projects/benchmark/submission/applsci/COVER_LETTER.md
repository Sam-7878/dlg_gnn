# Cover Letter for MDPI Applied Sciences (A20 Final Submission)

**Date:** 2026-10-29  
**To:**  
Guest Editors, Special Issue **"Machine Learning and Its Application for Anomaly Detection"**  
Dr. Naeem Ayoub and Dr. Amira Mouakher  
Editorial Office, *Applied Sciences*, MDPI  

**Subject:** Submission of Manuscript: *"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"*

Dear Guest Editors and Editorial Board,

We are pleased to submit our original research manuscript titled **"Scalability-Aware Benchmarking of Graph Anomaly Detection on Financial and Cyber Networks: Evaluated Population, Alert Budgets, and the Exact Limits of Decoupled GNNs"** for consideration as a Research Article in the Special Issue **"Machine Learning and Its Application for Anomaly Detection"** of *Applied Sciences* (Special Issue ID: `C67H3U9QGT`, deadline 2026-12-31).

### Special Issue Fit & Scientific Contributions
Your Special Issue invites rigorous, application-oriented research on unsupervised anomaly detection, financial fraud investigation, cybersecurity defense, and scalable machine learning implementations. Our manuscript directly addresses these core priorities:
1. **Multi-Domain Support-Aware Benchmark:** Evaluates seven functioning unsupervised graph anomaly detectors across 13 primary graph benchmarks (five native financial, eight synthetic injected) and one external enterprise cybersecurity network (LANL-RedTeam) over five random seeds (455 total runs: 420 primary and 35 external runs), documenting hardware execution boundaries (84 of 91 primary pairs supported) without heuristic score imputation.
2. **Evaluated Population & Prevalence Ledger:** Formulates a programmatic audit ledger separating raw upstream graph totals and full-graph target prevalence from evaluated test-set positive rates (Ethereum 0.4089, BSC 0.1473, Polygon 0.0259), anchoring Average Precision comparisons strictly to empirical test prevalence baselines.
3. **Hardware Execution Envelope:** Explicitly audits execution failures from `support_14x7.csv`, categorizing the 7 unsupported primary pairs into three exact provenance classes: 4 historical execution limitation pairs for GADNR (Elliptic, DGraphFin, Yelp-Syn, Reddit-Syn; historical execution failure / numerical covariance determinant limitation), 1 historical out-of-memory pair for GADNR (Flickr-Syn on an 8-GB legacy environment), and 2 out-of-memory pairs for AnomalyDAE (Yelp-Syn, Reddit-Syn; CUDA OutOfMemoryError in the full encoder forward pass on the 24-GB RTX 3090).
4. **Operational Alert-Budget Analysis:** Evaluates precision and recall under strict forensic triage budgets ($P@1\%$, $R@1\%$, $P@5\%$, $R@5\%$) and pairs Average Precision with evaluation prevalence baselines.
5. **Empirical Low-AUC Audit:** Empirically documents below-random ($<0.50$) ROC-AUC values under standard detector orientation on financial transaction graphs, presenting informed topological hypotheses while preserving authentic empirical measurements without test-driven sign inversion.
6. **Exact Closed-Form Decoupling:** Mathematically derives closed-form Gram-matrix reconstructions that eliminate dense $O(N^2)$ memory allocation for unsquashed dot-product decoders, enabling exact full-graph training without sampling approximations.
7. **Actionable Decision Matrix:** Synthesizes topological density, feature availability, and alert budgets into practical guidelines for deploying graph anomaly detection systems.

### Disclosure of Closely Related Preceding Work & Preprint Policy
To ensure transparency and ethical compliance:
- An earlier architectural manuscript introducing the foundational Decoupled Local-to-Global (DLG-GNN) concept for blockchain fraud detection was previously posted as a preprint (Park et al., 2026, doi:10.20944/preprints202610.1878.v1).
- The present submission is a distinct, comprehensive benchmark study. It evaluates broader comparative GAD methods (DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, and DLG-Aug), investigates multi-domain topologies (financial, social, citation, cyber), contributes exact closed-form algebraic decoders, and provides critical practitioner decision guidelines.
- Prior to journal submission, a publisher-neutral version of this benchmark study has been prepared for deposit on Preprints.org to secure a distinct benchmark preprint DOI. There is no concurrent submission to any other journal.

### Data Availability & Reviewer Access
All aggregate numerical evidence, scripts, and reproduction materials are openly archived under immutable Git tag `benchmark-a22-submission-2026-10-30` (superseding `benchmark-a20-preprint-2026-10-29`, `benchmark-a19-submission-2026-10-28`, and `benchmark-a16-submission-2026-10-28`) at https://github.com/Sam-7878/dlg_gnn. High-resolution node-level score arrays for the 105 cryptocurrency evaluations are author-locally preserved and will be promptly provided to peer reviewers upon request. Historical baseline runs that did not record node score arrays are truthfully disclosed.

### Declarations
- **Author Contributions:** Conceptualization, S.-S.P. and K.-H.K.; methodology, S.-S.P. and K.-H.K.; software, S.-S.P.; validation, S.-S.P. and K.-H.K.; formal analysis, S.-S.P.; investigation, S.-S.P.; data curation, S.-S.P.; writing—original draft preparation, S.-S.P.; writing—review and editing, S.-S.P. and K.-H.K.; visualization, S.-S.P.; supervision, K.-H.K.; project administration, S.-S.P. and K.-H.K.; funding acquisition, K.-H.K. All authors have read and agreed to the published version of the manuscript.
- **Conflict of Interest:** The authors declare no conflicts of interest.
- **Acknowledgments / Use of Generative AI:** During the preparation of this manuscript, the authors utilized Claude Opus, Google Gemini/AntiGravity and OpenAI ChatGPT/Codex tools for initial drafting review, script formatting, translation of the text, and LaTeX table structuring. All scientific data, mathematical derivations, empirical analyses, and manuscript conclusions were conceived, independently verified, and finalized under the direct oversight of the authors, who take full responsibility for the integrity of the published work.

Sincerely,

**Ki-Hyung Kim**, Ph.D. (Corresponding Author)  
Professor, Department of Cyber Security  
Ajou University, Suwon 16499, Republic of Korea  
Email: `kkim86@ajou.ac.kr` | Tel: +82-31-219-2433  

**Seong-Su Park**  
Department of Computer Engineering  
Ajou University, Suwon 16499, Republic of Korea  
Email: `parky@ajou.ac.kr`

