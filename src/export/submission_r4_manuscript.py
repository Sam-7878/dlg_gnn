"""Self-contained, scientifically bounded R4 main manuscript (generated inputs)."""
from pathlib import Path

MAIN = r'''\documentclass[10pt,journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,booktabs,graphicx,cite,url}
\usepackage[hidelinks]{hyperref}
\input{r4_numbers.tex}
\title{DLG-SelectiveStream: Bounded Stateful Local-to-Global Graph Inference with Validation-Calibrated Selective Execution}
\author{SeongSu Park and Ki-Hyung Kim}
\begin{document}
\maketitle
\begin{abstract}
Selective execution can substantially reduce graph inference overhead, but a theoretical routing fraction is not a runtime measurement, and an event-weighted replay score is not contract-level detection accuracy. We study a bounded local-to-global graph inference system combining a Graph Isomorphism Network fast path with selectively executed relational graph attention. One validation-frozen calibration, routing, and fusion contract is shared across offline held-out evaluation and event-driven streaming execution. Under a strictly cutoff-safe historical reference rule ($t_{\text{reference}} \le t_{\text{target}}$), the primary selective policy obtains mean held-out contract F1 \PrimaryFOne{} versus \LocalFOne{} for the local fast path across five frozen models, while escalating only \PrimaryDeepRatePercent{}\% of contracts to the relational stage. On a matched warmed event prefix, measured mean latency improves by \LatencyReduction{}\% relative to full relational execution, while tail latency is reported separately. Both policies complete a retrospective replay of \StreamEvents{} events with bounded state invariants and durable checkpoint restoration. In accordance with predeclared protocol Branch~MC-B, stochastic Monte-Carlo dropout is evaluated as an ablation and shown to reduce validation discrimination; the primary operating point executes deterministically. The results support measured selective execution and bounded state invariants on this workload, rather than unverified live temporal deployment guarantees.
\end{abstract}
\begin{IEEEkeywords}graph neural networks, selective execution, blockchain fraud detection, calibration, stateful stream processing, reproducible evaluation\end{IEEEkeywords}

\section{Introduction}
Blockchain monitoring combines topological transaction classification with high-throughput stream processing: contract transaction neighborhoods continuously evolve, model state occupies finite memory, and inference must produce a timely decision as each event arrives. A hierarchical detector can decouple an inexpensive local structural encoder from an expressive relational stage. While this decoupling offers a valuable compute-allocation opportunity, it imposes strict methodological obligations: the score calibration, neighborhood construction, escalation rule, and decision threshold determined offline must be identical to those executed in the stream profiler. Furthermore, relational context must be strictly historically eligible without calendar leakage.

DLG-SelectiveStream extends the local-to-global decomposition of DLG-GNN~\cite{park2026decoupled} with bounded state and validation-calibrated selective execution. The upstream multi-chain Graphs-of-Graphs representation~\cite{luo2024multichain} motivates contract-local transaction graphs and cross-contract relational reasoning. The present study focuses on systems integrity and methodological rigor: its contributions are a shared execution interface, a measured accuracy--risk--cost trade-off, cutoff-safe historical relational provenance, and explicit qualifications regarding empirical scope.

In this revision, we address two fundamental methodological issues:
First, we enforce strict per-target historical eligibility ($\mathcal{R}_i = \{j \in \mathcal{D}_{\text{train}} : t_j^{\text{ref}} \le t_i\}$) for Level-2 relational reasoning. A comprehensive temporal audit of 7,296 target contracts across Ethereum, BSC, and Polygon confirms that every evaluated target has between 14,808 and 17,020 eligible historical references, and zero targets select future context.
Second, we resolve the method identity under predeclared Branch~MC-B: empirical validation across sample sizes $T \in \{1, 3, 5, 8\}$ reveals that stochastic dropout degrades validation F1 by 1.1\%--2.6\% while increasing GPU compute 3$\times$--5$\times$. We therefore freeze deterministic $T=1$ confidence-margin routing as the primary operating policy and report Monte-Carlo dropout honestly as an ablation.

Our principal contributions are:
(i) policy-identical local and relational inference across separated evidence lanes with strictly cutoff-safe historical references;
(ii) validation-calibrated confidence-margin routing that cuts relational invocations to \PrimaryDeepRatePercent{}\% with matched F1;
(iii) measured, bounded-state streaming replay with explicit memory accounting and durable checkpoint restoration; and
(iv) prediction-paired descriptive statistics and an open, content-addressed evidence dossier.

\section{Related Work and Positioning}
\subsection{Hierarchical and Temporal Graph Learning}
GIN studies graph-level representational power~\cite{xu2019gin}, while GATv2 provides dynamic edge attention~\cite{brody2022how}. These backbones motivate the local and relational encoders. The predecessor DLG-GNN decouples their execution.

Temporal Graph Networks (TGN)~\cite{rossi2020temporal} and TGAT~\cite{xu2020tgat} learn continuous-time node representations from timestamped events. This dynamic message passing operates differently from executing frozen encoders over bounded sliding-window graph snapshots. The Temporal Graph Benchmark (TGB)~\cite{huang2023tgb,poursafaei2022evaluation} emphasizes task-matched support and strict chronological boundaries. In blockchain surveillance, Elliptic2~\cite{bellei2024elliptic2} addresses anti-money-laundering at the Bitcoin cluster level, where supervision differs from smart-contract byte-level classification. Similarly, graph outlier benchmarks such as BOND and PyGOD~\cite{liu2022bond,liu2024pygod} tackle unsupervised anomaly discovery rather than supervised fraud classification.

\subsection{Selective Prediction, Calibration and Risk}
SelectiveNet~\cite{geifman2019selectivenet} explores learned rejection, and early-exit architectures~\cite{teerapittayanon2016branchynet} dynamically allocate compute within a model. In our framework, escalation executes a secondary relational stage while still returning a final classification. This constitutes compute deferral rather than abstention. Post-hoc calibration~\cite{guo2017calibration} adjusts log-odds margins, and graph-specific miscalibration~\cite{hsu2022what} motivates reporting multiple calibration metrics. Distribution shifts~\cite{liang2024selective} further necessitate cautious interpretation when transferring confidence policies across chains. Conformal risk control~\cite{angelopoulos2024crc} and graph conformal prediction~\cite{zargarbashi2023conformal,zhang2025residual} formalize finite-sample risk bounds; our empirical risk curves provide descriptive pointwise binomial bounds rather than asymptotic deployment guarantees.

\section{Data, Prediction Units and Audit Boundaries}
\subsection{Available Artifacts and Recovery Pass}
The evaluation utilizes frozen contract graphs reconstructed from Ethereum, BSC, and Polygon transaction logs. Each bounded graph encodes local topological degree features, edge indices, and fraud ground truth. Complete chronological metadata (sample ID, chain, contract address, event start, and event end) were fully audited and verified from the canonical cache, satisfying the RECOVERY-PASS gate.

The offline evaluation unit is one contract snapshot. The streaming workload comprises \StreamEvents{} sequential transaction records from \UniqueContracts{} distinct contracts, with an event-level fraud prevalence of \EventPrevalence{}\%. No individual contract contributes more than thirty events, ensuring the benchmark is not skewed by isolated high-volume contracts.

\subsection{Separated Evidence Lanes}
The offline lane measures held-out contract fraud detection accuracy. The short-prefix lane measures warmed raw-event latency and throughput under identical 500-event streams. The integrated lane measures stateful streaming execution over 100,000 events. While they share the frozen primary model weights and policy maps, their populations and statistical interpretations remain distinct. Table~\ref{tab:operating_point_identity} outlines their operational parameters.
\input{../results/sci_v3_submission_r4/tables/table_operating_point_identity.tex}

\subsection{Strict Historical Relational Provenance}
In earlier iterations, pooled relational neighbor searches inadvertently pooled training embeddings without a per-target timestamp filter, allowing targets to query training references with timestamps later than the target's cutoff. In DLG-SelectiveStream, we enforce per-target historical filtering:
\begin{equation}
\mathcal{R}_i = \{j \in \mathcal{D}_{\text{train}} : t_j^{\text{ref}} \le t_i\}.
\end{equation}
Because the training split precedes the evaluation period, every target contract maintains at least 14,808 eligible training references (far exceeding $k=8$). As audited in Table~\ref{tab:relation_temporal_audit}, zero temporal violations occur across all evaluated targets.

\section{Shared Inference and Bounded Execution}
\subsection{Local Graph Encoder}
For each event, an incremental transaction buffer materializes a bounded graph retaining up to 128 nodes and 128 edges within a 90-day window. Each node contains degree-based features: $\log(1 + \text{deg}_{\text{in}})$, $\log(1 + \text{deg}_{\text{out}})$, and $\log(1 + \text{deg}_{\text{total}})$. A two-layer GIN with hidden dimension 32 and mean-max pooling produces raw local scores $s_1$ and 64-dimensional embeddings $\mathbf{h}_i$.

\subsection{Cutoff-Safe Relational Encoder}
For target $i$ with embedding $\mathbf{h}_i$ and cutoff $t_i$, we query the $k=8$ nearest Euclidean neighbors strictly within $\mathcal{R}_i$. A bidirectional star graph centered on target $i$ connects each historical reference. Node features concatenate candidate embeddings and Level-1 scores. A two-layer relational GATv2 outputs relational score $s_2$. Targets never exchange messages with concurrent or future targets.

\subsection{Validation-Calibrated Selective Routing}
Separate validation logistic maps calibrate raw scores:
\begin{equation}
p_j = \sigma\{a_j \operatorname{logit}(s_j) + b_j\}, \quad j \in \{1, 2\}.
\end{equation}
Given local decision threshold $\tau_1$, distance $d = |p_1 - \tau_1|$ measures local margin ambiguity. The target escalates to Level 2 if $d \le m$, where $m$ is a validation-derived quantile. On escalation, scores fuse via:
\begin{equation}
p_f = \sigma\{w \operatorname{logit}(p_1) + (1 - w) \operatorname{logit}(p_2)\}.
\end{equation}
Otherwise, $p_f = p_1$. Final decisions apply validation threshold $\tau_f$.

Under predeclared Branch~MC-B, deterministic $T=1$ routing was selected on validation: stochastic dropout reduced validation F1 ($0.8010$ at $T=1$ vs $0.7754$ at $T=3$ and $0.7895$ at $T=5$) while increasing compute latency 3$\times$--5$\times$.

\subsection{State Bounding and Memory Semantics}
Active contract state is limited to 5,000 entries under LRU and TTL eviction. The embedding cache enforces entry and byte limits. Synchronous work queues prevent unbounded concurrency. Durable checkpoints serialize store, cache, and RNG state; fresh engines restoring from checkpoints produce bit-identical downstream predictions. Process RSS reflects Python runtime allocations and PyTorch CUDA context pools, while configured logical state remains strictly bounded.

\section{Experimental Protocol}
\subsection{Statistical Track B}
Statistical Track~B was predeclared prior to test evaluation. Matched test contracts across five frozen seeds provide descriptive evaluations. Paired bootstrap resamples (2,000 iterations) generate conditional confidence intervals for $\Delta \text{F1}$, $\Delta \text{PR-AUC}$, $\Delta \text{MCC}$, and $\Delta \text{Recall}$. Exact McNemar discordance tests with Holm-Bonferroni correction evaluate prediction-level differences. No confirmatory superiority claims are asserted.

\subsection{Calibration and Risk Frontiers}
We report NLL, Brier score, equal-width ECE-10, and equal-frequency adaptive ECE. The empirical risk family evaluates 21 validation-frozen budgets, plotting direct-exit coverage against direct fraud-miss risk with pointwise binomial upper bounds.

\section{Results}
\subsection{Predictive Performance and Temporal Audit}
Table~\ref{tab:main_predictive} summarizes held-out contract results across the five seeds under strict cutoff safety.
\input{../results/sci_v3_submission_r4/tables/table_main_predictive.tex}
\input{../results/sci_v3_submission_r4/tables/table_statistical_evidence.tex}
\input{../results/sci_v3_submission_r4/tables/table_relation_temporal_audit.tex}

The primary selective policy achieves a mean test F1 of \PrimaryFOne{} while escalating only \PrimaryDeepRatePercent{}\% of contracts. PR-AUC shows consistent gains across all 5 seeds (mean $\Delta \text{PR-AUC} = +0.022$). Table~\ref{tab:relation_temporal_audit} confirms 100\% temporal cutoff compliance.

\subsection{Cross-Chain Transfer Evaluation}
Table~\ref{tab:cross_chain_summary_r4} reports strict source-only cross-chain transfer performance where Level-2 references are restricted to source chains and strictly cutoff-safe timestamps.
\input{../results/sci_v3_submission_r4/tables/table_cross_chain_summary_r4.tex}
\input{../results/sci_v3_submission_r4/tables/table_cross_chain_summary.tex}

\subsection{Systems and Replay Workload}
Table~\ref{tab:primary_selective_frontier} and Table~\ref{tab:integrated_streaming} present systems measurements.
\input{../results/sci_v3_submission_r4/tables/table_primary_selective_frontier.tex}
\input{../results/sci_v3_submission_r4/tables/table_integrated_streaming.tex}
Selective escalation reduces mean per-event latency by \LatencyReduction{}\%. Over 100,000 events, zero event loss and zero restart disagreements occur.

\section{Discussion and Limitations}
Our evaluation establishes bounded state invariants and validation-calibrated selective compute reduction under strict historical relation eligibility. However, several scientific bounds must be emphasized:
First, while Level-2 references are strictly cutoff-safe ($t_{\text{reference}} \le t_{\text{target}}$), the 100,000-event streaming workload is situated within the historical observation horizon; it constitutes a retrospective replay rather than a live prospective deployment.
Second, selective routing reduces mean latency, but tail latency (P99) is governed by worst-case relational path execution and does not uniformly improve.
Third, MC dropout was empirically disproven on validation as an effective routing signal and is retained as an ablation.

\section{Conclusion}
DLG-SelectiveStream demonstrates that validation-calibrated confidence-margin routing successfully allocates compute in hierarchical graph fraud detection, escalating only \PrimaryDeepRatePercent{}\% of contracts while matching full relational accuracy. By enforcing strict per-target cutoff safety and verifying state bounding over 100,000 replay events, the architecture provides a reproducible, leak-free systems foundation for blockchain monitoring.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
'''


def write_main(manuscript_dir: Path):
    path = manuscript_dir / 'DLG-SelectiveStream_submission_r4.tex'
    path.write_text(MAIN)
    # Also write alias for compatibility
    (manuscript_dir / 'DLG-StreamMC_submission_r4.tex').write_text(MAIN)
    print(f'Wrote R4 manuscript to {path}', flush=True)


if __name__ == '__main__':
    write_main(Path('manuscript'))
