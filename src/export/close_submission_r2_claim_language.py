"""Apply conservative claim-language closure after evidence integration."""
from __future__ import annotations

import re
from pathlib import Path


def main() -> None:
    path = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC/DLG-StreamMC.tex")
    text = path.read_text(encoding="utf-8")
    text = text.replace("% \\usepackage{microtype}\n", "")
    text = text.replace("unresolved placeholders", "unresolved manuscript tokens")
    text = text.replace("paper-table placeholder or missing-value rules", "paper-table missing-value and unresolved-token rules")
    text = text.replace("This improvement is reported as descriptive", "This observed difference is reported as descriptive")
    text = text.replace("XGBoost/LightGBM/Production-GIN fast paths and cascades", "XGBoost/LightGBM fast-path controls and the Production-GIN cascade")
    discussion = r"""The final submission evidence supports DLG-StreamMC as a \emph{bounded stateful local-to-global graph inference architecture with selective relational escalation}. After validation-only calibration and operating-point selection, the production GIN fast path obtained mean F1 $0.600$ and the selective GATv2 cascade obtained $0.621$ (observed $\Delta$F1 $=+0.021$). The seed-bootstrap interval includes zero and only one of five seed-level exact McNemar tests remains significant after Holm correction. The difference is therefore descriptive, not confirmatory evidence that relational escalation improves F1.

The tabular cascade comparison is also deliberately fail closed. XGBoost and LightGBM remain strong fast-path controls, but their frozen validation feature matrix is unavailable; revised cascade operating points are neither selected on test labels nor presented as new evidence. The systems claim is independent and bounded: existing raw-event measurements demonstrate actual deep-stage skipping, and the 100,000-event replay demonstrates loss-free execution with bounded configured state and reproducible restart under the evaluated environment."""
    text = re.sub(r"The final submission evidence supports DLG-StreamMC.*?Accordingly, the contribution is better described as \\emph\{measured selective execution\} than as an assumed compute reduction inferred from routing coverage\.",
                  lambda _: discussion, text, count=1, flags=re.DOTALL)
    text = text.replace("Relational escalation can improve detection &\nProduction-GIN cascade mean F1 $0.621$; $\\Delta$F1 $=0.021$ &\nThe gain is conditional; deeper reasoning is not assumed to help every sample",
                        "Relational escalation has a positive observed mean difference &\nProduction-GIN cascade mean F1 $0.621$; observed $\\Delta$F1 $=+0.021$ &\nDescriptive only: the seed-bootstrap interval includes zero and Holm-controlled support is insufficient")
    text = text.replace("The final production result strengthens this interpretation because the production-GIN cascade improves mean F1 while true selective execution also lowers mean end-to-end latency. Detection quality and runtime therefore do not have to move in opposite directions for every operating point.",
                        "The final production result yields a positive observed mean F1 difference while true selective execution lowers mean end-to-end latency. Because the F1 difference is not confirmatory after multiplicity control, it motivates further repeated temporal evaluation rather than a general accuracy-improvement claim.")
    text = text.replace("In the final frozen production comparison, the production GIN is the strongest evaluated fast path and its relational cascade provides the observed positive mean F1 gain. This result justifies the DLG-specific production path under the current protocol.",
                        "In the final frozen production comparison, the production GIN is the evaluated production fast path and its relational cascade has a positive but statistically inconclusive mean F1 difference. This supports retaining the path as the evaluated architecture, not claiming a confirmed accuracy gain.")
    conclusion = r"""The final evidence supports three bounded conclusions. First, after validation-only calibration, the production GIN fast path achieved mean F1 $0.600$ and selective relational escalation achieved $0.621$ (observed $\Delta$F1 $=+0.021$); the difference is descriptive because its seed-bootstrap interval includes zero and Holm-controlled paired evidence is insufficient. Second, raw-event traces demonstrate actual selective execution and expose mean and tail latency, throughput, route rate, RSS, and VRAM instead of inferring savings from route coverage. Third, an integrated 100,000-event replay executed the fast path, router, optional deep stage, fusion, bounded store, cache, and queue with zero event loss and zero checkpoint/restart prediction disagreement. Tabular cascade claims are excluded under FAIL-C because their frozen validation features are unavailable and test-label tuning is forbidden."""
    text = re.sub(r"The final evidence supports three main conclusions\..*?checkpoint/restart prediction disagreement\.", lambda _: conclusion,
                  text, count=1, flags=re.DOTALL)
    path.write_text(text, encoding="utf-8")
    print("closed stale and confirmatory claim language")


if __name__ == "__main__":
    main()
