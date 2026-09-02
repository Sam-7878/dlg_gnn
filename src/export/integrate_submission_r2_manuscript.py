"""Integrate generated R2 evidence into the authoritative shared LaTeX master."""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manuscript = Path(cfg["manuscript"]); manuscript_dir = manuscript.parent
    root = Path(cfg["output_root"])
    source = manuscript.read_text(encoding="utf-8")
    metrics = pd.read_csv(root / "cascade/calibrated_cascade_metrics.csv")
    fast = metrics[metrics.model == "ProductionLevel1GIN"].f1
    cascade = metrics[metrics.model.str.contains("GATv2")].f1
    fast_mean, cascade_mean = float(fast.mean()), float(cascade.mean())
    delta = cascade_mean - fast_mean
    deep_rate = float(metrics[metrics.model.str.contains("GATv2")].deep_route_rate.mean())
    claim = pd.read_csv(root / "statistics/claim_status.csv").iloc[0]
    claim_phrase = ("supported by the prespecified paired analysis" if claim.status == "SUPPORTED"
                    else "reported as descriptive because the multiplicity-controlled paired evidence is insufficient")

    generated = manuscript_dir / "generated_r2"
    if generated.exists(): shutil.rmtree(generated)
    shutil.copytree(root / "manuscript", generated)
    shutil.copyfile(manuscript_dir / "references.tex", manuscript_dir / "references.bib")

    # Preamble hygiene and IEEE journal author syntax.
    source = source.replace("\\usepackage[hidelinks]{}", "\\usepackage[hidelinks]{hyperref}")
    source = source.replace("\\usepackage{hyperref}\n", "")
    source = source.replace("\\usepackage{kotex}          % 한글 사용.\n", "")
    seen = set(); clean_lines = []
    for line in source.splitlines():
        normalized = re.sub(r"\s*%.*$", "", line).strip()
        if normalized in {"\\usepackage{url}", "\\usepackage{tikz}"}:
            if normalized in seen: continue
            seen.add(normalized)
        clean_lines.append(line)
    source = "\n".join(clean_lines) + "\n"
    source = re.sub(r"\\author\{.*?\n\}\n\n\\maketitle",
        r"\\author{SeongSu Park and Ki-Hyung Kim%\n\\thanks{The authors are with Ajou University, Suwon, Republic of Korea (e-mail: parky@ajou.ac.kr; kkim86@ajou.ac.kr).}}\n\n\\maketitle",
        source, count=1, flags=re.DOTALL)

    abstract = f"""\\begin{{abstract}}
DLG-StreamMC is a bounded stateful local-to-global graph inference architecture for resource-aware multi-chain fraud detection. It couples a production Graph Isomorphism Network (GIN) fast path, Monte Carlo dropout uncertainty, selective routing, an optional relational GATv2 stage, and log-odds fusion. We evaluate it on leakage-audited temporal splits of GoG-StreamFraud v2.0 using five fixed seeds, supervised tabular and graph controls, graph-anomaly baselines, strict source-only cross-chain transfer, and raw-event replay. After validation-only log-odds calibration and operating-point selection, the GIN fast path obtained mean F1 {fast_mean:.3f}; selective GATv2 escalation obtained {cascade_mean:.3f} (observed $\\Delta$F1={delta:+.3f}) at mean deep-route rate {deep_rate:.3f}. This improvement is {claim_phrase}. Frozen validation features required to reselect the XGBoost and LightGBM cascades were unavailable, so those cascade claims fail closed and are excluded rather than tuned on test labels. Existing measured raw-event traces show true selective execution, and a 100,000-event integrated replay completed with bounded configured state, zero event loss, and zero checkpoint/restart prediction disagreement. The evidence supports a temporally auditable selectively executed graph architecture under the evaluated protocol; it does not establish universal GNN superiority, bitwise LPP identity, asymptotically constant process RSS, or a distribution-free deployment guarantee.
\\end{{abstract}}"""
    source = re.sub(r"\\begin\{abstract\}.*?\\end\{abstract\}", lambda _: abstract, source, count=1, flags=re.DOTALL)

    results = f"""\\section{{Results}}
\\label{{sec:results}}

\\subsection{{Complete Baseline Evaluation}}
The generated tables report every evaluated chain--model cell with five-seed mean and standard deviation; undefined metrics remain explicit in the machine-readable source. Supervised graph controls, tabular controls, and graph-anomaly detectors are separated because they solve different learning problems.
\\begin{{table*}}[t]\\centering\\caption{{Supervised graph baselines on frozen temporal tests.}}\\label{{tab:supervised_complete}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_supervised_baselines.tex}}}}\\end{{table*}}
\\begin{{table*}}[t]\\centering\\caption{{Graph-anomaly baselines and DLG variants.}}\\label{{tab:gad_complete}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_graph_anomaly_baselines.tex}}}}\\end{{table*}}
\\begin{{figure}}[t]\\centering\\includegraphics[width=\\columnwidth]{{generated_r2/figures/figure_dataset_test_counts.pdf}}\\caption{{Frozen temporal test support by chain. Fraud counts are overlaid.}}\\label{{fig:data_counts}}\\end{{figure}}

\\subsection{{Validation-Calibrated Production Cascade}}
The score contract is explicit: GIN and GATv2 emit probabilities, separate Platt maps are fitted on validation log-odds, fusion occurs in calibrated log-odds space, and the router cutoff, fusion weight, and final decision threshold are selected only on validation data. ProductionLevel2GATv2 consumes the GIN embedding and score (interface Case~A), so a tabular control would have to pay for GIN representation construction before the common relational stage.
Across five held-out tests, calibrated GIN achieved mean F1 {fast_mean:.3f}, while the selectively executed GATv2 cascade achieved {cascade_mean:.3f}, an observed change of {delta:+.3f} at deep-route rate {deep_rate:.3f}. The claim is {claim_phrase}; Table~\\ref{{tab:paired_stats}} exposes every seed pair and the multiplicity-controlled result.
\\begin{{table}}[t]\\centering\\caption{{Calibrated production cascade summary.}}\\label{{tab:calibrated_cascade}}\\resizebox{{\\columnwidth}}{{!}}{{\\input{{generated_r2/tables/table_calibrated_production_cascade.tex}}}}\\end{{table}}
\\begin{{table*}}[t]\\centering\\caption{{Paired five-seed production results.}}\\label{{tab:paired_stats}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_calibrated_seed_pairs.tex}}}}\\end{{table*}}
The XGBoost and LightGBM cascade rows from the earlier submission draft are not retained as scientific results. Their frozen validation feature matrix was not preserved after the configured dataset root became unavailable; selecting a replacement operating point on test labels would violate the protocol. The acceptance gate is therefore FAIL-C: tabular fast-path controls remain in Table~\\ref{{tab:tabular_controls}}, but their cascade claims are removed.
\\begin{{table*}}[t]\\centering\\caption{{Supervised tabular fast-path controls (no revised cascade claim).}}\\label{{tab:tabular_controls}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_tabular_controls.tex}}}}\\end{{table*}}

\\subsection{{Calibration, Routing, and Accuracy--Cost Evidence}}
Calibration is evaluated on both validation and untouched test scores. Reliability curves in Fig.~\\ref{{fig:reliability}} show raw GIN, calibrated GIN, and the final selective score. Routing summaries preserve route counts, wrong-to-correct and correct-to-wrong flips, and all classification metrics; favorable uncorrected directions are not described as significant.
\\begin{{figure}}[t]\\centering\\includegraphics[width=\\columnwidth]{{generated_r2/figures/figure_reliability.pdf}}\\caption{{Pooled five-seed reliability.}}\\label{{fig:reliability}}\\end{{figure}}
\\begin{{figure}}[t]\\centering\\includegraphics[width=\\columnwidth]{{generated_r2/figures/figure_accuracy_cost_frontier.pdf}}\\caption{{Measured production accuracy versus deep-route rate.}}\\label{{fig:frontier}}\\end{{figure}}
\\begin{{table*}}[t]\\centering\\caption{{Routing summary with full accuracy and route statistics.}}\\label{{tab:routing_complete}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_routing_summary.tex}}}}\\end{{table*}}

\\subsection{{Measured Runtime and Streaming State}}
The raw-event profiler measures ingestion, state update, subgraph extraction, feature construction, GIN/MC execution, routing, optional relation construction and GATv2/fusion, queues, and trace construction. Direct exits skip all deep-stage operations. Table~\\ref{{tab:runtime_complete}} reports mean, P95, P99, throughput, route rate, RSS, and VRAM rather than only percentage reductions. Consequently, the previously reported 9.41\\% and 31.65\\% reductions are interpreted as configuration-specific measured observations; the distinct route rates explain why the empirical-risk policy may be faster than the dual policy.
\\begin{{table*}}[t]\\centering\\caption{{Measured raw-event runtime frontier.}}\\label{{tab:runtime_complete}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_runtime_measured.tex}}}}\\end{{table*}}
The integrated 100,000-event run processed all events with no OOM or event loss, bounded graph-store/cache state, and zero restart disagreement. Process RSS nevertheless had a positive fitted slope, so the evidence supports bounded configured application state, not asymptotically constant process memory.
\\begin{{figure}}[t]\\centering\\includegraphics[width=\\columnwidth]{{generated_r2/figures/figure_streaming_memory.pdf}}\\caption{{Process RSS and configured state during the 100,000-event replay.}}\\label{{fig:stream_memory}}\\end{{figure}}
\\begin{{table*}}[t]\\centering\\caption{{Integrated 100,000-event replay.}}\\label{{tab:stream_complete}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_streaming_100k.tex}}}}\\end{{table*}}

\\subsection{{LPP and Strict Cross-Chain Evaluation}}
LPP preserves all evaluated threshold decisions and stable ranking while small floating-point score differences make prediction hashes unequal. We therefore claim decision equivalence only, and report the accompanying 100,000-event resource measurements.
\\begin{{table*}}[t]\\centering\\caption{{LPP score and decision equivalence.}}\\label{{tab:lpp_score}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_lpp_score_equivalence.tex}}}}\\end{{table*}}
Strict transfer fits preprocessing, thresholds, and model state on source chains only, then evaluates the target temporal test interval. Table~\\ref{{tab:cross_complete}} includes support-sensitive results; this is descriptive evidence for the evaluated transfers rather than a universal cross-chain guarantee.
\\begin{{table*}}[t]\\centering\\caption{{Strict source-only cross-chain temporal transfer.}}\\label{{tab:cross_complete}}\\resizebox{{\\textwidth}}{{!}}{{\\input{{generated_r2/tables/table_cross_chain.tex}}}}\\end{{table*}}
"""
    source = re.sub(r"\\section\{Results\}.*?(?=\\section\{Discussion\})", lambda _: results + "\n", source, count=1, flags=re.DOTALL)
    # Remove stale exact claims outside Results and use the new measured means.
    replacements = {"0.607376": f"{fast_mean:.3f}", "0.651634": f"{cascade_mean:.3f}", "0.044258": f"{delta:.3f}",
                    "0.6074": f"{fast_mean:.3f}", "0.6516": f"{cascade_mean:.3f}", "0.0443": f"{delta:.3f}"}
    for old, new in replacements.items(): source = source.replace(old, new)
    source = source.replace("\\bibliographystyle{elsarticle-num}", "\\bibliographystyle{IEEEtran}")
    manuscript.write_text(source, encoding="utf-8")
    print(f"integrated {manuscript} with fast_f1={fast_mean:.3f}, cascade_f1={cascade_mean:.3f}, gate=FAIL-C")


if __name__ == "__main__":
    main()
