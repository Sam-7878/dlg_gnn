#!/usr/bin/env python3
"""Apply A07 writing changes to the frozen A06 candidate without model training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import statistics
import subprocess
from pathlib import Path

from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]
A05 = ROOT / "evaluation/benchmark/v2/paper_ready_a05"
A06 = ROOT / "projects/benchmark/manuscript_base_a06"
OUT = ROOT / "projects/benchmark/paper/current"
PAPER_READY = ROOT / "evaluation/benchmark/v2/paper_ready_a07"
ADD_BIB = ROOT / "projects/benchmark/paper/references_additions.bib"
ALERT_SHA256 = "7d64d0733fd1090890aafffa5934fde09b483e546762d0a780cf86cf3c0cd131"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise ValueError(f"expected one text match, found {count}: {old[:85]!r}")
    return source.replace(old, new, 1)


def pairwise() -> str:
    cells = {(r["dataset"], r["model"], r["metric"]): float(r["mean"])
             for r in rows(A05 / "publication_evidence_a05/paper_metric_cells.csv")}
    views = rows(A05 / "statistics_s1_s5.csv")[:2]
    result = []
    for view in views:
        models = view["models"].split(";")
        datasets = view["included_datasets"].split(";")
        target = [cells[d, "DLG-Aug", "pr_auc"] for d in datasets]
        tests = []
        for model in models:
            if model == "DLG-Aug":
                continue
            other = [cells[d, model, "pr_auc"] for d in datasets]
            test = wilcoxon(target, other, alternative="two-sided")
            raw = float(test.pvalue)
            delta = [a-b for a,b in zip(target, other)]
            tests.append((model, float(test.statistic), raw, statistics.median(delta),
                          sum(x > 0 for x in delta), sum(x == 0 for x in delta), sum(x < 0 for x in delta)))
        tests.sort(key=lambda x: x[2])
        adjusted = {}
        running = 0.0
        for rank, test in enumerate(tests):
            running = max(running, min(1.0, test[2] * (len(tests)-rank)))
            adjusted[test[0]] = running
        frozen = json.loads(view["holm_adjusted_p_json"])
        for model, value in adjusted.items():
            assert abs(value - float(frozen[model])) < 1e-12, (view["view"], model)
        for model in ("DLG-Base", "DOMINANT"):
            test = next(x for x in tests if x[0] == model)
            result.append({"view":view["view"].split("_")[0], "metric":"PR-AUC",
                           "comparison":f"DLG-Aug vs {model}","n_complete":len(datasets),
                           "wilcoxon_statistic":test[1], "raw_p":test[2],
                           "holm_adjusted_p":adjusted[model],"median_signed_delta":test[3],
                           "wins":test[4],"ties":test[5],"losses":test[6],
                           "significant_at_0_05":adjusted[model] < .05})
    PAPER_READY.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (PAPER_READY / "statistics_pairwise_s1_s2.csv", OUT / "statistics_pairwise_s1_s2.csv"):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
    tex = [r"\section{Selected Holm-Adjusted Pairwise Comparisons}",
           r"\label{sec:a07_pairwise}",
           r"\begin{table}[ht]",r"\centering\small",
           r"\caption{Selected two-sided Wilcoxon signed-rank tests on seed-aggregated dataset PR-AUC. Holm adjustment is across all six S1 or four S2 comparisons of DLG-Aug with the other models; these four displayed rows are not a separate correction family. Positive median difference favors DLG-Aug. W/T/L counts datasets, not seeds.}",
           r"\label{tab:a07_holm}",r"\resizebox{\columnwidth}{!}{%",r"\begin{tabular}{llrrrrrr}",r"\toprule",
           r"View & Comparison & $N$ & $W$ & Raw $p$ & Holm $p$ & Median $\Delta$ & W/T/L " + r"\\",
           r"\midrule"]
    for item in result:
        tex.append(f"{item['view']} & Aug--{item['comparison'].split(' vs ')[1]} & {item['n_complete']} & "
                   f"{item['wilcoxon_statistic']:.1f} & {item['raw_p']:.4f} & "
                   f"{item['holm_adjusted_p']:.4f} & {item['median_signed_delta']:+.4f} & "
                   f"{item['wins']}/{item['ties']}/{item['losses']} " + r"\\")
    tex += [r"\bottomrule",r"\end{tabular}}",r"\end{table}",
            "Across the specified complete-case views, the displayed Holm-adjusted comparisons do not establish DLG-Aug superiority over DLG-Base or DOMINANT. Non-rejection is not evidence of equivalence; S1 contains only five complete datasets and differs from the S2 thirteen-dataset comparison."]
    return "\n".join(tex)


def alert_budget() -> None:
    """Rebuild the A06 alert table from 60 archived, approved A03 JSON records."""
    base = A05 / "publication_evidence_a05"
    registry = {(r["dataset"], r["model"], r["seed"]): r for r in rows(base / "approved_run_registry.csv")}
    fields = ("precision_at_1pct", "recall_at_1pct", "precision_at_5pct", "recall_at_5pct")
    outputs = []
    for dataset in ("Ethereum", "BSC", "Polygon"):
        for model in ("DOMINANT", "CoLA", "OCGNN", "DLG-Base"):
            points = []
            for seed in range(42, 47):
                record = registry[dataset, model, str(seed)]
                path = base / "legacy_metric_json" / (record["run_id"] + ".json")
                raw = path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == record["metric_json_hash"]
                result = json.loads(raw)
                assert result["run_id"] == record["run_id"]
                metrics = result.get("result", result)
                assert all(metrics.get(field) is not None for field in fields)
                points.append((record, metrics))
            output = {"dataset":dataset,"model":model,"seed_count":5,
                      "approved_run_ids":";".join(r["run_id"] for r,_ in points)}
            for field in fields:
                numbers = [float(metric[field]) for _,metric in points]
                output[field + "_mean"] = f"{statistics.mean(numbers):.9f}"
                output[field + "_population_sd"] = f"{statistics.pstdev(numbers):.9f}"
            outputs.append(output)
    path = OUT / "table_alert_budget_a07.csv"
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=outputs[0].keys())
        writer.writeheader();writer.writerows(outputs)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == ALERT_SHA256, "A06 alert-budget table drift"


RELATED = r"""\section{Related Work}
\label{sec:related}
\subsection{Unsupervised graph anomaly detection}
DOMINANT and AnomalyDAE use reconstruction, while CoLA and CONAD use contrastive objectives. OCGNN is one-class, and GADNR reconstructs neighborhoods \cite{ding2019deep,fan2020anomalydae,liu2022cola,xu2022conad,wang2021ocgnn,roy2024gadnr}. BOND and PyGOD organize implementations and evaluations, while GADBench studies supervised detection under a different label regime \cite{liu2022bond,liu2024pygod,tang2023gadbench}. Surveys summarize task and model distinctions \cite{ma2023survey,qiao2025survey}. PREM, ADA-GAD, TAM, and GADAM develop efficient representation, anomaly-denoised reconstruction, local affinity, and adaptive propagation respectively; recent evaluation work tests realistic missing-attribute and prevalence conditions \cite{pan2023prem,he2024adag,qiao2023tam,chen2024gadam,zhou2026wild}. These methods are context, not retrospectively inserted benchmark cells.

\subsection{Imbalance-aware fraud learning}
GraphSMOTE synthesizes minority representations and relations; PC-GNN uses label-aware sampling for fraud classification \cite{zhao2021graphsmote,liu2021pcgnn}. Dual-Augment GNN and graph Tran-SMOTE address camouflage, heterophily, or minority-node construction in fraud graphs \cite{li2022dualaugment,wen2024gts}. These methods rely on labels or supervised objectives and therefore are not directly comparable to the primarily unsupervised detector contract here. They nevertheless motivate reporting prevalence, PR-AUC, and alert-budget diagnostics instead of relying on ranks alone.

\subsection{Temporal graphs and scalable execution}
TGAT, TGN, EvolveGCN, DyRep, and JODIE model evolving graph or interaction histories \cite{xu2020tgat,rossi2020tgn,pareja2020evolvegcn,trivedi2019dyrep,kumar2019jodie}. LANL and the transaction sources originate in temporal systems, but this benchmark freezes constructed static graphs to maintain a common detector contract; it does not measure dynamic-model quality. GraphSAGE, FastGCN, VR-GCN, LADIES, Cluster-GCN, and GraphSAINT reduce large-graph training cost through neighborhood, layer, or subgraph sampling and variance reduction \cite{hamilton2017graphsage,chen2018fastgcn,chen2018vrgcn,zou2019ladies,chiang2019clustergcn,zeng2020graphsaint}. Their scalability techniques are valuable but alter the sampled training computation or estimator. Our narrower execution question is whether the frozen full-graph anomaly objective and score semantics can be retained while reducing memory materialization. Sampled substitutes are not silently used for unsupported cells.

\subsection{Benchmark methodology and reproducibility}
Graph model rankings can depend on splits and training procedures \cite{shchur2018pitfalls,errica2020fair}. OGB and controlled GNN benchmarks demonstrate the value of explicit datasets, splits, metrics, and reproducible code \cite{hu2020ogb,dwivedi2023benchmarking}. The present suite therefore reports support status and provenance alongside performance. Its seven primary detector configurations were frozen for implementation comparability and exact-path auditing; newer methods were not added after inspecting the results.

"""


def release_text(tex: str, tag: str | None, commit: str | None, url: str | None, doi: str | None) -> str:
    if not any((tag, commit, url, doi)):
        return tex
    if not all((tag, commit, url)):
        raise ValueError("tag, commit and release URL must be supplied together")
    resolved = subprocess.check_output(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], cwd=ROOT, text=True).strip()
    if resolved != commit:
        raise ValueError("tag does not resolve to the specified commit")
    old = "The A06 evidence release URL, immutable tag and commit will be inserted after archive publication and before preprint registration or journal submission."
    new = (f"The frozen benchmark release is \\url{{{url}}} (tag \\texttt{{{tag}}}, "
           f"commit \\texttt{{{commit}}}).")
    if doi:
        new += f" Its archive DOI is \\url{{https://doi.org/{doi}}}."
    return once(tex, old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("release-tag", "release-commit", "release-url", "archive-doi"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    alert_budget()
    table = pairwise()
    tex = (A06 / "DLG-Benchmark_A06.tex").read_text()
    start = tex.index(r"\section{Related Work}")
    end = tex.index(r"\section{Models and Exact Execution}", start)
    tex = tex[:start] + RELATED + tex[end:]
    tex = once(tex,
        "Pairwise Holm results are in the release statistics file; dataset-level performance should be interpreted conditionally",
        "Selected S1/S2 Holm-adjusted pairwise tests are reproduced in Appendix~\\ref{sec:a07_pairwise}; the full comparison family remains in the release statistics file. Dataset-level performance should be interpreted conditionally")
    tex = once(tex,
        "S3 contains six financial/blockchain graphs with mixed label provenance;",
        "S3 contains six financial/blockchain graphs with mixed label provenance (five source/native-label graphs and BitcoinOTC with controlled node-level injection on a real trust graph);")
    tex = once(tex,
        "The DLG-Aug versus DLG-Base effects vary by graph.",
        "BSC is the clearest negative native-blockchain case: DLG-Base obtains PR-AUC $0.6161\\pm0.1079$, versus $0.4179\\pm0.0182$ for DLG-Aug (difference $-0.1982$ in the rounded means). The larger Base seed spread indicates substantial sensitivity under the frozen setting; these diagnostics do not isolate its cause. Ethereum also favors Base, though by a smaller margin ($0.9803$ versus $0.9788$). GADNR on BitcoinOTC is similarly variable ($0.8125\\pm0.2296$). We keep those cells and do not tune after observing them; these observations support conditional rather than unconditional augmentation.\\par\n\nThe DLG-Aug versus DLG-Base effects vary by graph.")
    tex = once(tex,
        "The publication evidence standard requires run identity,",
        "Reviewer reproduction has three levels: integrity verification of frozen manifests and equivalence reports, regeneration of paper tables/statistics from approved records, and a separately authorized full rerun requiring third-party data and GPUs. The first two are exposed through \\texttt{projects/benchmark/reproduce.sh} and require no raw graph download.\\par\n\nThe publication evidence standard requires run identity,")
    tex = once(tex,
        "Source datasets must be obtained from their cited providers and reconstructed under provider terms.",
        "Third-party source datasets must be obtained under provider terms. The Ethereum, BSC, and Polygon inputs are locally constructed hybrid graph artifacts with frozen hashes, but the current package does not establish a public raw-data download or redistribution permission; independent full reruns of those three cases require separate data access. The evidence-only verification and paper-regeneration paths need no raw graph download.")
    # Keep the two appendix metrics tables beside their section text. Starred
    # floats otherwise postpone the second table and strand a near-empty page.
    anchor = r"\section{Additional Five-Seed Metrics}\label{sec:a06_metrics}"
    head, appendix = tex.split(anchor, 1)
    appendix = appendix.replace(r"\begin{table*}[!htbp]", r"\begin{table}[H]")
    appendix = appendix.replace(r"\end{table*}", r"\end{table}")
    tex = head + anchor + appendix
    tex = once(tex, r"\section{Additional Five-Seed Metrics}\label{sec:a06_metrics}",
               table + "\n\n" + r"\section{Additional Five-Seed Metrics}\label{sec:a06_metrics}")
    tex = release_text(tex, args.release_tag, args.release_commit, args.release_url, args.archive_doi)
    outfile = OUT / "DLG-Benchmark_A07.tex"
    outfile.write_text(tex)
    bibliography = (A06 / "references.bib").read_text() + "\n" + ADD_BIB.read_text()
    blocks = [b.strip() for b in re.split(r"(?=@[A-Za-z]+\{)", bibliography) if b.strip()]
    parsed = {}
    for block in blocks:
        match = re.match(r"@[A-Za-z]+\{([^,]+),", block)
        if not match or match.group(1) in parsed:
            raise ValueError("malformed or duplicate BibTeX key")
        parsed[match.group(1)] = block
    cited = {x.strip() for group in re.findall(r"\\cite\{([^}]*)\}", tex) for x in group.split(",")}
    if cited - parsed.keys():
        raise ValueError(f"missing citations: {sorted(cited - parsed.keys())}")
    selected = "\n\n".join(parsed[k] for k in parsed if k in cited) + "\n"
    (OUT / "references.bib").write_text(selected)
    mdpi = OUT / "mdpi"
    mdpi.mkdir(exist_ok=True)
    mdpi_a06 = A06 / "mdpi"
    header = (mdpi_a06 / "DLG-Benchmark_A06_MDPI.tex").read_text().split(r"\begin{document}")[0]
    body = tex[tex.index(r"\section{Introduction}"):tex.index(r"\end{document}")]
    body = body.replace(r"\bibliographystyle{unsrt}" + "\n", "")
    f1_match = re.search(r"\\begin\{table\}\[H\](?:(?!\\end\{table\}).)*\\label\{tab:a05_f1\}(?:(?!\\end\{table\}).)*\\end\{table\}", body, re.S)
    if not f1_match:
        raise ValueError("F1 appendix table missing")
    f1_table = f1_match.group(0)
    supplement = (r"\documentclass[11pt]{article}" + "\n" + r"\usepackage[a4paper,margin=2cm]{geometry}" + "\n"
                  + r"\usepackage{booktabs,graphicx,float}" + "\n" + r"\begin{document}" + "\n"
                  + r"\section*{A07 Supplementary Table: Validation-Selected Test F1}" + "\n"
                  + "The table uses the frozen five-seed A05 approved registry; unsupported cells are unscored.\n"
                  + f1_table + "\n" + r"\end{document}" + "\n")
    (OUT / "Supplementary_F1_A07.tex").write_text(supplement)
    body = body.replace(f1_table, "")
    body = body.replace("ROC-AUC and validation-selected F1 are reported in Appendix~\\ref{sec:a06_metrics}.",
                        "ROC-AUC is reported in Appendix~\\ref{sec:a06_metrics}; validation-selected F1 is in the supplementary table.")
    (mdpi / "DLG-Benchmark_A07_MDPI.tex").write_text(header + r"\begin{document}" + "\n\\setlength{\\headheight}{21pt}\n" + body + "\\end{document}\n")
    shutil.copytree(mdpi_a06 / "Definitions", mdpi / "Definitions", dirs_exist_ok=True)
    shutil.copy2(mdpi_a06 / "soul.sty", mdpi / "soul.sty")
    shutil.copy2(OUT / "references.bib", mdpi / "references.bib")
    for target in (OUT / "generated", mdpi / "generated"):
        target.mkdir(exist_ok=True)
        shutil.copy2(A05 / "figure_memory_measured_a05.pdf", target / "figure_memory_measured_a05.pdf")
    claims = rows(A06 / "claims_to_evidence_a06.csv")
    claims += [{"claim_id":"C13","manuscript_location":"Appendix pairwise table","claim":"S1/S2 Holm pairwise PR-AUC",
                "paper_table_or_figure":"A07 Holm table","a05_source_csv_json":"publication_evidence_a05/paper_metric_cells.csv; statistics_s1_s5.csv",
                "approved_registry":"approved_run_registry.csv","statistics_view":"S1; S2","verification":"PASS: recalculated Wilcoxon and frozen Holm cross-check"},
               {"claim_id":"C14","manuscript_location":"Results and Discussion","claim":"BSC and Ethereum Base > Aug; BSC and BitcoinOTC high variance",
                "paper_table_or_figure":"Table 2","a05_source_csv_json":"table_main_13_pr_auc.csv","approved_registry":"approved_run_registry.csv",
                "statistics_view":"","verification":"PASS: descriptive only"}]
    with (OUT / "claims_to_evidence_a07.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=claims[0].keys())
        writer.writeheader(); writer.writerows(claims)
    shutil.copy2(OUT / "claims_to_evidence_a07.csv", PAPER_READY / "claims_to_evidence_a07.csv")
    print(json.dumps({"references":len(cited),"pairwise_rows":4,"claims":len(claims),"release_final":bool(args.release_tag)},indent=2))


if __name__ == "__main__":
    main()
