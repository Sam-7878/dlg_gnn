"""Render Defense Round D2 audit reports from frozen/generated evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

OUTPUT = Path("outputs/sci_defense_extension/d2")
DOCS = Path("docs/work_reports/209_Defense_Extension_Round_D2")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name: str, text: str) -> Path:
    path = DOCS / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def fmt(value: float) -> str:
    return f"{value:.4f}"


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    lineage = json.loads((OUTPUT / "manifests/defense_source_lineage.json").read_text(encoding="utf-8"))
    gate = json.loads((OUTPUT / "manifests/paper_readiness_gate.json").read_text(encoding="utf-8"))
    hashes = json.loads((OUTPUT / "manifests/final_extension_hashes.json").read_text(encoding="utf-8"))
    gadnr = json.loads((OUTPUT / "gadnr/gadnr_compatibility_equivalence.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((OUTPUT / "sensitivity/manifest.json").read_text(encoding="utf-8"))
    sensitivity_summary = pd.read_csv(OUTPUT / "sensitivity/theia_no_total_events_summary.csv").set_index("model")
    primary = pd.read_csv("outputs/sci_defense_extension/raw/benchmark_raw.csv")
    primary = primary[(primary.dataset == "DARPA-TC-THEIA") & primary.model.isin(["DOMINANT", "DLG-Base", "DLG-Aug"])]
    primary_summary = primary.groupby("model")[["roc_auc", "pr_auc", "f1"]].mean()
    stats = json.loads((OUTPUT / "statistics/verification.json").read_text(encoding="utf-8"))
    correlations = pd.read_csv(OUTPUT / "leakage/feature_label_correlation_diagnostic.csv")
    max_corr = correlations.loc[correlations.pearson_r.abs().groupby(correlations.dataset).idxmax()]

    report1 = f"""
# Defense Extension Round D2 — Raw Source Lineage Audit

## Audit outcome

**FAIL — the D1 graphs are deterministic synthetic simulations, not processed official DARPA TC or LANL records.**

The actual preparation programs instantiate entities such as `proc_mal_apt_0`, `C_WS_15`, and randomly generated events. Neither program accepts or opens an official raw input file. Consequently, an official raw-record manifest, parser accounting, and official ground-truth record mapping cannot be produced from the current artifacts.

## Actual artifact lineage

| Dataset label used in D1 | Actual source | Official raw files read | Final graph |
|---|---|---:|---:|
| DARPA-TC-THEIA | `prepare_darpa_theia.py` synthetic generator (`random.choice`, hand-declared APT nodes) | 0 | 1,156 nodes / 4,223 edges |
| LANL-RedTeam | `prepare_lanl_redteam.py` synthetic enterprise/red-team generator | 0 | 1,310 nodes / 10,765 edges |

Generator hashes and sizes are frozen in `outputs/sci_defense_extension/d2/lineage/source_file_manifest.csv`. Expected-but-absent official objects are listed in `expected_official_sources.csv`.

## THEIA accounting

The 1,156 nodes are generated directly as 182 Process, 818 File, 156 Socket, and 0 Other nodes. The count does not result from parsing or filtering `ta1-theia-e3-official-1r`. The graph is therefore neither a full official stream nor a defensible official subset; it is an attack-centered synthetic simulation.

The 37 positives are one generated user process, 12 generated APT processes, 18 generated malicious files, and 6 generated sockets. No official event UUID or ground-truth record ID exists for them.

## LANL accounting

The cited 17,684-computer universe was never loaded. The code directly creates 10 domain controllers, 80 servers, 1,200 workstations, and 20 gateways, totaling 1,310. Therefore, **17,684 → 1,310 is not a filtering reduction** and cannot be represented as one.

The 32 positives are selected from a hard-coded synthetic list (20 workstations, 10 servers, and 2 domain controllers); they are not derived from `redteam.txt`. Destination-computer semantics cannot be revalidated without that official file.

## Paper-use consequence

The D1 performance CSV remains frozen for auditability, but it must not be cited as DARPA-TC-THEIA or LANL empirical evidence. This directly triggers the D2 `NOT_PAPER_READY` rule for synthetic/fallback processed graphs.
"""
    p1 = write("01_defense_source_lineage_audit.md", report1)

    sensitivity_rows = []
    for model in ["DOMINANT", "DLG-Base", "DLG-Aug"]:
        base = primary_summary.loc[model]
        variant = sensitivity_summary.loc[model]
        sensitivity_rows.append(
            f"| {model} | {fmt(base.pr_auc)} | {fmt(variant.pr_auc_mean)} | {fmt(variant.pr_auc_mean-base.pr_auc)} | "
            f"{fmt(base.f1)} | {fmt(variant.validation_f1_mean)} | {fmt(variant.validation_f1_mean-base.f1)} |"
        )
    corr_text = ", ".join(f"{row.dataset}: {row.feature_name} r={row.pearson_r:.4f}" for _, row in max_corr.iterrows())
    report2 = f"""
# Defense Extension Round D2 — Feature Provenance and Sensitivity Audit

## Leakage criterion correction

The D1 rule “maximum |Pearson r| < 0.95 implies zero leakage” is retired. Correlation is diagnostic only. The observed maxima are {corr_text}.

## Code dependency result

Feature extraction and label construction are now explicit APIs:

- `DarpaTheiaGraphBuilder.extract_features()` consumes only accumulated telemetry state; `build_labels()` consumes the separately marked entity set.
- `LanlRedTeamGraphBuilder.extract_features()` consumes only telemetry state; `build_labels()` consumes the separately recorded red-team target set.
- Changing the ground-truth sets after telemetry ingestion leaves the feature matrix byte-identical in unit tests.

All 32 feature definitions, equations, and dependency flags are frozen in `outputs/sci_defense_extension/d2/leakage/feature_lineage.csv`. They are independent of label-builder inputs, but their source fields are generated synthetic fields rather than official telemetry. API independence therefore does not cure the source-lineage failure.

## THEIA-no-total-events diagnostic

The required 3 models × 5 seeds = {sensitivity['runs']}/15 runs completed after removing only feature index 13, `total_events_log1p`. Graph and labels were unchanged.

| Model | Primary PR | No-feature PR | ΔPR | Primary F1 | No-feature F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(sensitivity_rows)}

The response is model-dependent: DLG-Base declines, DOMINANT is nearly stable in PR, and DLG-Aug improves. Thus no single conclusion that THEIA performance is wholly dominated by one activity-volume feature is supported. Because the underlying artifact is synthetic, this is a D1 artifact sensitivity diagnostic only, not official THEIA evidence.

Sensitivity raw hash: `{sensitivity['raw_csv_sha256']}`.
"""
    p2 = write("02_feature_provenance_and_sensitivity_audit.md", report2)

    connected = gadnr["connected_graph"]
    isolated = gadnr["isolated_node"]
    report3 = f"""
# Defense Extension Round D2 — GADNR Compatibility Equivalence

## Result

**PASS — semantics-preserving on a connected graph, with isolated-node correctness restored.**

Installed PyGOD 1.1.0 cannot run literally unmodified with the current PyG because it forwards the obsolete `tot_nodes` keyword into `MessagePassing`. The reference path therefore removes only that rejected keyword while retaining installed upstream preprocessing, forward, loss, scoring, and optimizer behavior.

## Patch classification

| Patch | Classification | Evidence |
|---|---|---|
| Unused `mp.Pool(4)` bypass | Semantics-preserving compatibility | Upstream AST: 1 assignment, 0 computational reads |
| Explicit `num_nodes=N` and exact `bincount(minlength=N)` | Equivalent on connected graphs; correctness fix with isolates | Same preprocessed edges/degrees on connected graph |
| Removal of obsolete `tot_nodes` forwarding | Required current-PyG compatibility | Literal upstream fails before training |

## Numeric equivalence

| Quantity | Max absolute error |
|---|---:|
| Normalized features | {connected['feature_max_abs_error']:.3e} |
| Degree statistics | {connected['degree_max_abs_error']:.3e} |
| Training loss per node | {connected['training_loss_per_node_max_abs_error']:.3e} |
| Node anomaly score | {connected['score_max_abs_error']:.3e} |

ROC-AUC is {connected['reference_roc_auc']:.6f} for both paths; PR-AUC is {connected['reference_pr_auc']:.6f} for both paths.

With one isolated node, upstream preprocessing returns {isolated['upstream_degree_outputs']} degree targets for {isolated['nodes']} nodes. The corrected path returns {isolated['corrected_score_outputs']} finite scores. The accepted manuscript term is **“GADNR with a semantics-preserving compatibility correction”**, not “unmodified GADNR.”
"""
    p3 = write("03_gadnr_compatibility_equivalence.md", report3)

    report4 = f"""
# Defense Extension Round D2 — Statistics Verification and Claim Corrigendum

## Independent arithmetic verification

- 10-dataset and 12-dataset seed-first ranks match the D1 tables: `{stats['rank_recomputation_matches_reported']}`.
- DLG-Aug pairwise Wilcoxon statistics and Holm-adjusted p-values match: `{stats['pairwise_recomputation_matches_reported']}`.
- The 12-dataset common scalable subset remains `CoLA`, `DOMINANT`, `OCGNN`, `DLG-Base`, `DLG-Aug`; GADNR is not included.
- No inferential superiority claim is permitted from the two defense-labeled datasets alone.

This verifies the arithmetic only. Since the two added graphs are synthetic, the resulting 12-dataset view is not a valid external-validation analysis for the paper.

## Required claim corrections

1. THEIA-labeled artifact: DLG-Base has the highest ROC-AUC (0.9897), while DOMINANT and CONAD have the highest PR-AUC (0.9357) and validation-selected F1 (0.9379). “Highest overall” is withdrawn.
2. LANL-labeled artifact: CONAD has the highest ROC-AUC (0.9281); AnomalyDAE has the highest PR-AUC (0.4415) and F1 (0.5504). DLG-Base PR-AUC is 0.3081.
3. The negative augmentation deltas are only consistent with a structural-signal hypothesis; they do not establish that mechanism causally.
4. DRQ2 is limited to an observed dataset-dependent pattern in the synthetic D1 artifacts.
5. DRQ3 must state that detector architecture/objectives were unchanged, while GADNR received documented compatibility corrections.
6. Replace “100% reproducibility” with “all planned five-seed runs completed under the frozen environment.”
7. Because source provenance fails, none of these performance statements may be presented as empirical DARPA/LANL findings in the SCI manuscript.
"""
    p4 = write("04_statistics_verification_and_claim_corrigendum.md", report4)

    report5 = f"""
# Defense Extension Round D2 — Final Paper-Readiness Audit

## 1. Final decision

# NOT_PAPER_READY

Round D2 successfully detected a fail-closed provenance violation: both D1 “defense datasets” are synthetic simulations produced by local random generators. They are not transformations of official DARPA TC or LANL raw records. The D1 80-run performance evidence and the derived 12-dataset view must be excluded from the SCI manuscript.

## 2. Raw source lineage

Official files traceable: **No**. Actual generator source and hashes are recorded, but there are zero official raw inputs.

## 3. Dataset reduction/accounting

- THEIA 1,156 nodes are declared directly by the generator; no raw→parsed→resolved filtering chain exists.
- LANL 1,310 nodes are declared directly; the cited 17,684-node universe was never loaded. The claimed reduction cannot be explained because it never occurred.

## 4. Ground-truth mapping

The 37 and 32 positives map reproducibly to hard-coded synthetic IDs, not to official attack/red-team record IDs. Official mapping criterion: **failed**.

## 5. Feature leakage audit

Feature-builder APIs are ground-truth independent, and correlation is diagnostic only. However, the feature source telemetry is synthetic and deliberately co-generated with attack scenarios, so source validity remains failed.

## 6. THEIA high-correlation sensitivity

All 15 planned diagnostic runs completed without mutating D1. Results are model-dependent and are limited to the synthetic artifact.

## 7. GADNR compatibility equivalence

Accepted. Connected-graph loss and score maximum absolute errors are 0; isolated-node cardinality is corrected. An additional obsolete-`tot_nodes` compatibility difference was identified and documented.

## 8. Benchmark statement corrections

The corrected result wording and causal boundaries are frozen in `04_statistics_verification_and_claim_corrigendum.md`. D1 reports are historical working artifacts and are superseded by this D2 audit.

## 9. 12-dataset statistics verification

Rank and Wilcoxon/Holm recomputations match. The arithmetic is reproducible, but its defense-extension interpretation is invalid because the additional graphs lack official lineage.

## 10. Final claim boundaries

Supported claims:

- Round 5 primary 10-dataset results remain frozen and unchanged.
- The benchmark adapter can execute the two synthetic graph shapes.
- GADNR compatibility corrections preserve connected-graph detector semantics.
- D1/D2 artifact-level sensitivity and statistical arithmetic are reproducible.

Not supported:

- Any claim of performance on official DARPA-TC-THEIA or LANL-RedTeam.
- Defense-domain external validation or generalization.
- A causal explanation for augmentation behavior.
- Inclusion of the current defense rows in paper inferential statistics.

## Gate summary

| Criterion | Result |
|---|---|
| Round 5 frozen hashes unchanged | PASS |
| Defense D1 raw unchanged | PASS |
| Official raw hashes/records available | **FAIL** |
| Raw→processed accounting complete | **FAIL** |
| Official positive mapping reproducible | **FAIL** |
| Feature API independent of labels | PASS |
| THEIA sensitivity complete | PASS |
| GADNR equivalence complete | PASS |
| 12-dataset arithmetic independently verified | PASS |

To restore paper readiness, acquire the official raw files and ground truth, implement real parsers, rebuild both graphs with complete record accounting, and then rerun only the defense extension. The frozen Round 5 355 runs remain reusable.
"""
    p5 = write("05_defense_extension_paper_readiness_audit.md", report5)

    registry = {
        "decision": gate["decision"],
        "reports": [{"path": str(path), "sha256": sha256(path)} for path in (p1, p2, p3, p4, p5)],
        "frozen_hashes": hashes,
    }
    (OUTPUT / "manifests/d2_report_registry.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": gate["decision"], "reports": len(registry["reports"])}, indent=2))


if __name__ == "__main__":
    main()
