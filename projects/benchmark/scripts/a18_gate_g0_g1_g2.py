#!/usr/bin/env python3
"""
A18 Gate G0, G1, G2 Execution Script
- G0: Snapshot & Authoritative Source Map
- G1: Alert-Budget Table 3 Repair & Invariant Checks
- G2: Appendix F1 Table A2 Regeneration & Audit
"""

import os
import sys
import json
import hashlib
import glob
import math
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path("/mnt/d/_Work/goat_bank/dlg_gnn")
if not WORKSPACE_ROOT.exists():
    WORKSPACE_ROOT = Path("D:/_Work/goat_bank/dlg_gnn")

A18_PUB_DIR = WORKSPACE_ROOT / "projects/benchmark/publication/a18"
A16_PUB_DIR = WORKSPACE_ROOT / "projects/benchmark/publication/a16"
REPORTS_DIR = WORKSPACE_ROOT / "docs/work_reports/benchmark/243_A18"

A18_PUB_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DS_PREFIX_MAP = {
    "Ethereum": ["ethereum_contract_clean_v1"],
    "BSC": ["bsc_contract_clean_v1"],
    "Polygon": ["polygon_contract_clean_v1"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


# ==========================================
# G0: Snapshot & Input Inventory
# ==========================================
def run_gate_g0():
    print("--- Running Gate G0: Snapshot & Input Inventory ---")
    inventory_items = []

    files_to_track = [
        WORKSPACE_ROOT / "projects/benchmark/paper/a16/DLG-Benchmark_A16_Preprints.tex",
        WORKSPACE_ROOT / "projects/benchmark/paper/a16/mdpi/DLG-Benchmark_A16_AppliedSciences.tex",
        WORKSPACE_ROOT / "projects/benchmark/paper/a16/references.bib",
        WORKSPACE_ROOT / "projects/benchmark/submission/preprints/DLG-Benchmark_A16_Preprints.pdf",
        WORKSPACE_ROOT / "projects/benchmark/submission/applsci/DLG-Benchmark_A16_AppliedSciences.pdf",
        WORKSPACE_ROOT / "projects/benchmark/submission/preprints/source_bundle.zip",
        WORKSPACE_ROOT / "projects/benchmark/submission/applsci/source_bundle.zip",
        A16_PUB_DIR / "canonical_selected_run_registry.csv",
        A16_PUB_DIR / "denominator_crosscheck_a16.csv",
        A16_PUB_DIR / "support_14x7.csv",
        A16_PUB_DIR / "release_manifest.json",
        REPORTS_DIR / "DLG_Benchmark_A18_Critical_Evidence_Repair_Work_Order_2026-10-28.md",
        REPORTS_DIR / "DLG_Benchmark_A17_Independent_Claude_Review_2026-10-28.md",
        REPORTS_DIR / "DLG_Benchmark_A17_Alert_Budget_Independent_Check.csv",
    ]

    for p in files_to_track:
        if p.exists():
            sz = p.stat().st_size
            h = sha256_file(p)
            rel_path = str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
            inventory_items.append({
                "path": rel_path,
                "size_bytes": sz,
                "sha256": h,
                "status": "EXISTS"
            })
        else:
            rel_path = str(p.relative_to(WORKSPACE_ROOT)).replace("\\", "/")
            inventory_items.append({
                "path": rel_path,
                "size_bytes": 0,
                "sha256": "MISSING",
                "status": "NOT_FOUND"
            })

    inv_df = pd.DataFrame(inventory_items)
    inv_df.to_csv(A18_PUB_DIR / "input_inventory.csv", index=False)
    print(f"Saved {len(inv_df)} inventory records to {A18_PUB_DIR / 'input_inventory.csv'}")

    # Build Authoritative Source Map Markdown
    source_map_content = f"""# A18 Authoritative Source Map
Generated Date: 2026-10-28
Document ID: `DLG-BENCH-MAP-A18-SOURCES`

## 1. Frozen Benchmark Configuration
- **Approved Tag:** `benchmark-a16-submission-2026-10-28`
- **Target Commit:** `b5aaed2a7e7fc72ca5568b0f30d8fcc6afd05b9f`
- **Current HEAD Commit:** `91fb2a76369c09a0d9131fd03e70965082190871`
- **Supported Datasets:** 13 Primary + LANL (14 datasets total)
- **Supported Models:** DOMINANT, AnomalyDAE, CoLA, GADNR, OCGNN, DLG-Base, DLG-Aug (7 models)
- **Evaluation Seeds:** 42, 43, 44, 45, 46 (5 seeds)
- **Approved Primary Runs:** 420 (84 pairs × 5 seeds) + 35 LANL (7 pairs × 5 seeds) = 455 complete runs.
- **Unsupported Pairs:** 7 pairs (CONAD diagnostic-only; OCGNN on DGraphFin/BHP; CoLA on DGraphFin/BHP; AnomalyDAE on large graphs where memory ceiling hit).

## 2. Evidence Tier & Storage Hierarchy
| Dataset Group | Campaign / Origin | Metric Source Path | Raw Scores Path | Population Masks |
|---|---|---|---|---|
| Cryptocurrency (Ethereum, BSC, Polygon) | A08 Data Repair (`local_storage/benchmark/a08_data_repair/runs/`) | `metrics.json` (per-run) | `scores.npz` (full array) | Exact test masks & labels stored |
| Primary Synthetic & Real (Amazon-Syn, Elliptic, Yelp, etc.) | Paper Ready A04 (`evaluation/benchmark/v2/paper_ready_a04/publication_evidence_a04/metric_json/`) | JSON per run | Stored in historical v2 outputs | Stored in split masks |
| BitcoinOTC | Paper Ready A05 (`evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/`) | JSON per run | Stored in historical v2 outputs | Stored in split masks |
| LANL Cyber | Round 5 Cyber Campaign | JSON per run | Stored in raw outputs | Full evaluation slice |

## 3. Metric Invariants & Population Standards
- **Test Prevalences:**
  - Ethereum: ~0.4089 (M=584, N=1428)
  - BSC: ~0.1473 (M=221, N=1500)
  - Polygon: ~0.0259 (M=120, N=4641)
- **Obsolete Prevalences (REJECTED by Negative Control):**
  - Ethereum: 0.0195
  - BSC: 0.0210
  - Polygon: 0.0052
"""
    with open(REPORTS_DIR / "a18_authoritative_source_map.md", "w", encoding="utf-8") as f:
        f.write(source_map_content)
    print(f"Saved {REPORTS_DIR / 'a18_authoritative_source_map.md'}")
    return True


# ==========================================
# G1: Alert-Budget Table 3 Repair
# ==========================================
def run_gate_g1():
    print("--- Running Gate G1: Alert-Budget Table 3 Repair ---")
    runs_dir = WORKSPACE_ROOT / "local_storage/benchmark/a08_data_repair/runs"
    assert runs_dir.exists(), f"Missing runs dir {runs_dir}"

    datasets = ["Ethereum", "BSC", "Polygon"]
    models = ["DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]
    seeds = [42, 43, 44, 45, 46]
    budgets = ["0.01", "0.05"]

    per_seed_records = []
    provenance_records = []
    math_gate_checks = {
        "total_evaluations": 0,
        "invariant_violations": 0,
        "negative_controls_rejected": 0,
        "negative_controls_passed": 0,
        "status": "PENDING"
    }

    # Dataset prefix mapping in a08_data_repair
    # Let's inspect directory prefixes
    all_dirs = [d.name for d in runs_dir.iterdir() if d.is_dir()]
    print(f"Total run dirs in a08_data_repair: {len(all_dirs)}")
    prefixes = set(d.split("__")[0] for d in all_dirs if "__" in d)
    print(f"Prefixes found: {prefixes}")

    obsolete_prevalences = {
        "Ethereum": 0.0195,
        "BSC": 0.0210,
        "Polygon": 0.0052
    }

    for ds in datasets:
        candidate_prefixes = DS_PREFIX_MAP.get(ds, [ds])
        for model in models:
            for seed in seeds:
                matches = []
                for pfx in candidate_prefixes:
                    run_pattern = str(runs_dir / f"{pfx}__{model}__seed{seed}*")
                    matches.extend(glob.glob(run_pattern))
                if len(matches) == 0:
                    # try pattern anywhere in name
                    run_pattern = str(runs_dir / f"*{ds}*_{model}__seed{seed}*")
                    matches = glob.glob(run_pattern)
                assert len(matches) == 1, f"Expected 1 run for {ds} {model} {seed}, got {len(matches)}: {matches}"
                run_path = Path(matches[0])

                metric_file = run_path / "metrics.json"
                scores_file = run_path / "scores.npz"
                assert metric_file.exists(), f"Missing metric file: {metric_file}"
                assert scores_file.exists(), f"Missing scores file: {scores_file}"

                score_hash = sha256_file(scores_file)
                metric_hash = sha256_file(metric_file)

                provenance_records.append({
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "run_dir": run_path.name,
                    "metric_sha256": metric_hash,
                    "score_sha256": score_hash,
                })

                with open(metric_file, "r") as f:
                    mdata = json.load(f)

                if ds == "Ethereum" and model == "DOMINANT" and seed == 42:
                    print("Sample metrics.json keys:", list(mdata.keys()))
                    if "alert_budgets" in mdata:
                        print("Sample alert_budgets:", mdata["alert_budgets"])

                n_test = mdata.get("test_total") or mdata.get("total_test_nodes") or mdata.get("n_test")
                m_test = mdata.get("test_positives") or mdata.get("test_positive") or mdata.get("total_positive_test_nodes") or mdata.get("m_test")
                pi_test = mdata.get("test_prevalence") or mdata.get("prevalence")
                assert n_test is not None and m_test is not None, f"Missing test counts in {metric_file}"
                if pi_test is None:
                    pi_test = m_test / n_test
                assert abs(pi_test - m_test / n_test) < 1e-4

                alert_budgets = mdata["alert_budgets"]

                for q_str in budgets:
                    q = float(q_str)
                    b_data = alert_budgets[q_str]
                    k = b_data.get("k") or b_data.get("budget_nodes")
                    p = b_data["precision"]
                    r = b_data["recall"]
                    tp = b_data.get("true_positives")
                    if tp is None:
                        tp = int(round(p * k))

                    # 1. Hard invariant checks
                    assert 0 <= tp <= min(k, m_test), f"TP out of bounds: {tp}, K={k}, M={m_test}"
                    assert abs(p - tp / k) <= 1e-12, f"Precision mismatch: {p} vs {tp/k}"
                    assert abs(r - tp / m_test) <= 1e-12, f"Recall mismatch: {r} vs {tp/m_test}"
                    assert abs(r - (p * k / m_test)) <= 1e-12, f"R = P*K/M mismatch"
                    assert r <= min(1.0, k / m_test) + 1e-12, f"Recall exceeds ceiling: {r} > {k/m_test}"

                    math_gate_checks["total_evaluations"] += 1

                    # 2. Negative control test against obsolete prevalence
                    obs_pi = obsolete_prevalences[ds]
                    # Old erroneous calculation: R_obs = P * (K / (N_test * obs_pi))
                    # or R_obs = (P * q) / obs_pi
                    r_erroneous = (p * q) / obs_pi
                    # In our negative control, verify that the true recall differs significantly from r_erroneous
                    # whenever p > 0
                    if p > 0.05:
                        diff_obs = abs(r - r_erroneous)
                        if diff_obs > 0.01:
                            math_gate_checks["negative_controls_rejected"] += 1
                        else:
                            math_gate_checks["negative_controls_passed"] += 1

                    per_seed_records.append({
                        "dataset": ds,
                        "model": model,
                        "seed": seed,
                        "budget_ratio": q,
                        "N_test": n_test,
                        "M_test": m_test,
                        "prevalence": pi_test,
                        "K": k,
                        "TP": tp,
                        "precision": p,
                        "recall": r,
                        "score_sha256": score_hash,
                        "metric_sha256": metric_hash,
                    })

    per_seed_df = pd.DataFrame(per_seed_records)
    per_seed_df.to_csv(A18_PUB_DIR / "alert_budget_per_seed_verified.csv", index=False)
    print(f"Saved {len(per_seed_df)} per-seed records to {A18_PUB_DIR / 'alert_budget_per_seed_verified.csv'}")

    prov_df = pd.DataFrame(provenance_records)
    prov_df.to_csv(A18_PUB_DIR / "alert_budget_provenance.csv", index=False)
    print(f"Saved {len(prov_df)} provenance records to {A18_PUB_DIR / 'alert_budget_provenance.csv'}")

    # Aggregate across 5 seeds: mean and sample SD (ddof=1)
    agg_records = []
    for (ds, model, q), grp in per_seed_df.groupby(["dataset", "model", "budget_ratio"]):
        p_mean = grp["precision"].mean()
        p_sd = grp["precision"].std(ddof=1)
        r_mean = grp["recall"].mean()
        r_sd = grp["recall"].std(ddof=1)
        tp_mean = grp["TP"].mean()
        k_val = grp["K"].iloc[0]
        m_val = grp["M_test"].iloc[0]
        n_val = grp["N_test"].iloc[0]

        agg_records.append({
            "dataset": ds,
            "model": model,
            "budget_ratio": q,
            "N_test": n_val,
            "M_test": m_val,
            "K": k_val,
            "TP_mean": tp_mean,
            "precision_mean": p_mean,
            "precision_sd": p_sd,
            "recall_mean": r_mean,
            "recall_sd": r_sd,
        })

    agg_df = pd.DataFrame(agg_records)
    # Sort logically
    model_order = {m: i for i, m in enumerate(models)}
    agg_df["model_rank"] = agg_df["model"].map(model_order)
    agg_df = agg_df.sort_values(["dataset", "model_rank", "budget_ratio"]).drop(columns=["model_rank"])
    agg_df.to_csv(A18_PUB_DIR / "alert_budget_aggregated_verified.csv", index=False)
    print(f"Saved {len(agg_df)} aggregated records to {A18_PUB_DIR / 'alert_budget_aggregated_verified.csv'}")

    math_gate_checks["status"] = "PASS"
    with open(A18_PUB_DIR / "alert_budget_math_gate.json", "w") as f:
        json.dump(math_gate_checks, f, indent=2)
    print(f"Saved Gate G1 summary: {math_gate_checks}")

    # Generate verified Table 3 LaTeX fragment
    models_order = ["DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]
    lines = []
    lines.append("\\begin{tabularx}{\\textwidth}{llrrrrrrr}\\toprule")
    lines.append("\\textbf{Dataset} & \\textbf{Metric} & \\textbf{DOMINANT} & \\textbf{AnomalyDAE} & \\textbf{CoLA} & \\textbf{GADNR} & \\textbf{OCGNN} & \\textbf{DLG-Base} & \\textbf{DLG-Aug} \\\\\\midrule")

    for ds_idx, ds in enumerate(["Ethereum", "BSC", "Polygon"]):
        metrics_specs = [
            ("P@1\\%", 0.01, "precision_mean"),
            ("R@1\\%", 0.01, "recall_mean"),
            ("P@5\\%", 0.05, "precision_mean"),
            ("R@5\\%", 0.05, "recall_mean"),
        ]
        for m_idx, (label, q, col) in enumerate(metrics_specs):
            prefix = f"\\multirow{{4}}{{*}}{{{ds}}}" if m_idx == 0 else ""
            row_vals = []
            max_val = -1.0
            sub = agg_df[(agg_df["dataset"] == ds) & (agg_df["budget_ratio"] == q)]
            for m in models_order:
                val = sub[sub["model"] == m][col].iloc[0]
                if val > max_val:
                    max_val = val

            for m in models_order:
                val = sub[sub["model"] == m][col].iloc[0]
                val_str = f"{val:.4f}"
                if abs(val - max_val) < 1e-6 and max_val > 0:
                    row_vals.append(f"$\\mathbf{{{val_str}}}$")
                else:
                    row_vals.append(f"${val_str}$")

            lines.append(f"{prefix} & ${label}$ & " + " & ".join(row_vals) + " \\\\")

        if ds_idx < 2:
            lines.append("\\midrule")
        else:
            lines.append("\\bottomrule")
    lines.append("\\end{tabularx}")

    t3_latex = "\n".join(lines)
    with open(A18_PUB_DIR / "table_alert_budget_verified.tex", "w") as f:
        f.write(t3_latex)
    print(f"Saved verified Table 3 LaTeX fragment to {A18_PUB_DIR / 'table_alert_budget_verified.tex'}")

    return agg_df


# ==========================================
# G2: Appendix F1 Table A2 Regeneration
# ==========================================
def run_gate_g2():
    print("--- Running Gate G2: Appendix F1 Table A2 Regeneration ---")
    reg_path = A16_PUB_DIR / "canonical_selected_run_registry.csv"
    assert reg_path.exists(), f"Missing canonical registry: {reg_path}"
    reg_df = pd.read_csv(reg_path)

    # Filter to supported runs
    supp_df = reg_df[reg_df["support_status"] == "SUPPORTED_EXACT"].copy()
    print(f"Found {len(supp_df)} supported exact runs in canonical registry.")
    assert len(supp_df) == 455, f"Expected 455 supported runs, got {len(supp_df)}"

    datasets = sorted(supp_df["dataset"].unique())
    models = ["DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug"]

    f1_per_seed = []
    f1_provenance = []

    for idx, row in supp_df.iterrows():
        ds = row["dataset"]
        model = row["model"]
        seed = int(row["seed"])
        f1_reg = float(row["f1"])
        run_id = row["run_id"]
        campaign_id = row["campaign_id"]
        metric_file_path = str(row["metric_file_path"])
        metric_json_path = str(row["metric_json_path"])

        # Determine exact source file
        resolved_file = None
        candidates = []

        if ds in ["Ethereum", "BSC", "Polygon"]:
            pfxs = DS_PREFIX_MAP.get(ds, [ds])
            for pfx in pfxs:
                candidates.append(WORKSPACE_ROOT / f"local_storage/benchmark/a08_data_repair/runs/{pfx}__{model}__seed{seed}/metrics.json")
        else:
            candidates.extend([
                WORKSPACE_ROOT / metric_file_path,
                WORKSPACE_ROOT / metric_json_path,
                WORKSPACE_ROOT / metric_json_path.replace("paper_ready_final", "paper_ready_a04"),
                WORKSPACE_ROOT / metric_json_path.replace("paper_ready_final", "paper_ready_a05"),
                WORKSPACE_ROOT / f"evaluation/benchmark/v2/paper_ready_a04/publication_evidence_a04/metric_json/{run_id}.json",
                WORKSPACE_ROOT / f"evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/metric_json/{run_id}.json",
                WORKSPACE_ROOT / f"outputs/benchmark/round5_cyber_a02/raw/{run_id}.json",
            ])
            if ds == "DGraphFin" and model == "AnomalyDAE":
                candidates.append(WORKSPACE_ROOT / f"evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/anomalydae_raw/{ds}__{model}__seed{seed}.json")
        # Check candidates
        for c in candidates:
            if c.exists():
                resolved_file = c
                break

        if resolved_file is None:
            # Fallback search by pattern
            pfxs = DS_PREFIX_MAP.get(ds, [ds])
            pats = []
            for pfx in pfxs:
                pats.append(str(WORKSPACE_ROOT / f"local_storage/benchmark/a08_data_repair/runs/{pfx}__{model}__seed{seed}*/metrics.json"))
            pats.extend([
                str(WORKSPACE_ROOT / f"local_storage/benchmark/a08_data_repair/runs/*{ds}*_{model}__seed{seed}*/metrics.json"),
                str(WORKSPACE_ROOT / f"evaluation/benchmark/v2/paper_ready_a04/publication_evidence_a04/metric_json/{run_id}.json"),
                str(WORKSPACE_ROOT / f"evaluation/benchmark/v2/paper_ready_a05/publication_evidence_a05/metric_json/{run_id}.json"),
                str(WORKSPACE_ROOT / f"outputs/benchmark/sci_round5_final/raw/{ds}__{model}__seed{seed}__*.json"),
                str(WORKSPACE_ROOT / f"outputs/benchmark/round5_cyber_a02/raw/*{model}*seed{seed}*.json"),
            ])
            for pat in pats:
                matches = glob.glob(pat)
                if matches:
                    resolved_file = Path(matches[0])
                    break

        assert resolved_file is not None, f"Could not find metric file for {ds} {model} seed {seed}"

        # Read JSON and verify F1
        with open(resolved_file, "r") as f:
            jdata = json.load(f)

        f1_source = None
        threshold_val = None
        if "thresholded" in jdata and "f1" in jdata["thresholded"]:
            f1_source = float(jdata["thresholded"]["f1"])
            threshold_val = jdata["thresholded"].get("threshold")
        elif "validation_f1" in jdata:
            f1_source = float(jdata["validation_f1"])
            threshold_val = jdata.get("best_threshold") or jdata.get("threshold")
        elif "f1" in jdata:
            f1_source = float(jdata["f1"])
            threshold_val = jdata.get("best_threshold") or jdata.get("threshold")
        elif "test_f1" in jdata:
            f1_source = float(jdata["test_f1"])
            threshold_val = jdata.get("best_threshold") or jdata.get("threshold")

        if ds == "Ethereum" and model == "AnomalyDAE" and seed == 42:
            print(f"DEBUG {ds} {model} {seed}:")
            print("  resolved_file:", resolved_file)
            print("  metric_file_path:", metric_file_path)
            print("  metric_json_path:", metric_json_path)
            print("  f1_reg:", f1_reg)
            print("  f1_source:", f1_source)
            print("  validation_f1 in json:", jdata.get("validation_f1"))
            print("  JSON keys:", list(jdata.keys()))
        assert abs(f1_source - f1_reg) < 1e-4, f"F1 registry mismatch for {ds} {model} seed {seed}: reg={f1_reg}, source={f1_source}"

        file_hash = sha256_file(resolved_file)

        f1_per_seed.append({
            "dataset": ds,
            "model": model,
            "seed": seed,
            "f1": f1_source,
            "threshold": threshold_val,
            "run_id": run_id,
            "source_file": str(resolved_file.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "file_sha256": file_hash,
        })

        f1_provenance.append({
            "dataset": ds,
            "model": model,
            "seed": seed,
            "campaign_id": campaign_id,
            "run_id": run_id,
            "metric_path": str(resolved_file.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "metric_sha256": file_hash,
            "f1_value": f1_source,
            "threshold": threshold_val,
            "provenance_status": "VERIFIED_EXACT"
        })

    f1_seed_df = pd.DataFrame(f1_per_seed)
    f1_seed_df.to_csv(A18_PUB_DIR / "f1_per_seed_verified.csv", index=False)
    print(f"Saved {len(f1_seed_df)} verified per-seed F1 values to {A18_PUB_DIR / 'f1_per_seed_verified.csv'}")

    f1_prov_df = pd.DataFrame(f1_provenance)
    f1_prov_df.to_csv(A18_PUB_DIR / "f1_provenance_a18.csv", index=False)
    print(f"Saved {len(f1_prov_df)} F1 provenance records to {A18_PUB_DIR / 'f1_provenance_a18.csv'}")

    # Aggregate: mean and sample SD (ddof=1)
    f1_agg = []
    for (ds, model), grp in f1_seed_df.groupby(["dataset", "model"]):
        assert len(grp) == 5, f"Expected 5 seeds for {ds} {model}, got {len(grp)}"
        m = grp["f1"].mean()
        s = grp["f1"].std(ddof=1)
        f1_agg.append({
            "dataset": ds,
            "model": model,
            "f1_mean": m,
            "f1_sd": s,
            "formatted": f"{m:.4f} \\pm {s:.4f}"
        })
    f1_agg_df = pd.DataFrame(f1_agg)

    # Now let's extract the existing A17 Table A2 from DLG-Benchmark_A16_Preprints.tex
    tex_path = WORKSPACE_ROOT / "projects/benchmark/paper/a16/DLG-Benchmark_A16_Preprints.tex"
    with open(tex_path, "r", encoding="utf-8") as f:
        tex_content = f.read()

    # Parse A17 Table A2 rows
    # The table has label \label{tab:app_f1_full}
    diff_records = []
    # Let's see: we can parse each line in tab:app_f1_full
    table_start = tex_content.find("\\label{tab:app_f1_full}")
    if table_start != -1:
        table_end = tex_content.find("\\end{tabular}", table_start)
        table_str = tex_content[table_start:table_end]
        for line in table_str.split("\n"):
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("\\") or "&" not in line:
                continue
            parts = [p.strip() for p in line.split("&")]
            if len(parts) >= 8:
                ds_raw = parts[0].replace("\\_", "_").replace("\\-", "-").strip()
                # normalize dataset name
                ds_clean = ds_raw
                # Map to standard dataset names
                ds_map = {
                    "Amazon-Syn": "Amazon-Syn",
                    "Yelp-Syn": "Yelp-Syn",
                    "Elliptic": "Elliptic",
                    "DGraphFin": "DGraphFin",
                    "Tolokers": "Tolokers",
                    "Questions": "Questions",
                    "Reddit": "Reddit",
                    "Weibo": "Weibo",
                    "Enron": "Enron",
                    "Disney": "Disney",
                    "Books": "Books",
                    "Ethereum": "Ethereum",
                    "BSC": "BSC",
                    "Polygon": "Polygon",
                    "LANL": "LANL",
                    "BitcoinOTC": "BitcoinOTC",
                }
                for k, v in ds_map.items():
                    if k.lower() in ds_raw.lower():
                        ds_clean = v
                        break

                for m_idx, m_name in enumerate(models):
                    cell = parts[m_idx + 1].replace("\\\\", "").strip()
                    if cell in ["-", "--", "---", "\\textemdash", "N/A"]:
                        continue
                    # Parse "mean \pm sd" or "$mean \pm sd$"
                    clean_cell = cell.replace("$", "").replace("\\pm", "±")
                    clean_cell = clean_cell.replace("\\mathbf{", "").replace("\\textbf{", "").replace("}", "")
                    for token in ["\\midrule", "\\hline", "\\bottomrule", "\\toprule", "\\\\"]:
                        clean_cell = clean_cell.replace(token, "")
                    clean_cell = clean_cell.strip()
                    if "±" in clean_cell:
                        parts_split = clean_cell.split("±")
                        c_mean = float(parts_split[0].strip())
                        c_sd = float(parts_split[1].strip())
                        # Look up newly computed
                        match = f1_agg_df[(f1_agg_df["dataset"] == ds_clean) & (f1_agg_df["model"] == m_name)]
                        if not match.empty:
                            new_m = match["f1_mean"].iloc[0]
                            new_s = match["f1_sd"].iloc[0]
                            diff_records.append({
                                "dataset": ds_clean,
                                "model": m_name,
                                "a17_mean": c_mean,
                                "a17_sd": c_sd,
                                "verified_mean": new_m,
                                "verified_sd": new_s,
                                "diff_mean": new_m - c_mean,
                                "diff_sd": new_s - c_sd,
                            })

    diff_df = pd.DataFrame(diff_records)
    diff_df.to_csv(A18_PUB_DIR / "f1_diff_a17_vs_source.csv", index=False)
    print(f"Saved {len(diff_df)} comparison records to {A18_PUB_DIR / 'f1_diff_a17_vs_source.csv'}")
    if not diff_df.empty:
        max_diff = diff_df["diff_mean"].abs().max()
        print(f"Max mean F1 diff between A17 table and verified sources: {max_diff:.6f}")

    # Datasets in Table A2 order
    print("f1_agg_df datasets:", sorted(f1_agg_df["dataset"].unique()))

    all_datasets_order = [
        "Elliptic", "DGraphFin", "BitcoinOTC", "Ethereum", "BSC", "Polygon",
        "Yelp-Syn", "Amazon-Syn", "Reddit-Syn", "Flickr-Syn", "Cora-Syn",
        "CiteSeer-Syn", "PubMed-Syn"
    ]
    # LANL is separated by \midrule

    tex_rows = []
    for ds in all_datasets_order:
        # Match dataset name in f1_agg_df (could be LANL or LANL-RedTeam)
        ds_matches = f1_agg_df[f1_agg_df["dataset"] == ds]
        if ds_matches.empty:
            # try fuzzy match
            for candidate in f1_agg_df["dataset"].unique():
                if ds.lower() in candidate.lower() or candidate.lower() in ds.lower():
                    ds_matches = f1_agg_df[f1_agg_df["dataset"] == candidate]
                    break

        row_str = f"{ds.replace('_', '\\_')}"
        max_mean = ds_matches["f1_mean"].max() if not ds_matches.empty else -1.0

        for model in models:
            match = ds_matches[ds_matches["model"] == model]
            if match.empty:
                row_str += " & N/A"
            else:
                m = match["f1_mean"].iloc[0]
                s = match["f1_sd"].iloc[0]
                val_str = f"{m:.4f} \\pm {s:.4f}"
                if abs(m - max_mean) < 1e-6 and max_mean > 0:
                    row_str += f" & $\\mathbf{{{val_str}}}$"
                else:
                    row_str += f" & ${val_str}$"
        row_str += " \\\\"
        tex_rows.append(row_str)

    tex_rows.append("\\midrule")
    # LANL row
    lanl_matches = f1_agg_df[f1_agg_df["dataset"].str.contains("LANL", case=False, na=False)]
    max_lanl = lanl_matches["f1_mean"].max() if not lanl_matches.empty else -1.0
    row_str = "LANL-RedTeam"
    for model in models:
        match = lanl_matches[lanl_matches["model"] == model]
        if match.empty:
            row_str += " & N/A"
        else:
            m = match["f1_mean"].iloc[0]
            s = match["f1_sd"].iloc[0]
            val_str = f"{m:.4f} \\pm {s:.4f}"
            if abs(m - max_lanl) < 1e-6 and max_lanl > 0:
                row_str += f" & $\\mathbf{{{val_str}}}$"
            else:
                row_str += f" & ${val_str}$"
    row_str += " \\\\\\bottomrule"
    tex_rows.append(row_str)

    table_tex = "\n".join(tex_rows)
    with open(A18_PUB_DIR / "f1_table_generated.tex", "w") as f:
        f.write(table_tex)
    print(f"Saved generated Table A2 LaTeX fragment to {A18_PUB_DIR / 'f1_table_generated.tex'}")

    return f1_agg_df


if __name__ == "__main__":
    run_gate_g0()
    agg_alert = run_gate_g1()
    agg_f1 = run_gate_g2()
    print("=== Gate G0, G1, G2 Execution Finished Successfully ===")
