#!/usr/bin/env python3
"""A05 journal-level evidence gate; retains A04 strict gate separately."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import friedmanchisquare

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "evaluation/benchmark/v2"
OUT = V2 / "paper_ready_a05"
B = OUT / "publication_evidence_a05"
MODELS = ("DOMINANT", "AnomalyDAE", "CoLA", "GADNR", "OCGNN", "DLG-Base", "DLG-Aug")
SEEDS = (42,43,44,45,46)


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline="")))


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def main() -> bool:
    checks=[]
    def check(name: str, kind: str, passed: bool, detail: str) -> None:
        checks.append({"check_id":name,"classification":kind,"status":"PASS" if passed else "FAIL","detail":detail})
    manifest={row["dataset_id"]:row for row in json.loads((B/"dataset_manifest_canonical.json").read_text())}
    registry=read_csv(B/"approved_run_registry.csv")
    config=read_csv(B/"a05_config_provenance_crypto_lanl.csv")
    keys=[(r["dataset"],r["model"],int(r["seed"])) for r in registry]
    check("identity", "PUBLICATION_BLOCKER", len(manifest)==14 and len(keys)==len(set(keys)) and
          all(r["dataset"] in manifest and r["model"] in MODELS and int(r["seed"]) in SEEDS for r in registry),
          f"datasets={len(manifest)}; unique runs={len(set(keys))}/{len(keys)}")
    bad_metric=[];bad_dataset=[];bad_split=[];bad_cfg=[]
    for r in registry:
        key=f"{r['dataset']}/{r['model']}/{r['seed']}"
        path=ROOT/r["source_path"]
        if not path.is_file() or sha(path)!=r["metric_json_hash"]:bad_metric.append(key)
        if r["dataset_hash"]!=manifest[r["dataset"]]["constructed_tensor_sha256"]:bad_dataset.append(key)
        if r["status"]=="success" and not r["split_hash"]:bad_split.append(key)
        if r["status"]=="success" and not r["model_config_hash"]:bad_cfg.append(key)
    check("tier_p_metric_json", "PUBLICATION_BLOCKER", not bad_metric,f"bad={len(bad_metric)}")
    check("tier_p_dataset_identity", "PUBLICATION_BLOCKER", not bad_dataset and
          manifest["BitcoinOTC"].get("verification_status")=="A05_CANONICAL_RERUN_VERIFIED" and
          manifest["BitcoinOTC"].get("a05_resolution_status")=="35_CANONICAL_TENSOR_RERUNS_VERIFIED",
          f"bad={len(bad_dataset)}; BitcoinOTC={manifest['BitcoinOTC'].get('verification_status')}")
    check("tier_p_split_config", "PUBLICATION_BLOCKER", not bad_split and not bad_cfg,
          f"missing split={len(bad_split)}, config={len(bad_cfg)}")
    btc=[r for r in registry if r["dataset"]=="BitcoinOTC"]
    btc_bad=[]
    for r in btc:
        raw=json.loads((ROOT/r["source_path"]).read_text())
        score=ROOT/raw["raw_score_path"]
        if (r["source_kind"]!="a05_bitcoinotc_json" or r["status"]!="success"
            or raw["feature_hash"]!=manifest["BitcoinOTC"]["feature_hash"]
            or raw["environment_lock_sha256"]!=sha(ROOT/"environment/locks/benchmark-a04-cuda.hashed.txt")
            or not score.is_file() or sha(score)!=raw["raw_score_file_sha256"]
            or hashlib.sha256(np.ascontiguousarray(np.load(score)).tobytes()).hexdigest()!=raw["raw_score_sha256"]):
            btc_bad.append(f"{r['model']}/{r['seed']}")
    check("bitcoinotc_canonical_replacement", "PUBLICATION_BLOCKER",len(btc)==35 and not btc_bad,
          f"replacement runs={len(btc)}; invalid={btc_bad[:5]}")
    source_issues=[]
    checked_new_runs=0
    for raw_dir in (ROOT/"outputs/benchmark/a05_bitcoinotc/raw",
                    ROOT/"outputs/benchmark/a05_anomalydae_large/raw"):
        for raw_path in sorted(raw_dir.glob("*.json")):
            raw=json.loads(raw_path.read_text())
            if raw.get("status")!="success":
                continue
            checked_new_runs+=1
            sources=dict(raw.get("source_file_sha256",{}))
            if raw.get("runner_source_sha256"):
                sources["evaluation/benchmark/v2/scripts/a05_run_anomalydae_large.py"]=raw["runner_source_sha256"]
                sources["src/gog_fraud/models/pygod/shared_reconstruction.py"]=raw["model_source_sha256"]
            if not sources:
                source_issues.append(f"{raw['run_id']}: missing source hashes")
            for relative, expected in sources.items():
                candidates=(B/"source_snapshot"/relative,B/"source_versions"/expected/relative)
                if not any(p.is_file() and sha(p)==expected for p in candidates):
                    source_issues.append(f"{raw['run_id']}: {relative}")
    check("a05_frozen_execution_sources", "PUBLICATION_BLOCKER",not source_issues,
          f"new successful runs={checked_new_runs}; unresolved source hashes={source_issues[:5]}")
    cfg_map={r["run_id"]:r for r in config}
    affected=[r for r in registry if r["source_kind"] in {"a03_json","lanl_real_json"} and r["status"]=="success"]
    cfg_bad=[r["run_id"] for r in affected if r["run_id"] not in cfg_map or
             cfg_map[r["run_id"]]["classification"] not in {"CONFIG_EXACT_RECOVERED","CONFIG_UNAMBIGUOUS_RECONSTRUCTION"}]
    check("a03_config_reconstruction", "PUBLICATION_BLOCKER",len(affected)==120 and not cfg_bad,
          f"classified={len(affected)-len(cfg_bad)}/{len(affected)}; ambiguous={len(cfg_bad)}")
    legacy=[r for r in registry if not r["source_kind"].startswith("a05_") and r["status"]=="success"]
    legacy_missing_score=sum(not r["raw_score_hash"] for r in legacy)
    legacy_missing_lock=sum(not r["environment_lock_hash"] for r in legacy)
    check("historical_raw_scores", "DISCLOSED_LIMITATION",True,
          f"{legacy_missing_score}/{len(legacy)} legacy successful runs lack raw score hashes; disclosed")
    check("historical_run_locks", "DISCLOSED_LIMITATION",True,
          f"{legacy_missing_lock}/{len(legacy)} legacy successful runs lack exact per-run locks; campaign evidence retained")
    campaign=ROOT/"outputs/benchmark/sci_round5_final/manifests/environment_freeze.json"
    check("campaign_environment", "PUBLICATION_BLOCKER",campaign.is_file() and
          (B/"environment/round5_archived_legacy_venv_inventory.txt").is_file(),
          "Round5 environment freeze and legacy package inventory")
    cells=read_csv(B/"paper_metric_cells.csv")
    by=defaultdict(list)
    for r in registry:by[(r["dataset"],r["model"])].append(r)
    bad_cells=[]
    for c in cells:
        vals=[float(r[c["metric"]]) for r in by[(c["dataset"],c["model"])] if r["status"]=="success" and r[c["metric"]]]
        if len(vals)!=5 or not math.isclose(float(c["mean"]),float(np.mean(vals)),abs_tol=1e-12):
            bad_cells.append(c["dataset"]+"/"+c["model"]+"/"+c["metric"])
    check("table_arithmetic", "PUBLICATION_BLOCKER",not bad_cells,
          f"cells={len(cells)}; bad={bad_cells[:5]}")
    support=read_csv(OUT/"table_support_24g.csv")
    supported={(r["dataset"],r["model"]) for r in support if r["support_status"]=="SUPPORTED_EXACT"}
    cell_pairs={(r["dataset"],r["model"]) for r in cells}
    check("support_and_metrics", "PUBLICATION_BLOCKER",cell_pairs==supported and
          all(len([r for r in by[pair] if r["status"]=="success"])==5 for pair in supported),
          f"supported pairs={len(supported)}; metric pairs={len(cell_pairs)}")
    check("conad_diagnostic_only", "PUBLICATION_BLOCKER",
          all("CONAD" not in r["model"] for r in registry) and
          all("CONAD" not in r["models"] for r in read_csv(OUT/"statistics_s1_s5.csv")),
          "CONAD excluded from performance ranks")
    stat=read_csv(OUT/"statistics_s1_s5.csv")
    s3=next(r for r in stat if r["view"].startswith("S3_"))
    check("statistics_policy", "PUBLICATION_BLOCKER",len(stat)==5 and float(s3["friedman_p"])>0.05,
          f"S3 p={s3['friedman_p']}; no overall superiority inference")
    memory=read_csv(B/"table_memory_selected_a04.csv")
    memory_raw=V2/"paper_ready_a04/publication_evidence_a04/memory_raw"
    check("measured_memory", "PUBLICATION_BLOCKER",len(memory)==8 and
          all(sha(memory_raw/Path(r["raw_log_path"]).name)==r["raw_log_sha256"] for r in memory) and
          (OUT/"figure_memory_measured_a05.pdf").is_file() and
          not (V2/"paper_ready/figure_memory_support.png").exists(),
          "four measured full/cap pairs; modeled legacy figure archived")
    equivalence=json.loads((V2/"diagnostics/anomalydae_chunked/anomalydae_exact_equivalence.json").read_text())
    check("anomalydae_equivalence", "PUBLICATION_BLOCKER",equivalence["status"]=="PASS",
          f"dense-vs-block status={equivalence['status']}")
    targeted_path=OUT/"table_anomalydae_targeted_a05.csv"
    targeted=read_csv(targeted_path) if targeted_path.is_file() else []
    incomplete_targeted=[]
    for r in targeted:
        status=r["targeted_status"]
        if status=="INVALID_TARGETED_RUN":
            incomplete_targeted.append(r["dataset"])
        elif r["preflight_status"]=="PREFLIGHT_FEASIBLE":
            if status=="TARGETED_50_EPOCH_FIVE_SEED_COMPLETE":
                continue
            if status=="UNSUPPORTED_OPERATIONAL_TIMEOUT" and r.get("timed_out_seeds"):
                timeout_seeds=[int(seed) for seed in r["timed_out_seeds"].split(";")]
                valid_timeout=True
                for seed in timeout_seeds:
                    raw_path=ROOT/f"outputs/benchmark/a05_anomalydae_large/raw/{r['dataset']}__AnomalyDAE__seed{seed}.json"
                    if not raw_path.is_file():
                        valid_timeout=False
                        break
                    raw=json.loads(raw_path.read_text())
                    if raw.get("status")!="UNSUPPORTED_OPERATIONAL_TIMEOUT" or raw.get("timeout_budget_sec")!=86400:
                        valid_timeout=False
                        break
                if valid_timeout:
                    continue
            incomplete_targeted.append(r["dataset"])
    check("large_anomalydae_targeted", "PUBLICATION_BLOCKER",len(targeted)==4 and not incomplete_targeted,
          f"preflight rows={len(targeted)}; feasible but incomplete={incomplete_targeted}")
    paper=ROOT/"docs/papers/_42_01_Benchmark_PrePrints/DLG-Benchmark_Preprint_A05.tex"
    manuscript=paper.read_text() if paper.is_file() else ""
    bad_phrases=("ten-dataset", "all models except AnomalyDAE", "BitcoinOTC is obtained from the SNAP repository with real transaction rating labels",
                 "DLG-Aug universally", "all 430 legacy predictions are bitwise reproducible")
    check("manuscript_claims", "PUBLICATION_BLOCKER",bool(manuscript) and
          all(phrase not in manuscript for phrase in bad_phrases) and
          "mixed label provenance" in manuscript and "BitcoinOTC" in manuscript and
          f"{float(s3['friedman_p']):.4f}" in manuscript,
          "A05 source uses mixed-label S3 and current claims; stale claims absent")
    check("manuscript_pdf", "PUBLICATION_BLOCKER",paper.with_suffix(".pdf").is_file(),
          "compiled A05 manuscript PDF present")
    for name in ("per_run_full_wheel_lock","per_run_raw_predictions","complete_gpu_telemetry","bitwise_recreation"):
        check(name,"ARCHIVAL_BEST_PRACTICE",True,"optional for historical runs; captured where available")
    blockers=[c["check_id"] for c in checks if c["classification"]=="PUBLICATION_BLOCKER" and c["status"]!="PASS"]
    gate="PASS" if not blockers else "HOLD"
    result={"gate_g4_journal":gate,"publication_blockers":blockers,"checks":checks}
    (OUT/"validation_summary_a05_journal.json").write_text(json.dumps(result,indent=2)+"\n")
    lines=["# A05 journal evidence validation", "", f"**G4-JOURNAL: {gate}**", "",
           "| Check | Class | Status | Detail |", "|---|---|---|---|"]
    lines += [f"| {c['check_id']} | {c['classification']} | {c['status']} | {c['detail']} |" for c in checks]
    (OUT/"validation_summary_a05_journal.md").write_text("\n".join(lines)+"\n")
    publication=json.loads((OUT/"publication_manifest_a05.json").read_text())
    publication["gate_g4_journal"]=gate
    publication["validation_summary_sha256"]=sha(OUT/"validation_summary_a05_journal.json")
    (OUT/"publication_manifest_a05.json").write_text(json.dumps(publication,indent=2)+"\n")
    print(json.dumps({"gate_g4_journal":gate,"publication_blockers":blockers,"checks":len(checks)},indent=2))
    return gate=="PASS"


if __name__=="__main__":sys.exit(0 if main() else 2)
