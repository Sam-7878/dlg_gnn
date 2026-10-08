#!/usr/bin/env python3
"""Build A05 publication tables with only validated BitcoinOTC replacements."""
from __future__ import annotations
import csv
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
V2 = ROOT / "evaluation/benchmark/v2"
OUT = V2 / "paper_ready_a05"
BUNDLE = OUT / "publication_evidence_a05"
A04 = V2 / "scripts/a04_build_release.py"
spec = importlib.util.spec_from_file_location("a04_build_release", A04)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.OUT = OUT
module.BUNDLE = BUNDLE


def main() -> None:
    raw_dir = ROOT / "outputs/benchmark/a05_bitcoinotc/raw"
    scores_dir = ROOT / "outputs/benchmark/a05_bitcoinotc/scores"
    config_rows = list(csv.DictReader((V2 / "paper_ready_final/a05_config_provenance_crypto_lanl.csv").open()))
    config_by_run = {row["run_id"]: row for row in config_rows}
    assert len(config_by_run) == 120
    originals, rejected = module.load_records()
    new_records = []
    for model in module.MODELS:
        for seed in range(42, 47):
            path = raw_dir / f"BitcoinOTC__{model}__seed{seed}.json"
            score_path = scores_dir / f"BitcoinOTC__{model}__seed{seed}.npy"
            if not path.is_file() or not score_path.is_file():
                raise RuntimeError(f"A05 BitcoinOTC run or score missing: {model}/{seed}")
            d = json.loads(path.read_text())
            if d["status"] != "success":
                raise RuntimeError(f"A05 BitcoinOTC run failed: {model}/{seed}: {d['status']}")
            import numpy as np
            scores = np.load(score_path)
            if module.hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest() != d["raw_score_sha256"]:
                raise RuntimeError(f"A05 raw score hash mismatch: {model}/{seed}")
            if module.sha(score_path) != d["raw_score_file_sha256"]:
                raise RuntimeError(f"A05 score file hash mismatch: {model}/{seed}")
            key = ("BitcoinOTC", model, seed)
            old = originals[key]
            rejected.append({"path": old["source_path"], "reason": "A05 superseded due to historical feature identity mismatch"})
            record = {"dataset": "BitcoinOTC", "model": model, "seed": seed,
                      "run_id": d["run_id"], "status": "success", "source_path": str(path.relative_to(ROOT)),
                      "source_kind": "a05_bitcoinotc_json", "metric_json_hash": module.sha(path),
                      "raw_score_hash": d["raw_score_sha256"], "source_config_hash": d["model_config_hash"],
                      "source_backend_hash": None, "nodes": scores.size,
                      "edges": None, "pr_auc": d["pr_auc"], "roc_auc": d["roc_auc"],
                      "validation_f1": d["validation_f1"],
                      "precision_at_k": d.get("precision_at_k"), "recall_at_k": d.get("recall_at_k"),
                      "topk_f1": d.get("topk_f1")}
            originals[key] = record
            new_records.append(d)
    if len(new_records) != 35:
        raise RuntimeError("A05 BitcoinOTC requires all 35 runs")
    dgraph_raw=ROOT/"outputs/benchmark/a05_anomalydae_large/raw"
    dgraph_paths=[dgraph_raw/f"DGraphFin__AnomalyDAE__seed{seed}.json" for seed in range(42,47)]
    dgraph_ready=all(p.is_file() and json.loads(p.read_text()).get("status")=="success" for p in dgraph_paths)
    if dgraph_ready:
        import numpy as np
        for path in dgraph_paths:
            d=json.loads(path.read_text())
            score_path=ROOT/d["raw_score_path"]
            scores=np.load(score_path)
            if (module.sha(score_path)!=d["raw_score_file_sha256"] or
                module.hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest()!=d["raw_score_sha256"]):
                raise RuntimeError(f"A05 DGraphFin score checksum mismatch: {path}")
            key=("DGraphFin","AnomalyDAE",int(d["seed"]))
            if key in originals:
                rejected.append({"path":originals[key]["source_path"],
                    "reason":"A05 superseded by five-seed exact DGraphFin AnomalyDAE completion"})
            originals[key]={"dataset":"DGraphFin","model":"AnomalyDAE","seed":int(d["seed"]),
                "run_id":d["run_id"],"status":"success","source_path":str(path.relative_to(ROOT)),
                "source_kind":"a05_dgraphfin_json","metric_json_hash":module.sha(path),
                "raw_score_hash":d["raw_score_sha256"],"source_config_hash":d["model_config_hash"],
                "source_backend_hash":None,"nodes":scores.size,"edges":None,
                "pr_auc":d["pr_auc"],"roc_auc":d["roc_auc"],"validation_f1":d["validation_f1"],
                "precision_at_k":d.get("precision_at_k"),"recall_at_k":d.get("recall_at_k"),
                "topk_f1":d.get("topk_f1")}
            new_records.append(d)
    a04_registry = {row["run_id"]: row for row in csv.DictReader(
        (V2 / "paper_ready_final/publication_evidence_a04/approved_run_registry.csv").open())}
    OUT.mkdir(parents=True, exist_ok=True)
    BUNDLE.mkdir(parents=True, exist_ok=True)
    legacy_metrics = V2 / "paper_ready_final/publication_evidence_a04/metric_json"
    if legacy_metrics.is_dir():
        shutil.copytree(legacy_metrics, BUNDLE / "legacy_metric_json", dirs_exist_ok=True)
    legacy_environment = V2 / "paper_ready_final/publication_evidence_a04/environment"
    if legacy_environment.is_dir():
        shutil.copytree(legacy_environment, BUNDLE / "environment", dirs_exist_ok=True)
    for source, target in (
        (V2 / "diagnostics/anomalydae_chunked/anomalydae_exact_equivalence.json", "anomalydae_exact_equivalence.json"),
        (V2 / "diagnostics/conad_dominant/conad_eq1_integration_a04.json", "conad_eq1_integration_a04.json"),
        (V2 / "paper_ready_final/table_memory_selected_a04.csv", "table_memory_selected_a04.csv"),
        (V2 / "a05/dgraphfin_hardware_runtime_note.json", "dgraphfin_hardware_runtime_note.json"),
        (V2 / "a05/source_versions_manifest.json", "source_versions_manifest.json"),
    ):
        if source.is_file(): shutil.copy2(source, BUNDLE / target)
    source_versions = V2 / "a05/source_versions"
    if source_versions.is_dir():
        shutil.copytree(source_versions, BUNDLE / "source_versions", dirs_exist_ok=True)
    source_files=(
        "evaluation/benchmark/v2/scripts/a05_run_bitcoinotc.py",
        "evaluation/benchmark/v2/scripts/a05_run_anomalydae_large.py",
        "evaluation/benchmark/v2/scripts/a05_continue_dgraphfin.sh",
        "evaluation/benchmark/v2/scripts/a05_anomalydae_equivalence.py",
        "evaluation/benchmark/v2/scripts/a05_classify_config_provenance.py",
        "evaluation/benchmark/v2/scripts/a05_finalize_anomalydae.py",
        "evaluation/benchmark/v2/scripts/a05_write_manuscript.py",
        "evaluation/benchmark/v2/scripts/a04_build_dataset_manifest.py",
        "evaluation/benchmark/v2/scripts/a04_build_memory_selected.py",
        "evaluation/benchmark/v2/scripts/a05_build_measured_memory_figure.py",
        "evaluation/benchmark/v2/scripts/a03_run_crypto_production.py",
        "evaluation/benchmark/v2/scripts/a03_rerun_crypto_dlg_aug.py",
        "evaluation/benchmark/v2/scripts/a05_build_release.py",
        "evaluation/benchmark/v2/scripts/90_validate_results_a05_journal.py",
        "src/gog_fraud/pipelines/run_sci_round4c.py",
        "src/gog_fraud/models/pygod/shared_reconstruction.py",
        "src/gog_fraud/models/pygod/exact_reconstruction.py",
        "configs/benchmark/sci_round5_final.yaml",
        "docs/architecture/benchmark/DLG_Benchmark_v2_Protocol_FROZEN_2026-09-30.md",
        "docs/architecture/benchmark/DLG_Benchmark_v2_Protocol_Amendment_A04_2026-10-03.md",
        "docs/architecture/benchmark/DLG_Benchmark_v2_Protocol_Amendment_A05_2026-10-03.md",
    )
    source_inventory=[]
    for relative in source_files:
        source=ROOT/relative
        target=BUNDLE/"source_snapshot"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        source_inventory.append({"path":relative,"sha256":module.sha(source)})
    lanl_revision="f0d63791da5dc031d34dd4ed0c9f9deba1020d74"
    for relative in ("scripts/defense_extension_real/run_defense_multiseed_real.py",
                     "configs/benchmark/sci_defense_extension_real.yaml"):
        payload=subprocess.check_output(["git","show",f"{lanl_revision}:{relative}"],cwd=ROOT)
        target=BUNDLE/"source_snapshot/historical_lanl"/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(payload)
        source_inventory.append({"path":f"{lanl_revision}:{relative}",
                                 "sha256":module.hashlib.sha256(payload).hexdigest()})
    module.write_json(BUNDLE/"source_snapshot_manifest.json",source_inventory)
    for d in new_records:
        source = (raw_dir if d["dataset"]=="BitcoinOTC" else dgraph_raw) / f"{d['dataset']}__{d['model']}__seed{d['seed']}.json"
        score = ROOT / d["raw_score_path"]
        raw_sub="bitcoinotc_raw" if d["dataset"]=="BitcoinOTC" else "anomalydae_raw"
        score_sub="bitcoinotc_scores" if d["dataset"]=="BitcoinOTC" else "anomalydae_scores"
        (BUNDLE / raw_sub).mkdir(exist_ok=True)
        (BUNDLE / score_sub).mkdir(exist_ok=True)
        shutil.copy2(source, BUNDLE / raw_sub / source.name)
        shutil.copy2(score, BUNDLE / score_sub / score.name)
    registry = []
    for rec in sorted(originals.values(), key=lambda x:(x["dataset"], x["model"], x["seed"])):
        d = next((d for d in new_records if d["run_id"] == rec["run_id"]), None)
        cfg = config_by_run.get(rec["run_id"])
        base = a04_registry.get(rec["run_id"], {})
        registry.append({**base, **rec,
                         "dataset_hash": d["constructed_tensor_sha256"] if d else base.get("dataset_hash"),
                         "feature_hash": d["feature_hash"] if d else base.get("feature_hash"),
                         "edge_hash": d["edge_hash"] if d else base.get("edge_hash"),
                         "label_hash": d["label_hash"] if d else base.get("label_hash"),
                         "split_hash": d["split_hash"] if d else base.get("split_hash"),
                         "model_config_hash": d["model_config_hash"] if d else
                                               cfg["reconstructed_config_sha256"] if cfg else rec["source_config_hash"],
                         "model_config_provenance_status": "CONFIG_EXACT_RECOVERED" if d else
                                                           cfg["classification"] if cfg else "ORIGINAL_RUN_CONFIG_HASH",
                         "environment_lock_hash": d["environment_lock_sha256"] if d else base.get("environment_lock_hash"),
                         "source_commit": d["source_commit"] if d else base.get("source_commit"),
                         "raw_score_path": d["raw_score_path"] if d else None,
                         "raw_score_status": "ARCHIVED_A05_SCORE" if d else base.get("raw_score_status"),
                         "dataset_verification_status": "A05_CANONICAL_RERUN" if d else base.get("dataset_verification_status"),
                         "metric_json_path": str((BUNDLE/("bitcoinotc_raw" if rec["dataset"]=="BitcoinOTC" else "anomalydae_raw")/f"{rec['dataset']}__{rec['model']}__seed{rec['seed']}.json").relative_to(ROOT)) if d else base.get("metric_json_path"),
                         "provenance_complete": bool(d) if d else False,
                         "provenance_tier": "R" if d else "P"})
    module.write_csv(BUNDLE / "approved_run_registry.csv", registry, list(registry[0]))
    module.write_json(BUNDLE / "rejected_or_superseded_runs.json", rejected)
    support, values, cells = module.aggregate(originals)
    # BitcoinOTC has injected node labels on a real trust graph. The A04
    # grouping name "real" denotes portfolio membership, not label truth.
    for filename in module.METRICS.values():
        metric_rows = list(csv.DictReader((OUT / filename).open()))
        for row in metric_rows:
            if row["dataset"] == "BitcoinOTC":
                row["category"] = "synthetic_injection_on_real_trust_graph"
        module.write_csv(OUT / filename, metric_rows, list(metric_rows[0]))
    statistics = module.stats(values)
    module.diagnostics()
    dataset_manifest=json.loads((V2/"paper_ready_final/dataset_manifest_canonical.json").read_text())
    old_freeze=json.loads((ROOT/"outputs/benchmark/sci_round5_final/manifests/data_freeze.json").read_text())
    old_bitcoin=next(row for row in old_freeze["datasets"] if row["dataset"]=="BitcoinOTC")
    for row in dataset_manifest:
        row["a05_release_builder_sha256"]=module.sha(Path(__file__))
        row["historical_round5_feature_hash"] = old_bitcoin["feature_hash"] if row["dataset_id"]=="BitcoinOTC" else None
        row["a05_resolution_status"] = "35_CANONICAL_TENSOR_RERUNS_VERIFIED" if row["dataset_id"]=="BitcoinOTC" else None
        if row["dataset_id"]=="BitcoinOTC":
            row["verification_status"]="A05_CANONICAL_RERUN_VERIFIED"
            row["source_evidence"]=(row.get("source_evidence") or "")+"; A05 superseding 35-run canonical-tensor campaign"
    module.write_json(BUNDLE/"dataset_manifest_canonical.json",dataset_manifest)
    module.write_csv(BUNDLE/"dataset_manifest_canonical.csv",dataset_manifest,list(dataset_manifest[0]))
    shutil.copy2(V2 / "paper_ready_final/a05_config_provenance_crypto_lanl.csv", BUNDLE / "a05_config_provenance_crypto_lanl.csv")
    shutil.copy2(V2 / "paper_ready_final/figure_memory_measured_a05.png", OUT / "figure_memory_measured_a05.png")
    shutil.copy2(V2 / "paper_ready_final/figure_memory_measured_a05.pdf", OUT / "figure_memory_measured_a05.pdf")
    module.write_json(OUT / "publication_manifest_a05.json", {
        "edition": "A05_CANDIDATE", "primary_datasets": module.PRIMARY,
        "external_datasets": ["LANL-RedTeam"], "primary_models": module.MODELS,
        "diagnostic_models": [module.DIAGNOSTIC_MODEL], "approved_run_count": len(registry),
        "a05_bitcoinotc_replacements": 35, "a05_dgraphfin_anomalydae_runs":5 if dgraph_ready else 0,
        "publication_metric_cell_count": len(cells),
        "s3_friedman_p": next(r["friedman_p"] for r in statistics if r["view"].startswith("S3_")),
        "gate_g4_journal": "PENDING_A05_VALIDATION"})
    print(json.dumps({"approved_runs":len(registry),"bitcoinotc_replacements":35,
                      "metric_cells":len(cells),"support_pairs":len(support)},indent=2))


if __name__ == "__main__":main()
