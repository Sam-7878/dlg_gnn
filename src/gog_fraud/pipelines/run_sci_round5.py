"""Round 5 Phase-0 qualification and support-aware five-seed execution."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from gog_fraud.experiments.round5_policy import sha256_file, supported_run_count, validate_support_matrix
from gog_fraud.pipelines.run_sci_round4c import benchmark_execution_source_hash, ensure_layout, hashes, result_path


HEAVY = ("AnomalyDAE", "GADNR")
REPRESENTATIVE = ("Cora", "Elliptic", "Yelp", "DGraphFin", "Reddit")
PHASE1_DATASET_ORDER = ("Cora", "CiteSeer", "BitcoinOTC", "Amazon", "PubMed", "Elliptic", "Flickr", "Yelp", "DGraphFin", "Reddit")
PHASE1_MODEL_ORDER = ("CoLA", "OCGNN", "DLG-Aug", "DLG-Base", "DOMINANT", "CONAD", "GADNR", "AnomalyDAE")


def _read_result(config: dict, output: Path, dataset: str, model: str, seed: int) -> dict | None:
    path = result_path(config, output, dataset, model, seed)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _run_fresh(config_path: Path, config: dict, dataset: str, model: str, seed: int,
               *, retry: bool = False, force: bool = False) -> dict:
    output = Path(config["experiment"]["output_root"])
    path = result_path(config, output, dataset, model, seed)
    if (not force and path.exists()
            and json.loads(path.read_text(encoding="utf-8")).get("status") == "success"):
        return json.loads(path.read_text(encoding="utf-8"))
    if path.exists() and not retry:
        return json.loads(path.read_text(encoding="utf-8"))
    command = [sys.executable, "-m", "gog_fraud.pipelines.run_sci_round4c", "--config", str(config_path),
               "--stage", "cell", "--dataset", dataset, "--model", model, "--seed", str(seed)]
    log = output / "logs" / f"{dataset}__{model}__seed{seed}.log"
    started = time.perf_counter()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"\nROUND5 fresh attempt started retry={retry}\n")
        try:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=float(config["execution"]["max_run_wall_hours"])*3600, check=False)
        except subprocess.TimeoutExpired:
            return {"dataset":dataset,"model":model,"seed":seed,"status":"unsupported_operational",
                    "total_wall_sec":time.perf_counter()-started,"failure_type":"WallClockTimeout"}
    if completed.returncode != 0 and not path.exists():
        raise RuntimeError(f"cell subprocess exited {completed.returncode} without an evidence record")
    return json.loads(path.read_text(encoding="utf-8"))


def phase0(config_path: Path, config: dict, *, force: bool = False) -> dict:
    output = Path(config["experiment"]["output_root"]); ensure_layout(output)
    remaining = list(config["remaining_five"])
    evidence = []
    for model in ("DOMINANT", "AnomalyDAE", "GADNR"):
        for dataset in remaining:
            print(f"[PHASE0] {dataset}/{model}/seed=42", flush=True)
            record = _run_fresh(config_path, config, dataset, model, 42, force=force)
            evidence.append({key:record.get(key) for key in (
                "dataset","model","seed","status","failure_type","failure_message","total_wall_sec",
                "actual_epochs","nodes","edges","config_hash","backend_hash")})
            print(f"[PHASE0-DONE] {dataset}/{model} {record.get('status')}", flush=True)
            if model == "DOMINANT" and record.get("status") != "success":
                raise RuntimeError(f"DOMINANT Phase-0 acceptance failed for {dataset}")
    ledger = pd.DataFrame(evidence)
    ledger.to_csv(output / "support/phase0_qualification_evidence.csv", index=False)
    return freeze_support_v2(config, output, ledger)


def _restriction_from_phase0(row: pd.Series) -> tuple[str, str]:
    status = str(row.status)
    if status == "success":
        return "supported", "seed42 exact 50-epoch production qualification succeeded"
    if status == "failed_oom":
        return "unsupported_resource_exact_implementation", str(row.failure_message)
    if status == "unsupported_operational":
        return "unsupported_operational", "predeclared 24-hour exact production guard exceeded"
    raise RuntimeError(f"unresolved Phase-0 failure: {row.dataset}/{row.model}/{status}")


def freeze_support_v2(config: dict, output: Path, phase0_evidence: pd.DataFrame) -> dict:
    source = Path(config["experiment"]["phase0_source"])
    representative = pd.read_csv(source / "tables/model_dataset_support_matrix_final.csv")
    phase0 = {(row.dataset, row.model): row for row in phase0_evidence.itertuples(index=False)}
    rows = []
    for dataset in config["datasets"]:
        for model in config["models"]:
            if dataset in REPRESENTATIVE:
                prior = representative.loc[representative.dataset.eq(dataset) & representative.model.eq(model)]
                if len(prior) != 1:
                    raise RuntimeError(f"missing representative support evidence: {dataset}/{model}")
                prior_row = prior.iloc[0]
                status = "supported" if bool(prior_row.primary_supported) else str(prior_row.restriction_class)
                reason = "Round4C two-seed exact production success" if status == "supported" else str(prior_row.restriction_reason)
                mode, evidence_path = "round4c_measured", "outputs/sci_round4c/tables/model_dataset_support_matrix_final.csv"
            elif model in HEAVY:
                measured = phase0.get((dataset, model))
                if measured is None:
                    raise RuntimeError(f"missing Phase-0 evidence: {dataset}/{model}")
                status, reason = _restriction_from_phase0(pd.Series(measured._asdict()))
                mode = "round5_phase0_measured"
                evidence_path = f"support/phase0_qualification_evidence.csv#{dataset}/{model}/seed42"
            else:
                dominant = phase0.get((dataset, "DOMINANT"))
                if dominant is None or dominant.status != "success":
                    raise RuntimeError(f"backend sanity not accepted: {dataset}")
                status, reason = "supported", "DOMINANT exact backend sanity passed; non-heavy exact path retained"
                mode, evidence_path = "phase0_backend_plus_round4b_exact_acceptance", "support/phase0_qualification_evidence.csv"
            rows.append({"dataset":dataset,"display_name":config["display_names"][dataset],"model":model,
                         "support_status":status,"supported":status=="supported","evidence_mode":mode,
                         "restriction_reason":reason,"evidence_path":evidence_path})
    support = pd.DataFrame(rows)
    validate_support_matrix(support, list(config["datasets"]), list(config["models"]))
    path = output / "manifests/model_dataset_support_matrix_v2.csv"
    support.to_csv(path, index=False, lineterminator="\n")
    digest = sha256_file(path)
    (output / "manifests/model_dataset_support_matrix_v2.sha256").write_text(digest+"\n", encoding="ascii")
    gate = {"decision":"PHASE0_READY", "pairs":len(support),
            "supported_pairs":int(support.supported.sum()),
            "unsupported_pairs":int((~support.supported).sum()), "unknown":0,
            "support_matrix_hash":digest,
            "target_runs":supported_run_count(support, list(config["seeds"])), "auto_started_phase1":False}
    (output / "manifests/phase0_gate.json").write_text(json.dumps(gate,indent=2)+"\n", encoding="utf-8")
    return gate


def phase1(config_path: Path, config: dict, *, resume: bool, force: bool = False) -> dict:
    output = Path(config["experiment"]["output_root"])
    support_path = output / "manifests/model_dataset_support_matrix_v2.csv"
    support = pd.read_csv(support_path)
    validate_support_matrix(support, list(config["datasets"]), list(config["models"]))
    expected_hash = (output / "manifests/model_dataset_support_matrix_v2.sha256").read_text().strip()
    if sha256_file(support_path) != expected_hash:
        raise RuntimeError("support matrix drift detected")
    violations = []
    for dataset in PHASE1_DATASET_ORDER:
        for model in PHASE1_MODEL_ORDER:
            pair = support.loc[support.dataset.eq(dataset) & support.model.eq(model)].iloc[0]
            if pair.support_status != "supported":
                continue
            for seed in config["seeds"]:
                existing = _read_result(config, output, dataset, model, int(seed))
                if resume and not force and existing is not None and existing.get("status") == "success":
                    print(f"[SKIP] {dataset}/{model}/seed={seed}", flush=True); continue
                print(f"[RUN] {dataset}/{model}/seed={seed}", flush=True)
                first = _run_fresh(config_path, config, dataset, model, int(seed), retry=True, force=force)
                if first.get("status") != "success":
                    first_path = result_path(config, output, dataset, model, int(seed))
                    if first_path.exists():
                        shutil.copy2(first_path, output/"logs"/(first_path.stem+"__attempt1.json"))
                    second = _run_fresh(config_path, config, dataset, model, int(seed), retry=True, force=force)
                    if second.get("status") != "success":
                        violations.append({"dataset":dataset,"model":model,"seed":seed,
                                           "attempt1":first.get("status"),"attempt2":second.get("status")})
                print(f"[DONE] {dataset}/{model}/seed={seed} status={_read_result(config,output,dataset,model,int(seed)).get('status')}", flush=True)
    result = {"support_matrix_hash":expected_hash,"violations":violations,
              "decision":"PHASE1_COMPLETE" if not violations else "SUPPORT_MATRIX_VIOLATION"}
    (output/"manifests/phase1_execution_gate.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


def main() -> int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True)
    parser.add_argument("--stage",choices=("phase0","phase1"),required=True)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--force",action="store_true",
                        help="rerun selected stage cells even when successful evidence already exists")
    args=parser.parse_args();config_path=Path(args.config).resolve()
    config=yaml.safe_load(config_path.read_text(encoding="utf-8"));output=Path(config["experiment"]["output_root"])
    ensure_layout(output)
    for name in ("summary","support","statistics","topology","ablation"):
        (output/name).mkdir(parents=True,exist_ok=True)
    config_hash,backend_hash=hashes(config)
    (output/"manifests/config_snapshot.yaml").write_text(yaml.safe_dump(config,sort_keys=False),encoding="utf-8")
    execution_manifest=output/"manifests/execution_hashes.json"
    current_hashes={
        "config_hash":config_hash,"backend_hash":backend_hash,
        "benchmark_execution_source_hash":benchmark_execution_source_hash(),
    }
    if execution_manifest.exists():
        frozen_hashes=json.loads(execution_manifest.read_text(encoding="utf-8"))
        if frozen_hashes != current_hashes:
            raise RuntimeError(f"Round 5 execution provenance drift: frozen={frozen_hashes} current={current_hashes}")
    else:
        execution_manifest.write_text(json.dumps(current_hashes,indent=2)+"\n",encoding="utf-8")
    result=(phase0(config_path,config,force=args.force) if args.stage=="phase0"
            else phase1(config_path,config,resume=args.resume,force=args.force))
    print(json.dumps(result,indent=2));return 0


if __name__=="__main__": raise SystemExit(main())
