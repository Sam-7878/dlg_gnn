#!/usr/bin/env python3
"""Audit A05 large AnomalyDAE preflights and any completed targeted seeds."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[4]
V2=ROOT/"evaluation/benchmark/v2"
OUT=V2/"paper_ready_a05"
BUNDLE=OUT/"publication_evidence_a05"
RAW=ROOT/"outputs/benchmark/a05_anomalydae_large/raw"
SCORES=ROOT/"outputs/benchmark/a05_anomalydae_large/scores"
DATASETS=("Ethereum","DGraphFin","Yelp-Syn","Reddit-Syn")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rows=[]
    for dataset in DATASETS:
        pre_path=V2/f"diagnostics/anomalydae_chunked/preflight_{dataset}.json"
        pre=json.loads(pre_path.read_text())
        if "RTX 3090" not in pre.get("gpu_name",""):
            raise RuntimeError(f"incorrect preflight hardware: {dataset}")
        records=[];invalid=[];timed_out=[]
        for seed in range(42,47):
            path=RAW/f"{dataset}__AnomalyDAE__seed{seed}.json"
            if not path.is_file():continue
            r=json.loads(path.read_text())
            if r["status"]=="UNSUPPORTED_OPERATIONAL_TIMEOUT":
                if r.get("timeout_budget_sec")==86400 and r.get("completed_epochs_at_timeout") is not None:
                    timed_out.append(seed)
                else:
                    invalid.append(seed)
                continue
            if r["status"]!="success":invalid.append(seed);continue
            score=ROOT/r["raw_score_path"]
            if (not score.is_file() or sha(score)!=r["raw_score_file_sha256"] or
                hashlib.sha256(np.ascontiguousarray(np.load(score)).tobytes()).hexdigest()!=r["raw_score_sha256"] or
                "RTX 3090" not in r["device"] or r["actual_epochs"]!=50):
                invalid.append(seed);continue
            records.append(r)
            for source, sub in ((path,"anomalydae_raw"),(score,"anomalydae_scores")):
                (BUNDLE/sub).mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,BUNDLE/sub/source.name)
        if invalid: status="INVALID_TARGETED_RUN"
        elif len(records)==5: status="TARGETED_50_EPOCH_FIVE_SEED_COMPLETE"
        elif timed_out: status="UNSUPPORTED_OPERATIONAL_TIMEOUT"
        elif pre["status"]=="PREFLIGHT_FEASIBLE":status="FEASIBLE_EXECUTION_PENDING"
        else:status=pre["status"]
        rows.append({"dataset":dataset,"preflight_status":pre["status"],"targeted_status":status,
            "gpu_name":pre["gpu_name"],"block_size":pre["block_size"],
            "sampled_blocks":pre.get("sampled_blocks"),
            "projected_50_epoch_hours":round(pre.get("projected_50_epoch_lower_bound_seconds",0)/3600,3) if pre.get("projected_50_epoch_lower_bound_seconds") else None,
            "successful_seeds":";".join(str(r["seed"]) for r in records),"invalid_seeds":";".join(map(str,invalid)),
            "timed_out_seeds":";".join(map(str,timed_out)),
            "mean_roc_auc":statistics.mean(r["roc_auc"] for r in records) if len(records)==5 else None,
            "mean_pr_auc":statistics.mean(r["pr_auc"] for r in records) if len(records)==5 else None,
            "mean_validation_f1":statistics.mean(r["validation_f1"] for r in records) if len(records)==5 else None,
            "max_peak_allocated_mib":max((r["peak_allocated_mib"] for r in records),default=None),
            "evidence_path":str(pre_path.relative_to(ROOT))})
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"table_anomalydae_targeted_a05.csv"
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/"anomalydae_targeted_summary_a05.json").write_text(json.dumps(rows,indent=2)+"\n")
    print(json.dumps({r["dataset"]:r["targeted_status"] for r in rows},indent=2))


if __name__=="__main__":main()
