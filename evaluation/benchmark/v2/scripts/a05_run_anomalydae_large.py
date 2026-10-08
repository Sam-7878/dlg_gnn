#!/usr/bin/env python3
"""Targeted 50-epoch AnomalyDAE large-graph run after A05 preflight gate."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"src"))
from a05_run_bitcoinotc import filehash, arrayhash, digest
from gog_fraud.evaluation.reproducibility import seed_everything
from gog_fraud.evaluation.threshold_protocol import evaluate_threshold_protocol
from gog_fraud.pipelines.run_sci_round4c import _instantiate, _models

V2 = ROOT/"evaluation/benchmark/v2"
DATASETS = ("Ethereum", "DGraphFin", "Yelp-Syn", "Reddit-Syn")
OUT = ROOT/"outputs/benchmark/a05_anomalydae_large"
GRAPH = ROOT/"outputs/benchmark/a04_constructed_graphs"
CONFIG = ROOT/"configs/benchmark/sci_round5_final.yaml"
LOCK = ROOT/"environment/locks/benchmark-a04-cuda.hashed.txt"
MANIFEST = V2/"paper_ready_final/dataset_manifest_canonical.json"


def split(data, dataset, seed):
    y=data.y.cpu().numpy().reshape(-1)
    if dataset=="Ethereum":
        perm=np.random.RandomState(seed).permutation(data.num_nodes)
        val=perm[int(.6*len(perm)):int(.8*len(perm))]
        test=perm[int(.8*len(perm)):]
        policy="a03_crypto_seeded_60_20_20"
    elif dataset=="DGraphFin":
        val=np.flatnonzero(data.val_mask.cpu().numpy());test=np.flatnonzero(data.test_mask.cpu().numpy())
        policy="official_random_70_15_15"
    else:
        from gog_fraud.pipelines.run_sci_round4c import _split
        config=yaml.safe_load(CONFIG.read_text())
        val,test,_,_,policy=_split(data,dataset.removesuffix("-Syn"),seed,config)
    return val,test,y[val],y[test],policy


def run(dataset,seed):
    if "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"A05 AnomalyDAE large-graph run requires RTX 3090, got {torch.cuda.get_device_name(0)}")
    OUT.joinpath("raw").mkdir(parents=True,exist_ok=True)
    OUT.joinpath("scores").mkdir(parents=True,exist_ok=True)
    OUT.joinpath("checkpoints").mkdir(parents=True,exist_ok=True)
    path=OUT/"raw"/f"{dataset}__AnomalyDAE__seed{seed}.json"
    pre=json.loads((V2/f"diagnostics/anomalydae_chunked/preflight_{dataset}.json").read_text())
    if pre["status"]!="PREFLIGHT_FEASIBLE" or "RTX 3090" not in pre.get("gpu_name", ""):
        raise RuntimeError(f"A05 preflight does not authorize full run: {dataset} {pre['status']}")
    if path.is_file() and json.loads(path.read_text()).get("status")=="success":
        print(f"[SKIP] {dataset} seed{seed}",flush=True);return 0
    graph_path=GRAPH/f"{dataset}.pt"
    manifest=next(r for r in json.loads(MANIFEST.read_text()) if r["dataset_id"]==dataset)
    if filehash(graph_path)!=manifest["constructed_artifact_sha256"]:raise RuntimeError("graph artifact drift")
    payload=torch.load(graph_path,map_location="cpu",weights_only=False)
    data=Data(x=payload["x"],edge_index=payload["edge_index"],y=payload["y"],num_nodes=payload["num_nodes"])
    for mask in ("train_mask","val_mask","test_mask","eval_mask"):
        if mask in payload:setattr(data,mask,payload[mask])
    val,test,val_y,test_y,policy=split(data,dataset,seed)
    split_hash=digest({"val":arrayhash(val),"test":arrayhash(test),"val_y":arrayhash(val_y),"test_y":arrayhash(test_y)})
    config=yaml.safe_load(CONFIG.read_text())
    source=Path(__file__)
    record={"run_id":f"A05-{dataset}-AnomalyDAE-seed{seed}","dataset":dataset,"model":"AnomalyDAE",
            "seed":seed,"status":"running","configured_epochs":50,"block_size":256,
            "split_policy":policy,"split_hash":split_hash,"val_count":len(val),"test_count":len(test),
            "test_positives":int(test_y.sum()),"dataset_artifact_sha256":filehash(graph_path),
            "constructed_tensor_sha256":manifest["constructed_tensor_sha256"],
            "feature_hash":manifest["feature_hash"],"edge_hash":manifest["edge_hash"],
            "label_hash":manifest["label_hash"],"environment_lock_sha256":filehash(LOCK),
            "source_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "runner_source_sha256":filehash(source),"model_source_sha256":filehash(ROOT/"src/gog_fraud/models/pygod/shared_reconstruction.py"),
            "model_config":config,"model_config_hash":digest(config),"torch":torch.__version__,
            "cuda":torch.version.cuda,"device":torch.cuda.get_device_name(0),"full_shared_graph":True,
            "approximation_used":False}
    path.write_text(json.dumps(record,indent=2,default=str)+"\n")
    started=time.perf_counter()
    try:
        seed_everything(seed,deterministic=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
        detector=_instantiate(config,"AnomalyDAE",_models(config)["AnomalyDAE"],0)
        detector.training_checkpoint_path=OUT/"checkpoints"/f"{dataset}__seed{seed}.pt"
        t0=time.perf_counter();detector.fit(data);train_sec=time.perf_counter()-t0
        t0=time.perf_counter();raw=detector.decision_function(data);torch.cuda.synchronize(0)
        infer_sec=time.perf_counter()-t0
        score=raw.detach().cpu().numpy().reshape(-1) if torch.is_tensor(raw) else np.asarray(raw).reshape(-1)
        if score.size!=data.num_nodes or not np.isfinite(score).all():raise FloatingPointError("invalid scores")
        score_path=OUT/"scores"/f"{dataset}__AnomalyDAE__seed{seed}.npy"
        np.save(score_path,score)
        if dataset=="Ethereum":
            threshold=float(np.percentile(score[val],100*(1-float(np.mean(val_y)))))
            threshold_metrics={"threshold":threshold,
                               "validation_f1":float(f1_score(test_y,score[test]>=threshold,zero_division=0))}
        else:
            result=evaluate_threshold_protocol(val_y,score[val],test_y,score[test],fixed_05_applicable=False)
            threshold_metrics=result.to_dict()
        record.update({"status":"success","actual_epochs":int(detector.actual_epochs_),
                       "roc_auc":float(roc_auc_score(test_y,score[test])),
                       "pr_auc":float(average_precision_score(test_y,score[test])),
                       **threshold_metrics,"train_time_sec":train_sec,"inference_time_sec":infer_sec,
                       "peak_allocated_mib":torch.cuda.max_memory_allocated(0)/2**20,
                       "raw_score_path":str(score_path.relative_to(ROOT)),"raw_score_sha256":arrayhash(score),
                       "raw_score_file_sha256":filehash(score_path)})
    except BaseException as exc:
        record.update({"status":"UNSUPPORTED_RESOURCE_OOM" if isinstance(exc,torch.OutOfMemoryError) else "FAILED_OTHER",
                       "error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc(),
                       "actual_epochs":None})
    record["total_wall_sec"]=time.perf_counter()-started
    path.write_text(json.dumps(record,indent=2,default=str)+"\n")
    print(f"[DONE] {dataset} seed{seed} {record['status']} {record['total_wall_sec']:.1f}s",flush=True)
    return 0 if record["status"]=="success" else 2


def main():
    parser=argparse.ArgumentParser();parser.add_argument("dataset",choices=DATASETS)
    parser.add_argument("--seed",type=int,choices=range(42,47),default=42)
    args=parser.parse_args();return run(args.dataset,args.seed)


if __name__=="__main__":raise SystemExit(main())
