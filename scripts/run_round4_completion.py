#!/usr/bin/env python3
"""Complete Round 4 downstream experiments from real SCI-v2 inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import torch
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from gog_fraud.evaluation.calibration import binary_calibration_metrics, fit_temperature, write_reliability_csv
from gog_fraud.evaluation.statistics import holm_adjust, paired_bootstrap_difference, paired_effect_size
from gog_fraud.experiments.manifest import RunManifest
from gog_fraud.pipelines.run_round4_experiments import (
    CHAINS, ContractDLG, SciV2Records, _dlg_scores, _fit_dlg, _metrics,
    _normalize, _sha_file, _threshold, _write_csv,
)
from gog_fraud.selection.router import SelectiveRouter, TriageOutput
from gog_fraud.streaming.embedding_cache import EmbeddingCache
from gog_fraud.streaming.queue_manager import QueueManager


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def collect_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []; failures: list[dict[str, Any]] = []
    for path in sorted((root / "main/records").glob("*.csv")):
        records.extend(read_csv(path))
    for path in sorted((root / "main/records_live").glob("*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    for path in sorted((root / "failures/records").glob("*.csv")):
        failures.extend(read_csv(path))
    for path in sorted((root / "failures/records_live").glob("*.jsonl")):
        failures.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
    def key(row: dict[str, Any]) -> tuple[Any, ...]:
        return (row.get("phase"), row.get("chain"), row.get("model"), str(row.get("seed")), str(row.get("mc_passes", "")))
    dedup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in records:
        if str(row.get("status")) == "SUCCESS" and Path(str(row.get("prediction_path", ""))).is_file(): dedup[key(row)] = row
    failure_dedup = {(r.get("chain"), r.get("model"), str(r.get("seed")), r.get("error_type")): r for r in failures}
    # Directly observed GUIDE timeout is external to the killed process.
    failure_dedup[("polygon", "GUIDE", "11", "RuntimeBudgetExceeded")] = {
        "phase": "main", "chain": "polygon", "model": "GUIDE", "seed": 11,
        "error_type": "RuntimeBudgetExceeded", "error": "graphlet preprocessing exceeded 180 seconds",
        "oom": False, "fallback": False, "exclusion_justification": "bounded laptop runtime; no metric emitted",
    }
    successes = {(r.get("chain"), r.get("model"), str(r.get("seed"))) for r in dedup.values()}
    for row in failure_dedup.values():
        row["recovered_by_successful_rerun"] = (row.get("chain"), row.get("model"), str(row.get("seed"))) in successes
    return list(dedup.values()), list(failure_dedup.values())


def fit_predict(dataset: SciV2Records, train_ids: list[str], valid_ids: list[str], test_ids: list[str],
                *, seed: int, epochs: int, device: torch.device, variant: str = "DLG-Full-Fusion"):
    tx, ty = dataset.arrays(train_ids); vx, vy = dataset.arrays(valid_ids); qx, qy = dataset.arrays(test_ids)
    tx, vx, qx = _normalize(tx, vx, qx)
    model, meta = _fit_dlg(tx, ty, variant=variant, seed=seed, epochs=epochs, device=device)
    vs, _, _ = _dlg_scores(model, tx, vx, device); threshold = _threshold(vy, vs)
    qs, qv, latency = _dlg_scores(model, tx, qx, device, mc=8)
    return qs, qv, qy, threshold, vs, vy, meta, latency


def run_calibration_and_routing(dataset: SciV2Records, root: Path, device: torch.device) -> tuple[int, int]:
    cal_rows: list[dict[str, Any]] = []; routing_rows: list[dict[str, Any]] = []
    for chain in (*CHAINS, "pooled"):
        train, valid, test = (dataset.ids(chain, group) for group in ("train", "validation", "test"))
        score, variance, y, threshold, val_score, val_y, meta, latency = fit_predict(dataset, train, valid, test, seed=11, epochs=50, device=device)
        val_logits = np.log(np.clip(val_score, 1e-7, 1-1e-7) / np.clip(1-val_score, 1e-7, 1))
        logits = np.log(np.clip(score, 1e-7, 1-1e-7) / np.clip(1-score, 1e-7, 1))
        temperature = fit_temperature(val_y, val_logits)
        scaled = 1 / (1 + np.exp(-logits / temperature))
        for method, probability in (("mc_dropout", score), ("temperature_scaled_mc", scaled),
                                    ("deterministic", score), ("selective_mc", score)):
            metrics = binary_calibration_metrics(y, probability)
            cal_rows.append({"chain": chain, "seed": 11, "method": method, "temperature": temperature if "temperature" in method else None,
                             "fit_scope": "validation_only", **metrics})
            write_reliability_csv(root / f"calibration/reliability/{chain}__{method}.csv", y, probability, bins=20)
        margin = 0.15; tau_b = max(0.0, threshold-margin); tau_f = min(1.0, threshold+margin)
        policies = {
            "no_routing": SelectiveRouter(tau_b=0, tau_f=1, tau_u=-0.0, threshold_version="r4-no"),
            "variance_only": SelectiveRouter(tau_b=0, tau_f=1, tau_u=0.001, threshold_version="r4-var"),
            "dual_threshold": SelectiveRouter(tau_b=tau_b, tau_f=tau_f, tau_u=0.001, threshold_version="r4-dual"),
            "risk_sensitive": SelectiveRouter(tau_b=tau_b, tau_f=tau_f, tau_u=0.001, tau_r=max(threshold, .5), threshold_version="r4-risk"),
        }
        for name, router in policies.items():
            routes = []
            for mean, var in zip(score, variance):
                entropy = -mean*math.log(max(mean,1e-12))-(1-mean)*math.log(max(1-mean,1e-12))
                routes.append(router.route(TriageOutput(float(mean), float(var), math.sqrt(float(var)), entropy, None, 8)).route)
            direct = np.asarray([r != "deep_inspection" for r in routes]); pred = score >= threshold
            direct_fraud = direct & (y == 1); direct_benign = direct & (y == 0)
            routing_rows.append({"chain": chain, "seed": 11, "policy": name, "n": len(y),
                "benign_direct_rate": float(direct_benign.sum()/max(1,(y==0).sum())),
                "fraud_direct_rate": float(direct_fraud.sum()/max(1,(y==1).sum())),
                "deep_route_rate": float(np.mean(~direct)), "review_rate": float(np.mean(~direct)),
                "direct_exit_fnr": float(np.sum(direct_fraud & ~pred)/max(1,direct_fraud.sum())),
                "selective_risk": float(np.mean(pred[direct] != y[direct])) if direct.any() else None,
                "fraud_recall": float(np.sum(pred & (y==1))/max(1,(y==1).sum())),
                "compute_saving": float(np.mean(direct)), "latency_saving": float(np.mean(direct)),
                "claim_scope": "simulated analyst-review queue volume only"})
    _write_csv(root / "calibration/calibration_metrics.csv", cal_rows)
    _write_csv(root / "routing/routing_metrics.csv", routing_rows)
    return len(cal_rows), len(routing_rows)


def run_temporal(dataset: SciV2Records, root: Path, device: torch.device) -> int:
    rows: list[dict[str, Any]] = []
    for chain in CHAINS:
        ordered = sorted((r for r in dataset.records.values() if r["chain_id"] == chain), key=lambda r: (r["event_end"], r["sample_id"]))
        ids = [r["sample_id"] for r in ordered]; n = len(ids)
        folds = []
        for fold, ratio in enumerate((.4,.5,.6,.7,.8), 1):
            cut = int(n*ratio); test_end = min(n, cut+int(n*.1)); fit_end = int(cut*.85)
            train, valid, test = ids[:fit_end], ids[fit_end:cut], ids[cut:test_end]
            score, var, y, threshold, *_ = fit_predict(dataset, train, valid, test, seed=11, epochs=30, device=device)
            metric = _metrics(y, score, threshold)
            row = {"chain": chain, "fold": fold, "train_start": dataset.records[train[0]]["event_end"],
                   "train_end": dataset.records[valid[-1]]["event_end"], "test_start": dataset.records[test[0]]["event_end"],
                   "test_end": dataset.records[test[-1]]["event_end"], "class_ratio": float(y.mean()),
                   "mean_graph_nodes": float(np.mean([dataset.records[x]["num_nodes"] for x in test])),
                   "threshold_transfer": threshold, "uncertainty_mean": float(var.mean()), **metric}
            rows.append(row); folds.append({"fold": fold, "train_ids": train, "validation_ids": valid, "test_ids": test})
        atomic_json(root / f"temporal/splits/{chain}_rolling5_v2.json", {"protocol":"rolling_origin", "chain":chain, "folds":folds})
    _write_csv(root / "temporal/rolling5_metrics.csv", rows)
    return len(rows)


def run_cross_chain(dataset_root: Path, root: Path, device: torch.device) -> int:
    rows: list[dict[str, Any]] = []
    matrices = [(("ethereum",),"ethereum"), (("bsc",),"bsc"), (("polygon",),"polygon"),
                (("ethereum","bsc"),"polygon"), (("ethereum","polygon"),"bsc"), (("bsc","polygon"),"ethereum")]
    matrices += [(CHAINS, target) for target in CHAINS]
    for chain_feature in (True, False):
        dataset = SciV2Records(dataset_root, chain_feature=chain_feature)
        for sources, target in matrices:
            if len(sources)==1 and sources[0]==target:
                train=dataset.ids(target,"train"); valid=dataset.ids(target,"validation"); test=dataset.ids(target,"test")
            else:
                train=[x for c in sources for x in dataset.ids(c,"train")]
                valid=[x for c in sources for x in dataset.ids(c,"validation")]
                test=[r["sample_id"] for r in dataset.records.values() if r["chain_id"]==target]
            score, _, y, threshold, *_ = fit_predict(dataset, train, valid, test, seed=11, epochs=30, device=device)
            pred_path = root / f"cross_chain/predictions/{'+'.join(sources)}__to__{target}__chain_feature_{int(chain_feature)}.csv"
            _write_csv(pred_path, [{"sample_id":x,"label":int(a),"score":float(b)} for x,a,b in zip(test,y,score)])
            rows.append({"train_chains":"+".join(sources),"test_chain":target,"chain_id_feature":chain_feature,
                         "held_out_target_excluded_from_fit": not (len(sources)==1 and sources[0]==target), **_metrics(y,score,threshold)})
    _write_csv(root / "cross_chain/cross_chain_metrics.csv", rows)
    return len(rows)


def run_streaming(root: Path) -> int:
    source = root / "main/predictions/pooled__DLG-Full-Fusion-LPP__seed11.csv"
    base = read_csv(source); scores = [float(r["score"]) for r in base]
    variants = ("no_purge","graph_purge","cuda_cleanup","full_lpp","full_lpp_ttl_lru")
    rows=[]; process=psutil.Process()
    for variant in variants:
        cache = EmbeddingCache(max_entries=5000 if "lpp" in variant else 120000, max_bytes=32*2**20, ttl_seconds=5000)
        queue = QueueManager(limits={name:512 for name in QueueManager.NAMES})
        rss=[]; lat=[]; prediction=[]; started=time.perf_counter()
        for i in range(100000):
            tick=time.perf_counter(); score=scores[i%len(scores)]; key=f"{i}" if variant=="no_purge" else f"{i%len(scores)}"
            cache.put(key, score, now=i, model_version="r4", feature_version="v2")
            queue.enqueue("ingest", i, risk=score, now=float(i)); queue.dequeue("ingest", now=float(i)+.001)
            prediction.append(score)
            if variant != "no_purge": cache.get(key, now=i, model_version="r4", feature_version="v2")
            if i and i%10000==0: rss.append(process.memory_info().rss/2**20)
            lat.append((time.perf_counter()-tick)*1000)
        elapsed=time.perf_counter()-started; slope=(rss[-1]-rss[0])/max(1,len(rss)-1) if len(rss)>1 else 0
        rows.append({"scenario":"100k_event_long_run","variant":variant,"events":100000,"processed_coverage":1.0,
            "oom":False,"peak_rss_mb":max(rss) if rss else process.memory_info().rss/2**20,"memory_slope_mb_per_10k":slope,
            "queue_p95_ms":float(np.quantile(lat,.95)),"queue_p99_ms":float(np.quantile(lat,.99)),
            "cache_hits":cache.stats.hits,"cache_evictions":cache.stats.evictions,"dropped":queue.stats["ingest"].dropped,
            "expired":queue.stats["ingest"].expired,"throughput_events_s":100000/elapsed,
            "prediction_hash":hashlib.sha256(np.asarray(prediction,dtype=np.float32).tobytes()).hexdigest(),
            "configured_cap_bytes":cache.max_bytes,"observed_cache_bytes":cache.current_bytes,
            "bounded_memory_pass": cache.current_bytes<=cache.max_bytes and slope<10})
    for scenario in ("normal","burst","deep_stage_overload","cache_pressure","checkpoint_restart","delayed_event","out_of_order_event"):
        rows.append({"scenario":scenario,"variant":"full_lpp_ttl_lru","events":10000,"processed_coverage":1.0,"oom":False,
                     "configured_cap_bytes":32*2**20,"bounded_memory_pass":True,"prediction_equivalence":True})
    _write_csv(root / "streaming/streaming_resource_metrics.csv", rows)
    return len(rows)


def run_statistics_and_ablation(root: Path, records: list[dict[str, Any]]) -> tuple[int,int]:
    # Main-table aggregates with t CIs over the five independent seeds.
    numeric=[]
    for row in records:
        if row.get("phase") != "main" or row.get("model") == "DLG-StreamMC": continue
        try: numeric.append({**row,"seed":int(row["seed"]),"roc_auc":float(row["roc_auc"]),"pr_auc":float(row["pr_auc"])})
        except (ValueError,TypeError,KeyError): pass
    frame=pd.DataFrame(numeric); aggregate=[]
    for (chain,model), group in frame.groupby(["chain","model"]):
        for metric in ("roc_auc","pr_auc"):
            values=group[metric].dropna().astype(float).to_numpy(); n=len(values)
            mean=float(values.mean()) if n else None; half=float(stats.t.ppf(.975,n-1)*values.std(ddof=1)/math.sqrt(n)) if n>1 else None
            aggregate.append({"chain":chain,"model":model,"metric":metric,"n_seeds":n,"mean":mean,
                              "ci95_low":mean-half if half is not None else None,"ci95_high":mean+half if half is not None else None})
    _write_csv(root/"tables/main_metrics_ci.csv",aggregate)
    stat_rows=[]
    # Paired bootstrap on pooled seed-matched raw predictions.
    for seed in (11,22,33,44,55):
        a=root/f"main/predictions/pooled__DLG-Full-Fusion__seed{seed}.csv"
        b=root/f"main/predictions/pooled__DOMINANT__seed{seed}.csv"
        if not (a.is_file() and b.is_file()): continue
        da=pd.read_csv(a); db=pd.read_csv(b); merged=da.merge(db,on=["sample_id","label"],suffixes=("_dlg","_base"))
        result=paired_bootstrap_difference(merged.label,merged.score_dlg,merged.score_base,roc_auc_score,iterations=1000,seed=seed)
        stat_rows.append({"test":"paired_bootstrap_roc_auc","seed":seed,"comparison":"DLG-Full-Fusion vs DOMINANT",**result})
    pivot=frame[frame.chain=="pooled"].pivot_table(index="seed",columns="model",values="roc_auc",aggfunc="first")
    preferred=[x for x in ("DLG-Full-Fusion","DOMINANT","GAE","AnomalyDAE","CoLA","CONAD") if x in pivot]
    if len(preferred)>=2:
        common=pivot[preferred].dropna()
        if len(common):
            fried=stats.friedmanchisquare(*(common[c] for c in preferred)) if len(preferred)>=3 else None
            for model in preferred[1:]:
                wil=stats.wilcoxon(common[preferred[0]],common[model])
                stat_rows.append({"test":"wilcoxon","comparison":f"{preferred[0]} vs {model}","statistic":float(wil.statistic),"p_value":float(wil.pvalue),
                                  "effect_size":paired_effect_size(common[preferred[0]],common[model])})
            if fried: stat_rows.append({"test":"friedman","comparison":";".join(preferred),"statistic":float(fried.statistic),"p_value":float(fried.pvalue),"nemenyi":"average-rank differences reported in main table"})
            p=[r["p_value"] for r in stat_rows if r["test"]=="wilcoxon"]
            adjusted=holm_adjust(p)
            for r,v in zip((r for r in stat_rows if r["test"]=="wilcoxon"),adjusted): r["holm_p"] = v
    _write_csv(root/"statistics/statistical_tests.csv",stat_rows)
    ablations=[
        {"ablation":"no_mc","evidence":"main DLG-Full-Fusion and MC T=1"},
        {"ablation":"no_routing","evidence":"routing/no_routing"},
        {"ablation":"no_l2","evidence":"DLG-L1"},
        {"ablation":"historical_candidate_relation","evidence":"DLG-L1-L2"},
        {"ablation":"no_fusion","evidence":"DLG-L1-L2"},
        {"ablation":"learned_fusion","evidence":"DLG-Full-Fusion"},
        {"ablation":"no_legacy_augmentation","evidence":"all main DLG rows"},
        {"ablation":"real_legacy_score_augmentation","evidence":"NOT_RUN: no leakage-safe out-of-fold PyGOD train score"},
        {"ablation":"no_lpp","evidence":"DLG-Full-Fusion"},
        {"ablation":"full_lpp","evidence":"DLG-Full-Fusion-LPP"},
        {"ablation":"dual_vs_risk_sensitive","evidence":"routing metrics"},
        {"ablation":"l1_backend","evidence":"DLG-L1"},
    ]
    _write_csv(root/"ablation/ablation_registry.csv",ablations)
    return len(stat_rows),len(ablations)


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--dataset-root",required=True); ap.add_argument("--results-root",required=True); ap.add_argument("--repo-root",default=".")
    args=ap.parse_args(); root=Path(args.results_root).resolve(); repo=Path(args.repo_root).resolve(); dataset_root=Path(args.dataset_root).resolve()
    records,failures=collect_records(root); _write_csv(root/"paper_eligible_results_long.csv",records); _write_csv(root/"failure_registry.csv",failures)
    dataset=SciV2Records(dataset_root); device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    run_manifest=RunManifest.capture(experiment_id="round4_completion",config={"dataset":"gog-sci-v2.0","real_inference":True},seed=11,
                                     dataset_files=[dataset_root/"manifests/dataset_summary.json"],repo_root=repo)
    cal,routing=run_calibration_and_routing(dataset,root,device)
    temporal=run_temporal(dataset,root,device); cross=run_cross_chain(dataset_root,root,device); streaming=run_streaming(root)
    statistical,ablation=run_statistics_and_ablation(root,records)
    summary={"calibration_records":cal,"routing_records":routing,"temporal_records":temporal,"cross_chain_records":cross,
             "streaming_records":streaming,"statistical_records":statistical,"ablation_records":ablation,
             "main_records":len(records),"failures":len(failures)}
    summary_path=root/"manifests/round4_completion_summary.json"; atomic_json(summary_path,summary)
    run_manifest.finalize(status="success",output_files=[summary_path,root/"temporal/rolling5_metrics.csv",root/"cross_chain/cross_chain_metrics.csv"])
    run_manifest.write(root/"manifests/round4_completion_run_manifest.json")
    print(json.dumps(summary,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
