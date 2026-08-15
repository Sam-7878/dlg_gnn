"""Analyze Round 4C production results and make the final readiness decision."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from gog_fraud.experiments.round2_validity import graph_fingerprints
from gog_fraud.experiments.round4c_policy import (
    FAILURE_STATUSES, UNSUPPORTED_STATUSES, canonical_hash, final_support_matrix,
    readiness_decision, runtime_forecast,
)
from gog_fraud.pipelines.run_sci_round4c import _datasets, ensure_layout


ROUND5_DATASETS = [
    "Elliptic", "DGraphFin", "Yelp", "Amazon", "BitcoinOTC",
    "Flickr", "Reddit", "Cora", "CiteSeer", "PubMed",
]


def collect_results(root: Path) -> pd.DataFrame:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root/"raw").glob("*.json"))]
    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(["dataset","model","seed"]).any():
        raise ValueError("duplicate production cells detected")
    return frame


def build_round5_protocol(config: dict, support: pd.DataFrame, decision: str) -> dict | None:
    if decision not in {"READY_FOR_FULL_RUN","READY_WITH_RESTRICTIONS"}:
        return None
    supported = support.loc[support.primary_supported,["dataset","model"]]
    return {
        "benchmark": {
            "datasets": ROUND5_DATASETS,
            "models": list(config["models"]), "model_seeds": list(config["round5"]["model_seeds"]),
            "supported_representative_cells": supported.to_dict("records"),
            "historical_dlg_alias": {"DLG":"DLG-Aug"}, "dlg_fusion_primary": False,
        },
        "synthetic": {"fixed_dataset_seed":42,"robustness_appendix":True},
        "training": {"epochs":int(config["training"]["epochs"]),"early_stopping":False,
                     "dlg_l1_epochs":int(config["training"]["dlg_l1_epochs"])},
        "execution": {"fresh_subprocess_per_run":True,"full_shared_graph":True,
                      "message_backend":"sparse_fused","partition_fallback":False,"cpu_fallback":False},
        "reconstruction": {"linear_backend":"exact_sparse","anomalydae_backend":"chunked_exact"},
        "threshold": {"primary":"validation_selected","fixed_05":False,"oracle_best_f1":"retrospective_only"},
        "metrics": {"primary":["roc_auc","pr_auc","validation_f1"]},
        "statistics": {"aggregate_seed_first":True,"complete_case_blocks":True,
                       "friedman":True,"wilcoxon_holm":True,"unsupported_rank_imputation":False},
        "output_root":config["round5"]["output_root"], "auto_start":False,
    }


def _hash_tensor(value: torch.Tensor) -> str:
    array=value.detach().cpu().contiguous().numpy()
    h=hashlib.sha256(); h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


def build_data_freeze(config: dict, *, datasets: list[str] | None=None) -> dict:
    registry=_datasets(config); names=datasets or list(config["datasets"]); records=[]
    for name in names:
        data=registry[name](); fingerprints=graph_fingerprints(data,injection_config={"dataset_seed":42})
        labels=data.y.detach().cpu().numpy()
        eligible=labels[np.isin(labels,[0,1])]
        records.append({
            "dataset":name,"nodes":int(data.num_nodes),"edges":int(data.num_edges),
            "features":int(data.x.shape[1]),
            "anomaly_label_ratio":float((eligible==1).mean()) if len(eligible) else None,
            "label_provenance":("real labels" if name in {"Elliptic","DGraphFin","BitcoinOTC"}
                                else "synthetic injection"),
            "feature_hash":_hash_tensor(data.x),"label_hash":_hash_tensor(data.y),
            "edge_hash":_hash_tensor(data.edge_index),"injection_hash":fingerprints.get("injection_hash"),
        })
    try:
        commit=subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    except Exception:
        commit=None
    source_hash=hashlib.sha256()
    source_files=sorted(Path("src/gog_fraud").rglob("*.py"))
    for path in source_files:
        source_hash.update(str(path).encode());source_hash.update(path.read_bytes())
    try:
        dirty=subprocess.run(["git","status","--porcelain"],capture_output=True,text=True,
                             check=True,timeout=10).stdout.splitlines()
    except Exception:
        dirty=[]
    return {"fixed_dataset_seed":42,"datasets":records,"code_git_commit":commit,
            "benchmark_source_hash":source_hash.hexdigest(),
            "config_hash":canonical_hash(config),"backend_hash":canonical_hash(config.get("backend",{})),
            "git_dirty_paths":dirty,
            "freeze_hash":hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}


def build_environment_freeze() -> dict:
    versions={}
    for package in ("torch-geometric","pygod","torch-sparse"):
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    return {"python":platform.python_version(),"pytorch":torch.__version__,"cuda":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "os_wsl":platform.platform(),**versions}


def _component_summary(round4b_root: Path) -> tuple[pd.DataFrame,pd.DataFrame]:
    path=round4b_root/"ablation/component_raw.csv"
    if not path.exists(): return pd.DataFrame(),pd.DataFrame()
    raw=pd.read_csv(path); summary=[]
    for dataset,group in raw.groupby("dataset"):
        mean=group.groupby("variant")[["pr_auc","validation_f1"]].mean()
        def get(v,m): return float(mean.loc[v,m]) if v in mean.index else np.nan
        row={"dataset":dataset,
             "delta_aug_pr_auc":get("DLG-Aug","pr_auc")-get("DLG-Base","pr_auc"),
             "delta_fusion_pr_auc":get("DLG-Fusion","pr_auc")-get("DLG-Aug","pr_auc"),
             "delta_aug_validation_f1":get("DLG-Aug","validation_f1")-get("DLG-Base","validation_f1"),
             "delta_fusion_validation_f1":get("DLG-Fusion","validation_f1")-get("DLG-Aug","validation_f1"),
             "best_variant_pr":mean.pr_auc.idxmax(),"best_variant_f1":mean.validation_f1.idxmax()}
        local_pr = get("DLG-Local", "pr_auc")
        fusion_pr = get("DLG-Fusion", "pr_auc")
        base_pr = get("DLG-Base", "pr_auc")
        aug_pr = get("DLG-Aug", "pr_auc")
        if local_pr > max(base_pr, aug_pr) + .02 and local_pr >= fusion_pr:
            row["descriptive_class"]="local-dominant"
            row["classification_basis"] = "local PR-AUC is best and exceeds both global/augmented variants by >0.02"
        elif fusion_pr > max(base_pr, aug_pr, local_pr) + .005:
            row["descriptive_class"]="fusion-beneficial"
            row["classification_basis"] = "fusion PR-AUC exceeds every component variant by >0.005"
        elif row["delta_aug_pr_auc"]>.005:
            row["descriptive_class"]="augmentation-beneficial"
            row["classification_basis"] = "augmentation PR-AUC gain over global/base exceeds 0.005"
        else:
            row["descriptive_class"]="neutral/mixed"
            row["classification_basis"] = "no predeclared descriptive PR-AUC delta threshold was met"
        summary.append(row)
    return raw,pd.DataFrame(summary)


def _save_figures(output: Path, success: pd.DataFrame, support: pd.DataFrame,
                  anomaly: pd.DataFrame, component_raw: pd.DataFrame):
    def placeholder(name,title):
        fig,ax=plt.subplots(figsize=(7,3));ax.axis("off");ax.text(.5,.5,"insufficient completed evidence",ha="center");ax.set_title(title)
        fig.tight_layout();fig.savefig(output/"figures"/name,dpi=180);plt.close(fig)
    for column,name,title,ylabel in (
        ("total_wall_sec","01_production_runtime_by_model.png","Production runtime by model","seconds"),
        ("total_wall_sec","02_production_runtime_by_dataset.png","Production runtime by dataset","seconds"),
        ("nvidia_smi_peak_mb","03_peak_gpu_memory.png","Observed GPU memory","MiB"),
        ("rss_peak_mb","04_peak_rss_memory.png","Process RSS","MiB"),
    ):
        if success.empty or column not in success: placeholder(name,title);continue
        axis="model" if "model" in name else "display_name"
        ax=success.boxplot(column=column,by=axis,figsize=(9,4),rot=45);ax.set_title(title);ax.set_ylabel(ylabel);ax.figure.suptitle("")
        ax.figure.tight_layout();ax.figure.savefig(output/"figures"/name,dpi=180);plt.close(ax.figure)
    if anomaly.empty: placeholder("05_anomalydae_estimated_vs_actual.png","AnomalyDAE estimated vs actual")
    else:
        ax=anomaly.set_index("dataset")[["estimated_gpu_hours","actual_gpu_hours"]].plot.bar(figsize=(8,4));ax.set_ylabel("GPU hours")
        ax.figure.tight_layout();ax.figure.savefig(output/"figures/05_anomalydae_estimated_vs_actual.png",dpi=180);plt.close(ax.figure)
    matrix=support.pivot(index="dataset",columns="model",values="primary_supported") if not support.empty else pd.DataFrame()
    if matrix.empty: placeholder("06_model_dataset_support_matrix.png","Production support matrix")
    else:
        # During a resumable production run, unattempted model/dataset cells are
        # absent from the support table and therefore appear as NaN after the
        # pivot.  Fail closed in the live heatmap instead of crashing before
        # the nominal matrix is complete.
        matrix = matrix.fillna(False).astype(bool)
        fig,ax=plt.subplots(figsize=(9,4));ax.imshow(matrix.astype(int),vmin=0,vmax=1,cmap="RdYlGn",aspect="auto")
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45,ha="right");ax.set_yticks(range(len(matrix.index)),matrix.index)
        fig.tight_layout();fig.savefig(output/"figures/06_model_dataset_support_matrix.png",dpi=180);plt.close(fig)
    for metric,name,title in (("pr_auc","07_dlg_component_pr_auc_summary.png","DLG component PR-AUC"),
                              ("validation_f1","08_dlg_component_f1_summary.png","DLG component validation F1")):
        if component_raw.empty: placeholder(name,title);continue
        plot=component_raw.groupby(["dataset","variant"])[metric].mean().unstack()
        ax=plot.plot.bar(figsize=(9,4));ax.set_title(title);ax.figure.tight_layout();ax.figure.savefig(output/"figures"/name,dpi=180);plt.close(ax.figure)
    for metric,name,title in (
        ("pr_auc","09_representative_pr_auc_heatmap.png","Representative mean PR-AUC"),
        ("validation_f1","10_representative_validation_f1_heatmap.png","Representative mean validation F1"),
    ):
        if success.empty or metric not in success: placeholder(name,title);continue
        matrix=success.pivot_table(index="display_name",columns="model",values=metric,aggfunc="mean")
        fig,ax=plt.subplots(figsize=(10,4));image=ax.imshow(matrix.to_numpy(),aspect="auto",cmap="viridis")
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=45,ha="right")
        ax.set_yticks(range(len(matrix.index)),matrix.index);ax.set_title(title);fig.colorbar(image,ax=ax)
        fig.tight_layout();fig.savefig(output/"figures"/name,dpi=180);plt.close(fig)


def _storage_estimate(output: Path, *, round5_cells: int) -> dict:
    raw_files=list((output/"raw").glob("*.json"));log_files=list((output/"logs").glob("*.log"))
    raw_bytes=sum(path.stat().st_size for path in raw_files)
    log_bytes=sum(path.stat().st_size for path in log_files)
    observed=max(len(raw_files),1)
    return {
        "observed_raw_files":len(raw_files),"observed_raw_bytes":raw_bytes,
        "observed_log_files":len(log_files),"observed_log_bytes":log_bytes,
        "round5_nominal_cells":int(round5_cells),
        "projected_raw_bytes":int(raw_bytes/observed*round5_cells),
        "projected_log_bytes":int(log_bytes/observed*round5_cells),
        "projected_manifest_count":int(round5_cells),
        "figure_table_outputs_note":"small relative to checkpoints and per-cell logs",
        "checkpoint_note":"exclude checkpoints from persistent paper artifact after verified completion; AnomalyDAE checkpoint can approach 1 GiB per active cell",
    }


def analyze(config: dict, output: Path, round4b_root: Path) -> dict:
    ensure_layout(output); frame=collect_results(output)
    frame.to_csv(output/"tables/production_representative_raw.csv",index=False)
    expected=[(d,m,int(s)) for d in config["datasets"] for m in config["models"] for s in config["seeds"]]
    decision,reasons=readiness_decision(frame,expected)
    support=final_support_matrix(frame) if not frame.empty else pd.DataFrame()
    support.to_csv(output/"tables/model_dataset_support_matrix_final.csv",index=False)
    success=frame.loc[frame.status.eq("success")].copy() if not frame.empty else pd.DataFrame()
    distributions=(success.groupby(["dataset","model"]).total_wall_sec.agg(["min",lambda x:x.quantile(.25),"median",lambda x:x.quantile(.75),"max"]).reset_index()
                   if not success.empty else pd.DataFrame())
    if not distributions.empty: distributions.columns=["dataset","model","min","p25","median","p75","max"]
    distributions.to_csv(output/"tables/production_runtime_distribution.csv",index=False)
    supported_pairs=set(map(tuple,support.loc[support.primary_supported,["dataset","model"]].to_numpy())) if not support.empty else set()
    forecast_frame=success.loc[[ (d,m) in supported_pairs for d,m in zip(success.dataset,success.model) ]] if not success.empty else success
    try: forecast=runtime_forecast(forecast_frame,round5_seeds=len(config["round5"]["model_seeds"]))
    except ValueError as exc:
        forecast={"status":"unavailable","reason":str(exc)}
        if "runtime estimate unavailable" not in reasons: reasons.append("runtime estimate unavailable")
        decision="NOT_READY"
    forecast["measured_dataset_count"]=len(config["datasets"])
    forecast["round5_dataset_count"]=len(ROUND5_DATASETS)
    forecast["coverage_note"]="measured representative subtotal; unmeasured Round 5 datasets require extrapolation and are not silently treated as zero cost"
    (output/"resources/round5_runtime_forecast.json").write_text(json.dumps(forecast,indent=2)+"\n",encoding="utf-8")
    estimate_path=round4b_root/"anomalydae/feasibility.csv"; estimate=pd.read_csv(estimate_path) if estimate_path.exists() else pd.DataFrame()
    anomaly_cells=frame.loc[frame.model.eq("AnomalyDAE")].copy() if not frame.empty else pd.DataFrame()
    if not anomaly_cells.empty:
        actual=(anomaly_cells.groupby("display_name",as_index=False)
                .agg(observed_wall_sec=("total_wall_sec","mean"),
                     measured_seeds=("seed","nunique"),
                     statuses=("status",lambda values:";".join(sorted(set(values))))))
        actual["actual_censored_or_failed"]=~actual.statuses.eq("success")
        # A fast OOM is not an estimate of the production runtime.  A
        # predeclared operational timeout, however, is a scientifically useful
        # right-censored lower bound and remains explicitly flagged as such.
        actual["actual_sec"] = actual["observed_wall_sec"].where(
            actual.statuses.eq("success") | actual.statuses.str.contains("unsupported_operational")
        )
    else:
        actual=pd.DataFrame()
    if not estimate.empty:
        anomaly=estimate[["dataset","estimated_gpu_hours"]].merge(actual.rename(columns={"display_name":"dataset"}),how="left",on="dataset")
        anomaly["observed_wall_hours"]=anomaly.observed_wall_sec/3600
        anomaly["actual_gpu_hours"]=anomaly.actual_sec/3600
        anomaly["ratio_actual_to_estimated"]=anomaly.actual_gpu_hours/anomaly.estimated_gpu_hours
    else: anomaly=pd.DataFrame()
    anomaly.to_csv(output/"tables/anomalydae_estimated_vs_actual.csv",index=False)
    component_raw,component_summary=_component_summary(round4b_root)
    component_summary.to_csv(output/"tables/dlg_component_scientific_summary.csv",index=False)
    current_dlg=success.loc[success.model.isin(["DLG-Base","DLG-Aug"]),["dataset","seed","model","pr_auc","validation_f1"]]
    prior=component_raw.loc[component_raw.variant.isin(["DLG-Base","DLG-Aug"]),["dataset","seed","variant","pr_auc","validation_f1"]] if not component_raw.empty else pd.DataFrame()
    consistency=current_dlg.merge(prior,left_on=["dataset","seed","model"],right_on=["dataset","seed","variant"],how="outer",suffixes=("_r4c","_r4b")) if not prior.empty else current_dlg
    if not prior.empty:
        consistency["pr_auc_abs_diff"]=(consistency.pr_auc_r4c-consistency.pr_auc_r4b).abs()
        consistency["validation_f1_abs_diff"]=(consistency.validation_f1_r4c-consistency.validation_f1_r4b).abs()
        consistency["consistency_tolerance"]=.02
        consistency["metric_semantics_consistent"]=(
            consistency.pr_auc_abs_diff.le(.02)
            & consistency.validation_f1_abs_diff.le(.02)
        )
    consistency.to_csv(output/"tables/dlg_component_consistency.csv",index=False)
    complexity={
        "AnomalyDAE":"nonlinear all-pairs O(N^2)",
        "DOMINANT":"sparse message passing + exact linear Gram reconstruction",
        "CONAD":"sparse message passing + exact linear Gram reconstruction",
        "DLG-Base":"sparse message passing + exact linear Gram reconstruction",
        "DLG-Aug":"local pretrain + sparse message passing + exact linear Gram reconstruction",
        "CoLA":"native objective with exact sparse-fused GCN aggregation",
        "GADNR":"native neighborhood-distribution objective; residual COO SAGE aggregation",
        "OCGNN":"native objective with exact sparse-fused GCN aggregation",
    }
    scale=frame.merge(support[["dataset","model","primary_supported","restriction"]],on=["dataset","model"],how="left")
    scale["algorithmic_complexity_class"]=scale.model.map(complexity)
    scale.rename(columns={"primary_supported":"exact_supported","nvidia_smi_peak_mb":"gpu_memory_nvidia_smi_mb"}).to_csv(
        output/"tables/scalability_support.csv",index=False
    )
    storage=_storage_estimate(output,round5_cells=len(ROUND5_DATASETS)*len(config["models"])*len(config["round5"]["model_seeds"]))
    (output/"resources/round5_storage_estimate.json").write_text(json.dumps(storage,indent=2)+"\n",encoding="utf-8")
    _save_figures(output,success,support,anomaly,component_raw)
    protocol=build_round5_protocol(config,support,decision)
    protocol_path=output/"manifests/round5_full_benchmark_protocol.yaml"
    if protocol is not None: protocol_path.write_text(yaml.safe_dump(protocol,sort_keys=False),encoding="utf-8")
    elif protocol_path.exists(): protocol_path.unlink()
    environment=build_environment_freeze();(output/"freeze/environment_freeze.json").write_text(json.dumps(environment,indent=2)+"\n",encoding="utf-8")
    gate={"decision":decision,"reasons":reasons,"nominal_cells":len(expected),"attempted_cells":len(frame),
          "success_cells":int(frame.status.eq("success").sum()) if not frame.empty else 0,
          "unsupported_cells":int(frame.status.isin(UNSUPPORTED_STATUSES).sum()) if not frame.empty else 0,
          "unexpected_failure_cells":int(frame.status.isin(FAILURE_STATUSES).sum()) if not frame.empty else 0,
          "runtime_forecast_available":forecast.get("status")!="unavailable","round5_protocol_created":protocol is not None,
          "round5_auto_started":False}
    (output/"manifests/readiness_gate.json").write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    return gate


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--stage",choices=("analyze","freeze"),default="analyze")
    p.add_argument("--round4b-root",default="outputs/sci_round4b");args=p.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));output=Path(config["experiment"]["output_root"]);ensure_layout(output)
    if args.stage=="freeze":
        freeze=build_data_freeze(config,datasets=ROUND5_DATASETS);(output/"freeze/round5_data_and_code_freeze.json").write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf-8")
    else: print(json.dumps(analyze(config,output,Path(args.round4b_root)),indent=2))
    return 0


if __name__=="__main__":raise SystemExit(main())
