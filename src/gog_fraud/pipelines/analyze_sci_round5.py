"""Freeze and analyze the final support-aware five-seed SCI benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from gog_fraud.evaluation.fraud_topology import compute_fraud_topology_metrics
from gog_fraud.evaluation.statistics import friedman_dataset_test, paired_model_tests, spearman_with_bootstrap
from gog_fraud.experiments.round5_policy import (
    complete_case_views, seed_first_summary, sha256_file, validate_final_raw, validate_support_matrix,
)
from gog_fraud.pipelines.analyze_sci_round4c import _hash_tensor, build_data_freeze, build_environment_freeze
from gog_fraud.pipelines.run_sci_round4c import _datasets, benchmark_execution_source_hash, hashes


METRICS = ["roc_auc", "pr_auc", "validation_f1", "mcc", "balanced_accuracy"]
PRIMARY = ["roc_auc", "pr_auc", "validation_f1"]
FRAUD_DATASETS = ["Elliptic", "DGraphFin", "Yelp", "Amazon", "BitcoinOTC", "Flickr", "Reddit"]


def _collect_json(output: Path) -> pd.DataFrame:
    rows=[]
    for path in sorted((output/"raw").glob("*.json")):
        row=json.loads(path.read_text(encoding="utf-8"));row["evidence_path"]=str(path.relative_to(output))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_hash(path: Path) -> str:
    digest=sha256_file(path);path.with_suffix(path.suffix+".sha256").write_text(digest+"\n",encoding="ascii")
    return digest


def _topology(config: dict) -> tuple[pd.DataFrame, list[dict]]:
    registry=_datasets(config);rows=[];fingerprints=[]
    for name in config["datasets"]:
        data=registry[name]();metrics=compute_fraud_topology_metrics(data.edge_index,data.y,directed=True)
        rows.append({"dataset":name,"display_name":config["display_names"][name],**metrics.to_dict()})
        fingerprints.append({"dataset":name,"edge_hash":_hash_tensor(data.edge_index),"label_hash":_hash_tensor(data.y),
                             "nodes":int(data.num_nodes),"edges":int(data.num_edges)})
    return pd.DataFrame(rows),fingerprints


def _dataset_table(config: dict, freeze: dict) -> pd.DataFrame:
    split={name:("official random 70/15/15" if name=="DGraphFin" else "stratified node-transductive") for name in config["datasets"]}
    return pd.DataFrame([{"dataset":row["dataset"],"display_name":config["display_names"][row["dataset"]],
                          "N":row["nodes"],"E":row["edges"],"F":row["features"],
                          "positive_ratio":row["anomaly_label_ratio"],"label_provenance":row["label_provenance"],
                          "split_type":split[row["dataset"]]} for row in freeze["datasets"]])


def _latex(frame: pd.DataFrame, path: Path, raw_hash: str, analysis_hash: str) -> None:
    comment=(f"% source_raw_sha256={raw_hash}\n% analysis_config_sha256={analysis_hash}\n"
             f"% generated_utc={datetime.now(timezone.utc).isoformat()}\n")
    path.write_text(comment+frame.to_latex(index=False,escape=True,float_format=lambda value:f"{value:.4f}"),encoding="utf-8")


def _performance_wide(summary: pd.DataFrame, support: pd.DataFrame, config: dict) -> pd.DataFrame:
    rows=[]
    lookup={(row.dataset,row.model):row for row in summary.itertuples(index=False)}
    restrictions={(row.dataset,row.model):row for row in support.itertuples(index=False)}
    for dataset in config["datasets"]:
        record={"dataset":config["display_names"][dataset]}
        for model in config["models"]:
            key=(dataset,model);sup=restrictions[key]
            if sup.support_status!="supported":
                record[model]="N/A†"
            else:
                row=lookup[key]
                record[model]=(f"ROC {row.roc_auc_mean:.4f}±{row.roc_auc_std:.4f}; "
                               f"PR {row.pr_auc_mean:.4f}±{row.pr_auc_std:.4f}; "
                               f"F1 {row.validation_f1_mean:.4f}±{row.validation_f1_std:.4f}")
        rows.append(record)
    return pd.DataFrame(rows)


def _ranking(summary: pd.DataFrame, view: pd.Series, metric: str) -> pd.DataFrame:
    models=str(view.models).split(";") if view.models else []
    datasets=str(view.datasets).split(";") if view.datasets else []
    selected=summary.loc[summary.model.isin(models)&summary.dataset.isin(datasets)]
    matrix=selected.pivot(index="dataset",columns="model",values=f"{metric}_mean")
    matrix=matrix.dropna(axis=0,how="any")
    ranks=matrix.rank(axis=1,ascending=False,method="average")
    result=ranks.reset_index().melt(id_vars="dataset",var_name="model",value_name="rank")
    result["view_name"]=view.view_name;result["metric"]=metric
    return result


def _statistics(raw: pd.DataFrame, views: pd.DataFrame, minimum: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    omnibus=[];pairs=[]
    for view in views.itertuples(index=False):
        models=str(view.models).split(";") if view.models else [];datasets=str(view.datasets).split(";") if view.datasets else []
        selected=raw.loc[raw.model.isin(models)&raw.dataset.isin(datasets)]
        for metric in PRIMARY:
            descriptive=len(datasets)<minimum or len(models)<3
            if len(datasets)>=2 and len(models)>=3:
                result=friedman_dataset_test(selected,metric=metric)
                omnibus.append({"view_name":view.view_name,**{k:v for k,v in result.items() if k not in {"average_ranks"}},
                                "descriptive_only":descriptive})
                pair=paired_model_tests(selected,metric=metric);pair["view_name"]=view.view_name;pairs.append(pair)
            else:
                omnibus.append({"view_name":view.view_name,"metric":metric,"n_models":len(models),
                                "n_datasets":len(datasets),"statistic":np.nan,"p_value":np.nan,
                                "descriptive_only":True,"reason":"insufficient complete dataset blocks"})
    return pd.DataFrame(omnibus),pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame()


def _topology_correlations(summary: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for model in sorted(summary.model.unique()):
        merged=summary.loc[summary.model.eq(model)].merge(topology,on="dataset",how="inner")
        for topo_metric in ("adjusted_homophily","fraud_homophily","mix_fraud_to_normal","positive_ratio","avg_degree"):
            for metric in PRIMARY:
                try:
                    result=spearman_with_bootstrap(merged[topo_metric],merged[f"{metric}_mean"],iterations=2000)
                    rows.append({"model":model,"topology_metric":topo_metric,"performance_metric":metric,**result})
                except ValueError as exc:
                    rows.append({"model":model,"topology_metric":topo_metric,"performance_metric":metric,
                                 "rho":np.nan,"p_value":np.nan,"ci95_low":np.nan,"ci95_high":np.nan,
                                 "n_datasets":len(merged),"reason":str(exc)})
    return pd.DataFrame(rows)


def _heatmap(matrix: pd.DataFrame, path: Path, title: str, *, vmin=None, vmax=None, cmap="viridis") -> None:
    fig,ax=plt.subplots(figsize=(max(8,.8*matrix.shape[1]),max(4,.5*matrix.shape[0])))
    values=np.ma.masked_invalid(matrix.to_numpy(dtype=float));image=ax.imshow(values,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]),matrix.columns,rotation=35,ha="right");ax.set_yticks(range(matrix.shape[0]),matrix.index)
    ax.set_title(title);fig.colorbar(image,ax=ax,shrink=.8);fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def _figures(output: Path, summary: pd.DataFrame, support: pd.DataFrame, views: pd.DataFrame,
             topology: pd.DataFrame, component: pd.DataFrame, raw: pd.DataFrame) -> None:
    figures=output/"figures";figures.mkdir(exist_ok=True)
    for number,metric,title in ((1,"roc_auc","ROC-AUC"),(2,"pr_auc","PR-AUC"),(3,"validation_f1","Validation F1")):
        matrix=summary.pivot(index="dataset",columns="model",values=f"{metric}_mean")
        _heatmap(matrix,figures/f"{number:02d}_final_{metric}_heatmap.png",title,vmin=0,vmax=1)
    rank_outputs=[]
    for view_name,number in (("broad_complete_case",4),("fraud_oriented",5)):
        view=views.loc[views.view_name.eq(view_name)].iloc[0];rank=_ranking(summary,view,"pr_auc");rank_outputs.append(rank)
        matrix=rank.pivot(index="dataset",columns="model",values="rank")
        _heatmap(matrix,figures/f"{number:02d}_rank_heatmap_{'broad' if number==4 else 'fraud'}.png",f"{view_name} PR-AUC ranks",cmap="magma_r")
    corr=summary.merge(topology,on="dataset")
    for number,x,title in ((6,"adjusted_homophily","Adjusted homophily vs PR-AUC"),(7,"fraud_homophily","Fraud homophily vs PR-AUC")):
        fig,ax=plt.subplots(figsize=(8,5))
        for model,group in corr.groupby("model"):
            ax.scatter(group[x],group.pr_auc_mean,label=model,s=28)
        ax.set_xlabel(x);ax.set_ylabel("PR-AUC mean");ax.set_title(title);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(figures/f"{number:02d}_{x}{'_correlation' if number==6 else '_vs_pr_auc'}.png",dpi=220);plt.close(fig)
    for number,metric,name in ((8,"pr_auc","pr_auc"),(9,"validation_f1","f1")):
        fig,ax=plt.subplots(figsize=(8,5));pivot=component.pivot_table(index="dataset",columns="variant",values=metric,aggfunc="mean")
        pivot.plot(kind="bar",ax=ax);ax.set_title(f"DLG components: {metric}");fig.tight_layout();fig.savefig(figures/f"{number:02d}_dlg_component_{name}.png",dpi=220);plt.close(fig)
    support_matrix=support.pivot(index="dataset",columns="model",values="supported").astype(float)
    _heatmap(support_matrix,figures/"10_model_dataset_support_matrix.png","Exact production support",vmin=0,vmax=1,cmap="RdYlGn")
    for number,y,title in ((11,"total_wall_sec","Runtime scaling"),(12,"nvidia_smi_peak_mb","GPU memory scaling")):
        fig,ax=plt.subplots(figsize=(8,5))
        for model,group in raw.groupby("model"):
            aggregate=group.groupby("nodes",as_index=False)[y].median();ax.plot(aggregate.nodes,aggregate[y],marker="o",label=model)
        ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("nodes");ax.set_ylabel(y);ax.set_title(title);ax.legend(fontsize=7,ncol=2);fig.tight_layout();fig.savefig(figures/f"{number:02d}_{'runtime' if number==11 else 'memory'}_scaling.png",dpi=220);plt.close(fig)


def finalize(config: dict) -> dict:
    output=Path(config["experiment"]["output_root"]);support_path=output/"manifests/model_dataset_support_matrix_v2.csv"
    support=pd.read_csv(support_path);validate_support_matrix(support,list(config["datasets"]),list(config["models"]))
    support_hash=(output/"manifests/model_dataset_support_matrix_v2.sha256").read_text().strip()
    if sha256_file(support_path)!=support_hash: raise RuntimeError("support matrix hash drift")
    observed=_collect_json(output);success=observed.loc[observed.status.eq("success")].copy()
    supported_keys=set(map(tuple,support.loc[support.support_status.eq("supported"),["dataset","model"]].itertuples(index=False,name=None)))
    raw=success.loc[[tuple(row) in supported_keys for row in success[["dataset","model"]].itertuples(index=False,name=None)]].copy()
    validate_final_raw(raw,support,list(config["seeds"]))
    raw=raw.sort_values(["dataset","model","seed"]);raw_path=output/"raw/benchmark_raw.csv";raw.to_csv(raw_path,index=False,lineterminator="\n")
    raw_hash=_write_hash(raw_path);analysis_hash=hashlib.sha256(yaml.safe_dump(config.get("analysis",{}),sort_keys=True).encode()).hexdigest()
    summary=seed_first_summary(raw,METRICS+["validation_threshold","validation_threshold_percentile"])
    summary.to_csv(output/"summary/seed_aggregated_performance.csv",index=False)
    views=complete_case_views(support,FRAUD_DATASETS);views.to_csv(output/"statistics/complete_case_views.csv",index=False)
    rankings=pd.concat([_ranking(summary,row,metric) for _,row in views.iterrows() for metric in PRIMARY],ignore_index=True)
    rankings.to_csv(output/"statistics/rankings.csv",index=False)
    omnibus,pairs=_statistics(raw,views,int(config["analysis"]["minimum_inference_datasets"]))
    omnibus.to_csv(output/"statistics/friedman_tests.csv",index=False);pairs.to_csv(output/"statistics/wilcoxon_holm.csv",index=False)
    if not pairs.empty:
        pairs.loc[pairs.model_a.eq("DLG-Aug")|pairs.model_b.eq("DLG-Aug")].to_csv(output/"statistics/dlg_focused_comparisons.csv",index=False)
    topology,graph_hashes=_topology(config);topology.to_csv(output/"topology/final_graph_topology.csv",index=False)
    (output/"manifests/final_graph_hashes.json").write_text(json.dumps(graph_hashes,indent=2)+"\n",encoding="utf-8")
    correlations=_topology_correlations(summary,topology);correlations.to_csv(output/"topology/topology_performance_correlations.csv",index=False)
    freeze=build_data_freeze(config,datasets=list(config["datasets"]));(output/"manifests/data_freeze.json").write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf-8")
    environment=build_environment_freeze();(output/"manifests/environment_freeze.json").write_text(json.dumps(environment,indent=2)+"\n",encoding="utf-8")
    dataset_table=_dataset_table(config,freeze);dataset_table.to_csv(output/"tables/01_dataset_characteristics.csv",index=False)
    performance=_performance_wide(summary,support,config);performance.to_csv(output/"tables/02_overall_performance.csv",index=False)
    performance.loc[performance.dataset.isin([config["display_names"][name] for name in FRAUD_DATASETS])].to_csv(output/"tables/03_fraud_oriented_performance.csv",index=False)
    resources=raw.groupby(["dataset","model"],as_index=False).agg(runtime_mean_sec=("total_wall_sec","mean"),gpu_memory_peak_mb=("nvidia_smi_peak_mb","max"),rss_peak_mb=("rss_peak_mb","max"))
    scalability=support.merge(resources,on=["dataset","model"],how="left");scalability.to_csv(output/"tables/04_scalability_support.csv",index=False)
    component_path=Path("outputs/sci_round4b/ablation/component_raw.csv");component=pd.read_csv(component_path);component.to_csv(output/"ablation/dlg_component_raw_frozen.csv",index=False)
    component_summary=component.groupby(["dataset","variant"],as_index=False)[["pr_auc","validation_f1"]].agg(["mean","std"]);component_summary.columns=["_".join(filter(None,col)) for col in component_summary.columns];component_summary.to_csv(output/"tables/05_dlg_extended_components.csv",index=False)
    for csv in sorted((output/"tables").glob("*.csv")):_latex(pd.read_csv(csv),csv.with_suffix(".tex"),raw_hash,analysis_hash)
    _figures(output,summary,support,views,topology,component,raw)
    registry={"models":list(config["models"]),"datasets":[{"id":name,"display_name":config["display_names"][name]} for name in config["datasets"]]}
    (output/"manifests/registry.json").write_text(json.dumps(registry,indent=2)+"\n",encoding="utf-8")
    execution_hashes=json.loads((output/"manifests/execution_hashes.json").read_text())
    if execution_hashes["benchmark_execution_source_hash"]!=benchmark_execution_source_hash():raise RuntimeError("benchmark execution source drift")
    gate={"decision":"FINAL_COMPLETE" if support.support_status.eq("supported").all() else "FINAL_COMPLETE_WITH_RESTRICTIONS",
          "supported_pairs":int(support.supported.sum()),"unsupported_pairs":int((~support.supported).sum()),
          "successful_runs":len(raw),"expected_runs":int(support.supported.sum()*len(config["seeds"])),
          "support_matrix_hash":support_hash,"benchmark_raw_hash":raw_hash,"unknown_failures":0}
    (output/"manifests/final_gate.json").write_text(json.dumps(gate,indent=2)+"\n",encoding="utf-8")
    write_report(output,gate,views,omnibus,correlations,scalability,component,raw_hash)
    return gate


def write_report(output: Path,gate:dict,views:pd.DataFrame,omnibus:pd.DataFrame,correlations:pd.DataFrame,
                 scalability:pd.DataFrame,component:pd.DataFrame,raw_hash:str)->None:
    report=Path("docs/work_reports/207_benchmark_round_5/01_round5_final_benchmark_report.md")
    unsupported=scalability.loc[~scalability.supported,["dataset","model","support_status","restriction_reason"]]
    significant=omnibus.loc[(~omnibus.descriptive_only)&(omnibus.p_value<.05)] if "p_value" in omnibus else pd.DataFrame()
    text=f"""# DLG-GNN SCI Benchmark Round 5 Final Report

## 1. Experimental protocol

Exact full-shared-graph backend, fixed synthetic instance seed 42, model seeds 42–46, validation-selected threshold를 사용했다. Score inversion, partition, sampling substitute, CPU fallback, worst-rank imputation은 사용하지 않았다.

## 2. Support matrix

Final decision: **{gate['decision']}**. Supported pairs {gate['supported_pairs']}, restricted pairs {gate['unsupported_pairs']}; successful performance runs {gate['successful_runs']}/{gate['expected_runs']}.

{unsupported.to_markdown(index=False)}

## 3. Five-seed performance

Source: `outputs/sci_round5_final/tables/02_overall_performance.csv`. Seed mean, sample standard deviation와 t-based 95% CI를 `summary/seed_aggregated_performance.csv`에 고정했다.

## 4. Fraud-oriented performance

Source: `tables/03_fraud_oriented_performance.csv`. Yelp-Syn은 synthetic injection이며 real Yelp fraud 성능으로 해석하지 않는다.

## 5. Statistical tests

Complete-case views는 final support matrix에서 재계산했다. All-8 view의 block 수가 부족하면 descriptive-only로 표시한다. Strong-inference-eligible significant omnibus rows: {len(significant)}.

{views.to_markdown(index=False)}

## 6. Topology associations

Adjusted/fraud-conditioned homophily와 성능의 Spearman association을 valid observed datasets만으로 계산했다. 결과는 연관성이지 인과 evidence가 아니다.

## 7. DLG component analysis

Historical DLG는 DLG-Aug이며 DLG-Fusion은 extended component이다. Round 4B component freeze와 Round 5 main five-seed 결과를 분리해 보고한다.

## 8. Scalability

Performance rank와 support/runtime/memory를 합치지 않았다. Exact unsupported는 낮은 성능이 아니라 실행 가능성 제한이다.

## 9. Failure/unsupported analysis

AnomalyDAE nonlinear all-pairs, GADNR native neighborhood-distribution, Reddit CONAD contrastive materialization 제한을 frozen support ledger에 보존했다.

## 10. Limitations

5 model seeds는 uncertainty 추정에 제한적이며, synthetic primary graphs는 injection seed 42에 고정됐다. Unsupported cells에는 performance estimate가 없다.

## 11. Claims supported by evidence

- Exact, leakage-safe protocol에서 supported detector-dataset pair의 reproducible five-seed 성능을 보고할 수 있다.
- Local augmentation의 이득/손해는 dataset-dependent한 descriptive pattern으로 평가할 수 있다.
- Exact historical implementations 사이에 명확한 scalability 차이가 존재한다.

## 12. Claims not supported by evidence

- DLG의 universal superiority 또는 universal SOTA.
- Fraud heterophily가 성능을 인과적으로 결정한다는 주장.
- Fusion의 universal benefit.
- Yelp-Syn을 real Yelp fraud result로 해석하는 주장.
- DGraphFin random split을 temporal superiority evidence로 해석하는 주장.

## Reproduction commands

```bash
PYTHONPATH=src ../.venv/bin/python -m gog_fraud.pipelines.run_sci_round5 --config configs/benchmark/sci_round5_final.yaml --stage phase0
PYTHONPATH=src ../.venv/bin/python -m gog_fraud.pipelines.run_sci_round5 --config configs/benchmark/sci_round5_final.yaml --stage phase1 --resume
PYTHONPATH=src ../.venv/bin/python -m gog_fraud.pipelines.analyze_sci_round5 --config configs/benchmark/sci_round5_final.yaml
```

Frozen raw SHA-256: `{raw_hash}`.
"""
    report.write_text(text,encoding="utf-8")


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"));print(json.dumps(finalize(config),indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
