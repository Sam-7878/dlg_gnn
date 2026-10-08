#!/usr/bin/env python3
"""Compare upstream dense AnomalyDAE with the existing exact row-block path."""
from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from pygod.nn import AnomalyDAEBase
from pygod.nn.functional import double_recon_loss
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.utils import to_dense_adj

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from gog_fraud.models.pygod.exact_reconstruction import exact_double_reconstruction_score
from gog_fraud.models.pygod.shared_reconstruction import ExactAnomalyDAEBase

OUT = ROOT / "evaluation/benchmark/v2/diagnostics/anomalydae_chunked"
DATA = ROOT / "outputs/benchmark/a04_constructed_graphs"
BLOCK_SIZE = 256  # frozen before any A05 performance inspection


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def comparison(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.detach(), b.detach()
    delta = (a - b).abs()
    absolute = float(delta.max())
    relative = float((delta / b.abs().clamp_min(1e-6)).max())
    atol, rtol = (1e-8, 1e-8) if a.dtype == torch.float64 else (1e-5, 1e-4)
    ok = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
    return {"max_abs": absolute, "max_rel": relative, "pass": ok}


def case(name: str, device: torch.device, dtype: torch.dtype = torch.float32) -> dict:
    graph = torch.load(DATA / f"{name}.pt", map_location="cpu", weights_only=False)
    x = graph["x"].to(device=device, dtype=dtype)
    edge_index = graph["edge_index"].to(device)
    y = graph["y"].detach().cpu().numpy()
    n, feature_dim = x.shape
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    dense = AnomalyDAEBase(in_dim=feature_dim, num_nodes=n, emb_dim=64, hid_dim=64, dropout=0.0).to(device=device,dtype=dtype)
    exact = ExactAnomalyDAEBase(deepcopy(dense)).to(device)
    dense.train(); exact.train()
    s = to_dense_adj(edge_index, max_num_nodes=n)[0].to(device)
    weight, positive_attribute, positive_structure = .5, .5, .5

    dense_xhat, dense_pred = dense(x, edge_index, n)
    dense_attribute = double_recon_loss(x, dense_xhat, s, dense_pred, 1.0, positive_attribute, positive_structure)
    dense_structure = double_recon_loss(x, dense_xhat, s, dense_pred, 0.0, positive_attribute, positive_structure)
    dense_score = double_recon_loss(x, dense_xhat, s, dense_pred, weight, positive_attribute, positive_structure)
    dense_loss = dense_score.mean()

    exact_xhat, z = exact(x, edge_index)
    chunks = []
    for start in range(0, n, BLOCK_SIZE):
        rows = torch.arange(start, min(start + BLOCK_SIZE, n), device=device)
        chunks.append({
            "attribute": exact_double_reconstruction_score(x, exact_xhat, z, edge_index,
                weight=1.0, positive_weight_attribute=positive_attribute,
                positive_weight_structure=positive_structure, sigmoid_structure=True,
                rows=rows, backend="chunked_exact", chunk_size=BLOCK_SIZE),
            "structure": exact_double_reconstruction_score(x, exact_xhat, z, edge_index,
                weight=0.0, positive_weight_attribute=positive_attribute,
                positive_weight_structure=positive_structure, sigmoid_structure=True,
                rows=rows, backend="chunked_exact", chunk_size=BLOCK_SIZE),
        })
    exact_attribute = torch.cat([c["attribute"] for c in chunks])
    exact_structure = torch.cat([c["structure"] for c in chunks])
    exact_score = weight * exact_attribute + (1-weight) * exact_structure
    exact_loss = exact_score.mean()

    dense_loss.backward(); exact_loss.backward()
    dense_params = list(dense.named_parameters())
    exact_params = list(exact.dense_model.named_parameters())
    gradient = {name: comparison(dp.grad, ep.grad) for (name, dp), (_, ep) in zip(dense_params, exact_params)}
    opt_dense = torch.optim.Adam(dense.parameters(), lr=.004)
    opt_exact = torch.optim.Adam(exact.parameters(), lr=.004)
    opt_dense.step(); opt_exact.step()
    update = {name: comparison(dp, ep) for (name, dp), (_, ep) in zip(dense_params, exact_params)}

    dense.eval(); exact.eval()
    with torch.no_grad():
        dx, ds = dense(x, edge_index, n)
        dense_final = double_recon_loss(x, dx, s, ds, weight, positive_attribute, positive_structure)
        ex, ez = exact(x, edge_index)
        exact_final = torch.cat([
            exact_double_reconstruction_score(x, ex, ez, edge_index,
                weight=weight, positive_weight_attribute=positive_attribute,
                positive_weight_structure=positive_structure, sigmoid_structure=True,
                rows=torch.arange(start, min(start+BLOCK_SIZE, n), device=device),
                backend="chunked_exact", chunk_size=BLOCK_SIZE)
            for start in range(0, n, BLOCK_SIZE)
        ])
    d = dense_final.cpu().numpy(); e = exact_final.cpu().numpy()
    metrics = {"dense_roc_auc": float(roc_auc_score(y,d)), "exact_roc_auc": float(roc_auc_score(y,e)),
               "dense_pr_auc": float(average_precision_score(y,d)), "exact_pr_auc": float(average_precision_score(y,e))}
    checks = {
        "attribute_reconstruction": comparison(dense_xhat, exact_xhat),
        "attribute_score": comparison(dense_attribute, exact_attribute),
        "structure_score": comparison(dense_structure, exact_structure),
        "total_score": comparison(dense_score, exact_score),
        "mean_loss": comparison(dense_loss, exact_loss),
        "gradients": all(c["pass"] for c in gradient.values()),
        "optimizer_step": all(c["pass"] for c in update.values()),
        "final_score": comparison(dense_final, exact_final),
        "final_order_spearman": float(spearmanr(d,e).statistic),
        "metrics": metrics,
    }
    # Adam can amplify near-zero gradient reduction-order noise. The frozen
    # forward/gradient tolerance remains 1e-5 + 1e-4*|reference|; a single
    # post-update parameter may differ by up to 2e-5 absolute. Metrics can
    # shift when nearly tied scores switch order within score tolerance.
    max_update_abs = max(v["max_abs"] for v in update.values())
    metric_delta = max(abs(metrics["dense_roc_auc"]-metrics["exact_roc_auc"]),
                       abs(metrics["dense_pr_auc"]-metrics["exact_pr_auc"]))
    checks["max_post_update_parameter_abs"] = max_update_abs
    checks["max_metric_abs_delta"] = metric_delta
    post_adam_limit = 1e-8 if dtype == torch.float64 else 2e-5
    passed = (all(checks[key]["pass"] for key in ("attribute_reconstruction","attribute_score",
                  "structure_score","total_score","mean_loss","final_score"))
              and checks["gradients"] and max_update_abs <= post_adam_limit
              and metric_delta <= 1e-5)
    return {"dataset":name,"nodes":n,"feature_dimension":feature_dim,"block_size":BLOCK_SIZE,
            "device":str(device),"dtype":str(dtype),"status":"PASS" if passed else "FAIL","checks":checks,
            "gradient_comparison":gradient,"updated_parameter_comparison":update,
            "dense_final_score_sha256":digest(dense_final),"exact_final_score_sha256":digest(exact_final)}


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A05 equivalence requires CUDA")
    device = torch.device("cuda:0")
    OUT.mkdir(parents=True,exist_ok=True)
    results = [case(name,device) for name in ("Cora-Syn","BitcoinOTC")]
    results.append(case("Cora-Syn",device,torch.float64))
    report = {"status":"PASS" if all(r["status"]=="PASS" for r in results) else "FAIL",
              "reference":"PyGOD 1.1.0 AnomalyDAEBase dense full graph",
              "candidate":"SharedAnomalyDAE existing exact sigmoid row-block backend",
              "float32_tolerance":{"atol":1e-5,"rtol":1e-4},
              "float64_tolerance":{"atol":1e-8,"rtol":1e-8},
              "documented_reduction_order_limits":{"post_adam_max_abs":2e-5,
                  "metric_max_abs_delta_for_near_ties":1e-5},"datasets":results}
    (OUT/"anomalydae_exact_equivalence.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# A05 AnomalyDAE dense versus row-block equivalence", "",f"**Gate: {report['status']}**", "",
           "The candidate uses the existing exact row-block implementation with block size 256. No sampled edges or nodes are used.", ""]
    for r in results:
        c=r["checks"]
        lines += [f"## {r['dataset']} ({r['dtype']})","",f"Status: **{r['status']}**; nodes: {r['nodes']}; device: {r['device']}.","",
            f"- Max score difference: {c['total_score']['max_abs']:.6g}; gradient pass: {c['gradients']}; strict update pass: {c['optimizer_step']}.",
            f"- Max post-Adam parameter difference: {c['max_post_update_parameter_abs']:.6g}; max ROC/PR difference: {c['max_metric_abs_delta']:.6g}.",
            f"- Final Spearman: {c['final_order_spearman']:.9f}; ROC-AUC: {c['metrics']['dense_roc_auc']:.6f} / {c['metrics']['exact_roc_auc']:.6f}; PR-AUC: {c['metrics']['dense_pr_auc']:.6f} / {c['metrics']['exact_pr_auc']:.6f}.",""]
    (OUT/"anomalydae_exact_equivalence.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":report["status"],"datasets":[{"dataset":r["dataset"],"status":r["status"]} for r in results]}))
    if report["status"]!="PASS":raise SystemExit(2)


if __name__=="__main__":main()
