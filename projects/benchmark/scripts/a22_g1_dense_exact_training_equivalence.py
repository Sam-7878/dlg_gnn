#!/usr/bin/env python3
"""A22-G1: dense PyGOD reference vs production exact-Gram TRAINING equivalence.

Scope (no GPU, no benchmark data, tiny fixtures only):
  * loss scalar, per-node score, parameter gradients and ONE Adam step
  * 10-epoch loss/parameter trajectory
  * float64 and float32 with PREDECLARED tolerances (see TOL below; fixed before first run)
  * directed asymmetric edges, isolated nodes, self-loops, duplicate edges
  * negative controls that deliberately use a *different* objective
  * function-level checks: positive-weight overrides, sigmoid/chunked path,
    exact-zero residual gradients, float32 Gram cancellation

Reference implementations are the INSTALLED libraries' own code paths:
  DOMINANT -> pygod.detector.DOMINANT.forward_model (+ pygod.nn.functional.double_recon_loss)
  DLG-Base -> gog_fraud.models.pygod.dlg.DLG.forward_model (dense DLGBase.loss_func)
  DLG-Aug  -> gog_fraud.models.pygod.dlg_full.DLGFull.forward_model (dense DLGFullBase.loss_func)
Production implementations are the shared exact classes that ran the benchmark:
  SharedDOMINANT / SharedDLGBase / SharedDLGFull.fit  (exact_sparse backend)
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata as md
import json
import math
import os
import platform
import sys
import traceback

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from pygod.detector import DOMINANT as PyGODDOMINANT  # noqa: E402
from pygod.nn.functional import double_recon_loss  # noqa: E402

from gog_fraud.models.pygod.dlg import DLG  # noqa: E402
from gog_fraud.models.pygod.dlg_full import DLGFull  # noqa: E402
from gog_fraud.models.pygod.exact_reconstruction import (  # noqa: E402
    exact_double_reconstruction_score,
)
from gog_fraud.models.pygod.shared_reconstruction import (  # noqa: E402
    SharedDLGBase,
    SharedDLGFull,
    SharedDOMINANT,
    SharedFullGraphMixin,
)

# ---------------------------------------------------------------------------
# PREDECLARED TOLERANCES (frozen before the first execution of this script)
# ---------------------------------------------------------------------------
TOL = {
    "float64": {
        "loss_rel": 1e-10,
        "score_max_abs": 1e-9,
        "grad_rel_l2": 1e-8,
        "step_param_max_abs": 1e-9,
        "trajectory_loss_rel": 1e-7,
        "trajectory_param_max_abs": 1e-7,
    },
    "float32": {
        "loss_rel": 1e-5,
        "score_max_abs": 5e-4,
        "grad_rel_l2": 1e-4,
        "step_param_max_abs": 2e-5,
        "trajectory_loss_rel": 1e-3,
        "trajectory_param_max_abs": 5e-4,
    },
}
NEGATIVE_CONTROL_FACTOR = 100.0  # control must exceed tolerance by >= this factor in grad_rel_l2

N, D, HID, LAYERS, LR, WEIGHT, L1DIM = 40, 8, 8, 4, 4e-3, 0.5, 6
EPOCHS_TRAJ = 10
SEEDS = [11, 23, 47]


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def make_graph(seed: int) -> tuple[Data, dict]:
    g = torch.Generator().manual_seed(seed)
    active = N - 6  # last 6 nodes are isolated
    src = torch.randint(0, active, (150,), generator=g)
    dst = torch.randint(0, active, (150,), generator=g)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    loops = torch.tensor([3, 7, 11])
    dup = torch.randperm(src.numel(), generator=g)[:5]
    src = torch.cat([src, loops, src[dup]])
    dst = torch.cat([dst, loops, dst[dup]])
    edge_index = torch.stack([src, dst])
    x = torch.randn(N, D, generator=g)
    data = Data(x=x, edge_index=edge_index, num_nodes=N)
    dense = to_dense_adj(edge_index, max_num_nodes=N)[0]
    stats = {
        "num_edges_with_dups_and_loops": int(edge_index.shape[1]),
        "asymmetric_entries": int((dense != dense.T).sum()),
        "isolated_nodes": int(((dense.sum(0) + dense.sum(1)) == 0).sum()),
        "self_loops": int(dense.diagonal().gt(0).sum()),
        "max_multiplicity": float(dense.max()),
    }
    return data, stats


def aug_data(data: Data, seed: int) -> Data:
    g = torch.Generator().manual_seed(seed + 1000)
    l1 = torch.randn(N, L1DIM, generator=g)
    out = data.clone()
    out.dlg_original_x = data.x.clone()
    out.x = torch.cat([data.x, l1], dim=-1)
    out._dlg_full_augmented = True
    return out


def sampled(data: Data) -> Data:
    s = data.clone()
    s.s = to_dense_adj(s.edge_index, max_num_nodes=N)[0]
    s.batch_size = N
    s.n_id = torch.arange(N)
    return s


def named_params(model):
    return {n: p for n, p in model.named_parameters()}


def flat(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def grad_metrics(ref: dict, prod: dict) -> dict:
    names = sorted(ref)
    if names != sorted(prod):
        return {"error": "parameter name mismatch", "ref": names, "prod": sorted(prod)}
    g_ref = flat([ref[n] for n in names]).double()
    g_prod = flat([prod[n] for n in names]).double()
    denom = float(g_ref.norm()) + 1e-300
    cos = float(torch.dot(g_ref, g_prod) / ((g_ref.norm() * g_prod.norm()) + 1e-300))
    return {
        "grad_rel_l2": float((g_ref - g_prod).norm() / denom),
        "grad_max_abs": float((g_ref - g_prod).abs().max()),
        "grad_cosine": cos,
        "grad_norm_ref": float(g_ref.norm()),
        "grad_norm_prod": float(g_prod.norm()),
        "all_finite_ref": bool(torch.isfinite(g_ref).all()),
        "all_finite_prod": bool(torch.isfinite(g_prod).all()),
    }


def make_reference(family: str, seed: int):
    torch.manual_seed(seed)
    if family == "DOMINANT":
        det = PyGODDOMINANT(hid_dim=HID, num_layers=LAYERS, dropout=0.0, weight_decay=0.0,
                            lr=LR, epoch=1, gpu=-1, weight=WEIGHT)
    elif family == "DLG-Base":
        det = DLG(hid_dim=HID, num_layers=LAYERS, dropout=0.0, weight_decay=0.0, lr=LR,
                  epoch=1, gpu=-1, weight=WEIGHT, alpha=0.5)
    elif family == "DLG-Aug":
        det = DLGFull(hid_dim=HID, num_layers=LAYERS, dropout=0.0, weight_decay=0.0, lr=LR,
                      epoch=1, gpu=-1, weight=WEIGHT, l1_hid_dim=L1DIM)
        det._orig_dim = D
    else:
        raise ValueError(family)
    return det


def make_production(family: str):
    common = dict(hid_dim=HID, num_layers=LAYERS, dropout=0.0, weight_decay=0.0, lr=LR,
                  gpu=-1, weight=WEIGHT)
    if family == "DOMINANT":
        prod = SharedDOMINANT(epoch=1, **common)
    elif family == "DLG-Base":
        prod = SharedDLGBase(epoch=1, alpha=0.5, **common)
    elif family == "DLG-Aug":
        prod = SharedDLGFull(epoch=1, l1_hid_dim=L1DIM, **common)
        prod._orig_dim = D
    else:
        raise ValueError(family)
    return prod


def init_ref_model(det, data):
    det.num_nodes, det.in_dim = data.x.shape
    det.model = det.init_model(**det.kwargs)
    return det.model


def init_prod_model(prod, data, ref_state):
    prod.num_nodes, prod.in_dim = data.x.shape
    model = prod.init_model(**prod.kwargs)
    model.load_state_dict(ref_state)
    prod.init_model = lambda **kw: model  # real fit() loop runs on the pre-loaded model
    return model


def run_pair(family: str, seed: int, dtype_name: str, epochs: int):
    """Return metrics for one family/seed/dtype combination."""
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    base, stats = make_graph(seed)
    data = aug_data(base, seed) if family == "DLG-Aug" else base
    data.x = data.x.to(dtype)
    if hasattr(data, "dlg_original_x"):
        data.dlg_original_x = data.dlg_original_x.to(dtype)

    # ---------------- reference (installed/dense code path) -------------------
    ref_det = make_reference(family, seed)
    ref_model = init_ref_model(ref_det, data)
    ref_state0 = copy.deepcopy(ref_model.state_dict())
    ref_model.train()
    opt = torch.optim.Adam(ref_model.parameters(), lr=LR, weight_decay=0.0)

    losses_ref, params_after_1 = [], None
    grads_ref = None
    score_ref0 = None
    for ep in range(epochs):
        loss, score = ref_det.forward_model(sampled(data))
        if ep == 0:
            score_ref0 = score.clone()
        opt.zero_grad()
        loss.backward()
        if ep == 0:
            grads_ref = {n: p.grad.detach().clone() for n, p in ref_model.named_parameters()}
        opt.step()
        losses_ref.append(float(loss))
        if ep == 0:
            params_after_1 = {n: p.detach().clone() for n, p in ref_model.named_parameters()}
    params_final_ref = {n: p.detach().clone() for n, p in ref_model.named_parameters()}

    # ---------------- production exact-Gram path (real fit loop) --------------
    def run_prod(n_epochs: int):
        prod = make_production(family)
        prod.epoch = n_epochs
        pdata = data.clone()
        if family == "DLG-Aug":
            pdata._dlg_full_augmented = True
        model = init_prod_model(prod, pdata, ref_state0)
        prod.fit(pdata)
        return prod, model

    prod1, model1 = run_prod(1)
    grads_prod = {n: p.grad.detach().clone() for n, p in model1.named_parameters()}
    params_prod1 = {n: p.detach().clone() for n, p in model1.named_parameters()}
    loss_prod0 = prod1.loss_history_[0]
    score_prod0 = prod1.decision_score_.clone()

    metrics = {"family": family, "seed": seed, "dtype": dtype_name, "graph": stats}
    metrics.update(grad_metrics(grads_ref, grads_prod))
    metrics["loss_ref"] = losses_ref[0]
    metrics["loss_prod"] = float(loss_prod0)
    metrics["loss_rel"] = abs(losses_ref[0] - float(loss_prod0)) / max(abs(losses_ref[0]), 1e-300)
    metrics["score_max_abs"] = float((score_ref0.double() - score_prod0.double()).abs().max())
    metrics["step_param_max_abs"] = float(max(
        (params_after_1[n].double() - params_prod1[n].double()).abs().max() for n in params_after_1))

    # ---------------- trajectory ---------------------------------------------
    if epochs > 1:
        prodN, modelN = run_prod(epochs)
        lp = prodN.loss_history_
        metrics["trajectory_loss_rel"] = max(
            abs(a - b) / max(abs(a), 1e-300) for a, b in zip(losses_ref, lp))
        metrics["trajectory_param_max_abs"] = float(max(
            (params_final_ref[n].double() - p.detach().double()).abs().max()
            for n, p in modelN.named_parameters()))
        metrics["loss_first_last_ref"] = [losses_ref[0], losses_ref[-1]]
        metrics["loss_first_last_prod"] = [lp[0], lp[-1]]

    # ---------------- negative controls (same weights, different objective) ---
    def neg_grads(kind: str):
        det = make_reference(family, seed)
        m = init_ref_model(det, data)
        m.load_state_dict(ref_state0)
        m.train()
        s = to_dense_adj(data.edge_index, max_num_nodes=N)[0]
        x_in = data.x
        x_t = data.x[:, :D] if family == "DLG-Aug" else data.x
        x_, s_ = m(x_in, data.edge_index)
        if kind == "squared_per_node_mean":
            attr = ((x_t - x_) ** 2).sum(1)
            stru = ((s - s_) ** 2).sum(1)
            loss = (WEIGHT * attr + (1 - WEIGHT) * stru).mean()
        elif kind == "global_mean_squared_frobenius":
            loss = WEIGHT * ((x_t - x_) ** 2).mean() + (1 - WEIGHT) * ((s - s_) ** 2).mean()
        else:
            raise ValueError(kind)
        m.zero_grad()
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in m.named_parameters()}

    metrics["negative_controls"] = {}
    for kind in ("squared_per_node_mean", "global_mean_squared_frobenius"):
        mm = grad_metrics(neg_grads(kind), grads_prod)
        mm["detected_as_mismatch"] = bool(
            mm["grad_rel_l2"] > NEGATIVE_CONTROL_FACTOR * TOL[dtype_name]["grad_rel_l2"])
        metrics["negative_controls"][kind] = mm
    torch.set_default_dtype(torch.float32)
    return metrics


def passes(m: dict) -> dict:
    tol = TOL[m["dtype"]]
    checks = {
        "loss_rel": m["loss_rel"] <= tol["loss_rel"],
        "score_max_abs": m["score_max_abs"] <= tol["score_max_abs"],
        "grad_rel_l2": m["grad_rel_l2"] <= tol["grad_rel_l2"],
        "step_param_max_abs": m["step_param_max_abs"] <= tol["step_param_max_abs"],
    }
    if "trajectory_loss_rel" in m:
        checks["trajectory_loss_rel"] = m["trajectory_loss_rel"] <= tol["trajectory_loss_rel"]
        checks["trajectory_param_max_abs"] = (
            m["trajectory_param_max_abs"] <= tol["trajectory_param_max_abs"])
    checks["negative_controls_detected"] = all(
        v["detected_as_mismatch"] for v in m["negative_controls"].values())
    return checks


# ---------------------------------------------------------------------------
# function-level checks
# ---------------------------------------------------------------------------
def function_level() -> dict:
    out = {}
    torch.set_default_dtype(torch.float64)
    data, _ = make_graph(5)
    x = data.x.double()
    s = to_dense_adj(data.edge_index, max_num_nodes=N)[0]

    def grads_of(fn):
        z = torch.randn(N, HID, generator=torch.Generator().manual_seed(9), dtype=torch.float64,
                        requires_grad=True)
        xh = torch.randn(N, D, generator=torch.Generator().manual_seed(10), dtype=torch.float64,
                         requires_grad=True)
        sc = fn(z, xh)
        sc.mean().backward()
        return sc.detach(), z.grad.clone(), xh.grad.clone()

    # F1 positive-weight overrides (score + gradients), linear decoder
    f1 = []
    for pa, ps in [(0.5, 0.5), (0.3, 0.5), (0.5, 0.8), (0.3, 0.8)]:
        ref = grads_of(lambda z, xh: double_recon_loss(x, xh, s, z @ z.T, 0.5, pa, ps, False))
        exa = grads_of(lambda z, xh: exact_double_reconstruction_score(
            x, xh, z, data.edge_index, weight=0.5, positive_weight_attribute=pa,
            positive_weight_structure=ps, backend="exact_sparse"))
        f1.append({
            "pos_weight_a": pa, "pos_weight_s": ps,
            "score_max_abs": float((ref[0] - exa[0]).abs().max()),
            "grad_z_max_abs": float((ref[1] - exa[1]).abs().max()),
            "grad_xhat_max_abs": float((ref[2] - exa[2]).abs().max()),
        })
    out["F1_positive_weight_overrides"] = f1

    # F2 sigmoid decoder through the REAL production chunked backward
    class Stub(SharedFullGraphMixin):
        pass

    f2 = []
    for backend, chunk, sig in [("exact_sparse", 8192, False), ("chunked_exact", 7, False),
                                ("chunked_exact", 7, True), ("chunked_exact", 40, True)]:
        z = torch.randn(N, HID, generator=torch.Generator().manual_seed(9), dtype=torch.float64,
                        requires_grad=True)
        xh = torch.randn(N, D, generator=torch.Generator().manual_seed(10), dtype=torch.float64,
                         requires_grad=True)
        dec = torch.sigmoid(z @ z.T) if sig else z @ z.T
        ref_score = double_recon_loss(x, xh, s, dec, 0.5)
        ref_score.mean().backward()
        gz_ref, gx_ref = z.grad.clone(), xh.grad.clone()
        z.grad = None
        xh.grad = None
        stub = Stub()
        stub.reconstruction_backend = backend
        stub.score_chunk_size = chunk
        stub.sigmoid_structure = sig
        stub.weight = 0.5
        stub.device = torch.device("cpu")
        stub.positive_weight_attribute = 0.5
        stub.positive_weight_structure = 0.5
        dd = Data(num_nodes=N)
        scores = stub._backward_exact(dd, (x, xh, z, data.edge_index), None)
        f2.append({
            "backend": backend, "chunk": chunk, "sigmoid": sig,
            "score_max_abs": float((ref_score.detach() - scores.double()).abs().max()),
            "grad_z_max_abs": float((gz_ref - z.grad).abs().max()),
            "grad_xhat_max_abs": float((gx_ref - xh.grad).abs().max()),
        })
    out["F2_sigmoid_chunked_via_production_backward"] = f2

    # F3 exact-zero residual: PyGOD sqrt vs production stable sqrt
    z0 = torch.zeros(N, HID, dtype=torch.float64, requires_grad=True)
    xh0 = x.clone().requires_grad_(True)  # x_hat == x -> zero attribute residual
    s_iso = torch.zeros(N, N, dtype=torch.float64)  # empty graph -> zero structure residual (z=0)
    ref_sc = double_recon_loss(x, xh0, s_iso, z0 @ z0.T, 0.5)
    ref_sc.mean().backward()
    ref_finite = bool(torch.isfinite(z0.grad).all() and torch.isfinite(xh0.grad).all())
    z1 = torch.zeros(N, HID, dtype=torch.float64, requires_grad=True)
    xh1 = x.clone().requires_grad_(True)
    ei_empty = torch.empty(2, 0, dtype=torch.long)
    ex_sc = exact_double_reconstruction_score(x, xh1, z1, ei_empty, weight=0.5,
                                              backend="exact_sparse")
    ex_sc.mean().backward()
    out["F3_exact_zero_residual"] = {
        "pygod_grad_finite": ref_finite,
        "pygod_grad_has_nan": bool(torch.isnan(z0.grad).any() or torch.isnan(xh0.grad).any()),
        "exact_grad_finite": bool(torch.isfinite(z1.grad).all() and torch.isfinite(xh1.grad).all()),
        "exact_grad_all_zero": bool((z1.grad == 0).all() and (xh1.grad == 0).all()),
        "score_pygod_max": float(ref_sc.abs().max()),
        "score_exact_max": float(ex_sc.abs().max()),
        "interpretation": "measure-zero event; production defines subgradient 0 where PyGOD's "
                          "sqrt backward yields NaN. Objective values coincide.",
    }

    # F4 float32 Gram cancellation (near-zero residual) vs float64 reference
    f4 = []
    for eps in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
        d = 16
        zt64 = torch.eye(d, dtype=torch.float64) * (1.0 + eps)
        ei = torch.stack([torch.arange(d), torch.arange(d)])  # identity adjacency (self-loops)
        xx = torch.zeros(d, 2, dtype=torch.float64)
        s_dense = torch.eye(d, dtype=torch.float64)
        ref64 = double_recon_loss(xx, xx, s_dense, zt64 @ zt64.T, 0.0)  # structure only
        ex64 = exact_double_reconstruction_score(xx, xx, zt64, ei, weight=0.0,
                                                 backend="exact_sparse")
        zt32 = zt64.float()
        ex32 = exact_double_reconstruction_score(xx.float(), xx.float(), zt32, ei, weight=0.0,
                                                 backend="exact_sparse")
        ref32 = double_recon_loss(xx.float(), xx.float(), s_dense.float(), zt32 @ zt32.T, 0.0)
        f4.append({
            "eps": eps,
            "true_score_float64_dense": float(ref64[0]),
            "exact_float64": float(ex64[0]),
            "exact_float32": float(ex32[0]),
            "dense_float32": float(ref32[0]),
            "exact_f32_abs_err_vs_f64": float(abs(ex32[0].double() - ref64[0])),
            "dense_f32_abs_err_vs_f64": float(abs(ref32[0].double() - ref64[0])),
        })
    out["F4_float32_gram_cancellation"] = f4
    torch.set_default_dtype(torch.float32)
    return out


def main(out_path: str):
    env = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "torch_geometric": md.version("torch_geometric"),
        "pygod": md.version("pygod"),
        "cuda_used": False, "device": "cpu",
    }
    try:
        import torch_sparse  # noqa: F401
        env["torch_sparse"] = md.version("torch_sparse")
    except Exception as exc:  # noqa: BLE001
        env["torch_sparse"] = f"unavailable ({type(exc).__name__})"
    src_files = {
        "src/gog_fraud/models/pygod/exact_reconstruction.py": None,
        "src/gog_fraud/models/pygod/shared_reconstruction.py": None,
        "src/gog_fraud/models/pygod/stable_reconstruction.py": None,
        "src/gog_fraud/models/pygod/dlg_base.py": None,
        "src/gog_fraud/models/pygod/dlg_full_base.py": None,
        "src/gog_fraud/models/pygod/dlg.py": None,
        "src/gog_fraud/models/pygod/dlg_full.py": None,
        "src/gog_fraud/models/pygod/sparse_message.py": None,
        "projects/benchmark/scripts/a22_g1_dense_exact_training_equivalence.py": None,
    }
    for rel in src_files:
        src_files[rel] = sha256_file(os.path.join(REPO, rel))

    results, errors = [], []
    for family in ("DOMINANT", "DLG-Base", "DLG-Aug"):
        for dtype_name in ("float64", "float32"):
            for seed in SEEDS:
                try:
                    m = run_pair(family, seed, dtype_name, EPOCHS_TRAJ)
                    m["checks"] = passes(m)
                    m["pass"] = all(m["checks"].values())
                    results.append(m)
                    print(f"{family:9s} {dtype_name} seed={seed} pass={m['pass']} "
                          f"loss_rel={m['loss_rel']:.2e} grad_rel={m['grad_rel_l2']:.2e} "
                          f"step={m['step_param_max_abs']:.2e} "
                          f"traj={m.get('trajectory_loss_rel', float('nan')):.2e}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    torch.set_default_dtype(torch.float32)
                    errors.append({"family": family, "dtype": dtype_name, "seed": seed,
                                   "error": repr(exc), "trace": traceback.format_exc()})
                    print(f"{family} {dtype_name} seed={seed} ERROR {exc!r}", flush=True)

    func = {}
    try:
        func = function_level()
    except Exception as exc:  # noqa: BLE001
        errors.append({"stage": "function_level", "error": repr(exc),
                       "trace": traceback.format_exc()})
        print("function_level ERROR", repr(exc), flush=True)

    overall = bool(results) and not errors and all(r["pass"] for r in results)
    report = {
        "document_id": "DLG-BENCH-A22-G1-TRAINING-EQUIVALENCE",
        "environment": env,
        "predeclared_tolerances": TOL,
        "negative_control_factor": NEGATIVE_CONTROL_FACTOR,
        "fixture": {"N": N, "D": D, "hid_dim": HID, "num_layers": LAYERS, "lr": LR,
                    "weight": WEIGHT, "l1_dim": L1DIM, "trajectory_epochs": EPOCHS_TRAJ,
                    "seeds": SEEDS},
        "source_sha256": src_files,
        "installed_pygod_functional_sha256": sha256_file(
            sys.modules["pygod.nn.functional"].__file__),
        "results": results,
        "function_level": func,
        "errors": errors,
        "overall_pass_all_predeclared_checks": overall,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print("OVERALL", overall, "-> wrote", out_path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(REPO, "projects", "benchmark", "publication", "a22",
                      "a22_dense_exact_training_equivalence.json"))

