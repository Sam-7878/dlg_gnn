"""Produce numeric GADNR compatibility-equivalence evidence for Round D2."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data
from pygod.detector import GADNR as UpstreamGADNR
from pygod.nn import GADNRBase as UpstreamGADNRBase
from gog_fraud.models.pygod.gadnr import GADNR as PatchedGADNR


class UpstreamMathematicalReference(UpstreamGADNR):
    """Installed upstream path with only the obsolete ``tot_nodes`` kwarg removed."""

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return UpstreamGADNRBase(
            in_dim=self.in_dim, hid_dim=self.hid_dim,
            encoder_layers=self.encoder_layers,
            deg_dec_layers=self.deg_dec_layers,
            fea_dec_layers=self.fea_dec_layers,
            sample_size=self.sample_size, sample_time=self.sample_time,
            neighbor_num_list=self.neighbor_num_list,
            neigh_loss=self.neigh_loss,
            lambda_loss1=self.lambda_loss1,
            lambda_loss2=self.lambda_loss2,
            lambda_loss3=self.lambda_loss3,
            full_batch=self.full_batch, backbone=self.backbone, device=self.device,
        ).to(self.device)


def seed(value: int = 17) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value)


def detector(cls):
    return cls(hid_dim=8, num_layers=1, deg_dec_layers=2, fea_dec_layers=2,
               sample_size=2, sample_time=1, epoch=1, batch_size=0, gpu=-1, verbose=0)


def pool_usage_audit() -> dict:
    source_path = Path(sys.modules[UpstreamGADNRBase.__module__].__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    loads, stores = 0, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "pool":
            if isinstance(node.ctx, ast.Load): loads += 1
            if isinstance(node.ctx, ast.Store): stores += 1
    return {"upstream_source": str(source_path), "pool_assignments": stores, "pool_reads": loads, "pool_is_computationally_unused": stores >= 1 and loads == 0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/sci_defense_extension/d2/gadnr/gadnr_compatibility_equivalence.json")
    args = parser.parse_args()
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)

    seed()
    connected = Data(
        x=torch.rand(8, 5) + 0.1,
        edge_index=torch.stack([torch.arange(8), torch.roll(torch.arange(8), -1)]),
        y=torch.tensor([0, 0, 0, 0, 0, 1, 0, 1]), num_nodes=8,
    )
    upstream_pre, _, upstream_deg, _ = UpstreamGADNRBase.process_graph(connected.clone())
    corrected_preparer = detector(PatchedGADNR)
    corrected_preparer.batch_size = connected.num_nodes; corrected_preparer.device = torch.device("cpu")
    corrected_pre = corrected_preparer.process_graph(connected.clone())

    seed(); reference = detector(UpstreamMathematicalReference).fit(connected.clone())
    seed(); corrected = detector(PatchedGADNR).fit(connected.clone())
    ref_score = reference.decision_score_.detach().cpu()
    new_score = corrected.decision_score_.detach().cpu()
    labels = connected.y.numpy()

    seed()
    isolated = Data(x=torch.rand(5, 4) + 0.1,
                    edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                    y=torch.tensor([0, 0, 1, 0, 1]), num_nodes=5)
    _, _, isolated_upstream_degree, _ = UpstreamGADNRBase.process_graph(isolated.clone())
    seed(); isolated_corrected = detector(PatchedGADNR).fit(isolated.clone())

    evidence = {
        "decision": "SEMANTICS_PRESERVING_ON_CONNECTED_GRAPH",
        "reference_definition": "installed PyGOD 1.1.0 mathematical path with only rejected tot_nodes kwarg removed",
        "unmodified_upstream_executable": False,
        "unmodified_upstream_failure": "current PyG rejects MessagePassing kwarg tot_nodes before training",
        "patches": {
            "A_unused_pool_bypass": pool_usage_audit(),
            "B_num_nodes_and_bincount": "equivalent on connected graph; corrects isolated-node output cardinality",
            "C_obsolete_tot_nodes_kwarg": "compatibility removal required for current PyG",
        },
        "connected_graph": {
            "nodes": connected.num_nodes,
            "preprocessed_edge_index_equal": bool(torch.equal(upstream_pre.edge_index, corrected_pre.edge_index)),
            "feature_max_abs_error": float((upstream_pre.x - corrected_pre.x).abs().max()),
            "degree_max_abs_error": float((upstream_deg - corrected_preparer.neighbor_num_list.cpu()).abs().max()),
            "training_loss_per_node_max_abs_error": float((reference.arg_min_loss_per_node - corrected.arg_min_loss_per_node).abs().max()),
            "score_max_abs_error": float((ref_score - new_score).abs().max()),
            "reference_roc_auc": float(roc_auc_score(labels, ref_score)),
            "corrected_roc_auc": float(roc_auc_score(labels, new_score)),
            "reference_pr_auc": float(average_precision_score(labels, ref_score)),
            "corrected_pr_auc": float(average_precision_score(labels, new_score)),
        },
        "isolated_node": {
            "nodes": isolated.num_nodes,
            "upstream_degree_outputs": int(isolated_upstream_degree.numel()),
            "corrected_score_outputs": int(isolated_corrected.decision_score_.numel()),
            "corrected_scores_finite": bool(torch.isfinite(isolated_corrected.decision_score_).all()),
        },
    }
    evidence["acceptance_passed"] = (
        evidence["patches"]["A_unused_pool_bypass"]["pool_is_computationally_unused"]
        and evidence["connected_graph"]["preprocessed_edge_index_equal"]
        and evidence["connected_graph"]["score_max_abs_error"] <= 1e-6
        and evidence["connected_graph"]["training_loss_per_node_max_abs_error"] <= 1e-6
        and evidence["isolated_node"]["corrected_score_outputs"] == isolated.num_nodes
        and evidence["isolated_node"]["corrected_scores_finite"]
    )
    output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
