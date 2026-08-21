"""LANL Comprehensive Multi-Source Cyber-Security Events Graph Adapter.

Converts multi-source host/network event streams (authentication, process, flow, DNS)
and official red-team compromise events into a computer-level interaction PyG graph.
Zero label leakage: features are strictly multi-source activity statistics.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from gog_fraud.extensions.defense.defense_schema import DefenseManifest, sha256_tensor


class LanlRedTeamGraphBuilder:
    """Deterministic, leakage-safe computer-level interaction graph builder for LANL Cyber1."""

    def __init__(self, name: str = "LANL-RedTeam"):
        self.name = name
        self.computers: Dict[str, Dict[str, Any]] = {}
        self.comp_to_idx: Dict[str, int] = {}
        self.auth_edges: List[Tuple[int, int]] = []
        self.redteam_compromised_computers: Set[str] = set()

    def get_or_create_computer(self, comp_id: str) -> int:
        if comp_id not in self.comp_to_idx:
            idx = len(self.computers)
            self.comp_to_idx[comp_id] = idx
            self.computers[comp_id] = {
                "idx": idx,
                "comp_id": comp_id,
                "in_auth": 0,
                "out_auth": 0,
                "success_auth": 0,
                "failed_auth": 0,
                "users": set(),
                "src_peers": set(),
                "dst_peers": set(),
                "proc_starts": 0,
                "proc_stops": 0,
                "procs": set(),
                "flows_count": 0,
                "bytes_sent": 0,
                "bytes_recv": 0,
                "flow_peers": set(),
                "dns_queries": 0,
            }
        return self.comp_to_idx[comp_id]

    def add_auth_event(self, time_sec: int, user: str, src_comp: str, dst_comp: str,
                       auth_type: str, logon_type: str, auth_orientation: str, success: bool) -> None:
        src_idx = self.get_or_create_computer(src_comp)
        dst_idx = self.get_or_create_computer(dst_comp)

        self.auth_edges.append((src_idx, dst_idx))

        src_info = self.computers[src_comp]
        dst_info = self.computers[dst_comp]

        src_info["out_auth"] += 1
        dst_info["in_auth"] += 1

        if success:
            src_info["success_auth"] += 1
            dst_info["success_auth"] += 1
        else:
            src_info["failed_auth"] += 1
            dst_info["failed_auth"] += 1

        src_info["users"].add(user)
        dst_info["users"].add(user)

        src_info["dst_peers"].add(dst_idx)
        dst_info["src_peers"].add(src_idx)

    def add_process_event(self, time_sec: int, user: str, comp: str, process_name: str, is_start: bool) -> None:
        idx = self.get_or_create_computer(comp)
        c_info = self.computers[comp]
        if is_start:
            c_info["proc_starts"] += 1
        else:
            c_info["proc_stops"] += 1
        c_info["procs"].add(process_name)
        c_info["users"].add(user)

    def add_flow_event(self, time_sec: int, duration_sec: int, src_comp: str, src_port: int,
                       dst_comp: str, dst_port: int, protocol: int, byte_count: int, packet_count: int) -> None:
        src_idx = self.get_or_create_computer(src_comp)
        dst_idx = self.get_or_create_computer(dst_comp)

        src_info = self.computers[src_comp]
        dst_info = self.computers[dst_comp]

        src_info["flows_count"] += 1
        dst_info["flows_count"] += 1

        src_info["bytes_sent"] += max(0, byte_count)
        dst_info["bytes_recv"] += max(0, byte_count)

        src_info["flow_peers"].add(dst_idx)
        dst_info["flow_peers"].add(src_idx)

    def add_dns_event(self, time_sec: int, src_comp: str, resolved_comp: str) -> None:
        self.get_or_create_computer(src_comp)
        self.computers[src_comp]["dns_queries"] += 1

    def add_redteam_compromise(self, time_sec: int, user: str, src_comp: str, dst_comp: str) -> None:
        """Mark computer as destination/compromised target in official LANL redteam event."""
        self.get_or_create_computer(dst_comp)
        self.redteam_compromised_computers.add(dst_comp)

    def extract_features(self) -> np.ndarray:
        """Extract 16 leakage-safe multi-source statistics per computer.

        Feature schema (16 dims):
          0: log1p(incoming_auth_count)
          1: log1p(outgoing_auth_count)
          2: log1p(successful_auth_count)
          3: log1p(failed_auth_count)
          4: log1p(unique_users_count)
          5: log1p(unique_src_computers)
          6: log1p(unique_dest_computers)
          7: auth_success_ratio (success / (success + fail + 1e-6))
          8: log1p(process_start_count)
          9: log1p(process_stop_count)
          10: log1p(unique_processes_count)
          11: log1p(network_flows_count)
          12: log1p(bytes_sent)
          13: log1p(bytes_recv)
          14: log1p(unique_flow_peers)
          15: log1p(dns_queries_count)
        """
        n = len(self.computers)
        feat = np.zeros((n, 16), dtype=np.float32)

        for comp_id, info in self.computers.items():
            idx = info["idx"]

            succ = info["success_auth"]
            fail = info["failed_auth"]
            tot_auth = succ + fail

            feat[idx, 0] = math.log1p(info["in_auth"])
            feat[idx, 1] = math.log1p(info["out_auth"])
            feat[idx, 2] = math.log1p(succ)
            feat[idx, 3] = math.log1p(fail)
            feat[idx, 4] = math.log1p(len(info["users"]))
            feat[idx, 5] = math.log1p(len(info["src_peers"]))
            feat[idx, 6] = math.log1p(len(info["dst_peers"]))
            feat[idx, 7] = float(succ / (tot_auth + 1e-6)) if tot_auth > 0 else 1.0

            feat[idx, 8] = math.log1p(info["proc_starts"])
            feat[idx, 9] = math.log1p(info["proc_stops"])
            feat[idx, 10] = math.log1p(len(info["procs"]))

            feat[idx, 11] = math.log1p(info["flows_count"])
            feat[idx, 12] = math.log1p(info["bytes_sent"])
            feat[idx, 13] = math.log1p(info["bytes_recv"])
            feat[idx, 14] = math.log1p(len(info["flow_peers"]))
            feat[idx, 15] = math.log1p(info["dns_queries"])

        feat = np.nan_to_num(feat, nan=0.0, posinf=100.0, neginf=-100.0)
        return feat

    def build_pyg_data(self) -> Tuple[Data, DefenseManifest]:
        """Construct PyG Data object and metadata manifest."""
        n_nodes = len(self.computers)
        if n_nodes == 0:
            raise ValueError("No computers added to LANL graph builder.")

        # Features
        x = torch.from_numpy(self.extract_features()).float()

        # Edges (Authentication interaction network)
        if self.auth_edges:
            edge_arr = np.array(self.auth_edges, dtype=np.int64).T
            # Ensure unique edges
            unique_edges = np.unique(edge_arr, axis=1)
            edge_index = torch.from_numpy(unique_edges).long()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Labels: 1 for red-team compromised computers, 0 otherwise
        y = torch.zeros(n_nodes, dtype=torch.long)
        for comp_id in self.redteam_compromised_computers:
            if comp_id in self.comp_to_idx:
                y[self.comp_to_idx[comp_id]] = 1

        n_pos = int(y.sum().item())
        n_neg = n_nodes - n_pos

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = n_nodes
        data.dataset_name = "LANL-RedTeam"

        manifest = DefenseManifest(
            dataset_name="LANL-RedTeam",
            official_source_name="Comprehensive Multi-Source Cyber-Security Events (Los Alamos National Laboratory)",
            source_citation="Kent, A. D. (2015). Cybersecurity Data Sources. Los Alamos National Laboratory. DOI: 10.17021/1179829",
            source_url_or_doi="https://doi.org/10.17021/1179829",
            provenance_details=(
                "58-day comprehensive enterprise network telemetry containing authentication, process, flow, and DNS events. "
                "Ground truth: redteam.txt known compromise events executed by internal red-team operations."
            ),
            num_nodes=n_nodes,
            num_edges=int(edge_index.size(1)),
            num_features=int(x.size(1)),
            num_positives=n_pos,
            num_negatives=n_neg,
            positive_ratio=float(n_pos / n_nodes) if n_nodes > 0 else 0.0,
            time_range_description="58 consecutive days enterprise collection window",
            node_definition="Computer entities (e.g. workstation, server, domain controller).",
            edge_definition="Directed authentication interactions (source computer -> destination computer).",
            ground_truth_definition="Computers targeted and compromised in official redteam.txt ground truth records.",
            negative_label_semantics="0 = not identified as red-team compromise in official redteam.txt (not certified benign).",
            graph_sha256=sha256_tensor(edge_index),
            feature_sha256=sha256_tensor(x),
            label_sha256=sha256_tensor(y),
            split_strategy="stratified_node_transductive",
            is_temporal=False,
            metadata={
                "feature_names": [
                    "in_auth_log1p", "out_auth_log1p", "success_auth_log1p", "failed_auth_log1p",
                    "unique_users_log1p", "unique_src_comp_log1p", "unique_dst_comp_log1p",
                    "auth_success_ratio", "proc_starts_log1p", "proc_stops_log1p", "unique_procs_log1p",
                    "flows_count_log1p", "bytes_sent_log1p", "bytes_recv_log1p",
                    "unique_flow_peers_log1p", "dns_queries_log1p"
                ]
            }
        )

        return data, manifest
