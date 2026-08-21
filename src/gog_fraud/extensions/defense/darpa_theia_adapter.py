"""DARPA-TC-THEIA Engagement 3 Provenance Graph Adapter.

Converts system provenance stream (Processes, Files, Network Sockets) and official
ground-truth APT attack events into a canonical PyG graph artifact.
Zero label leakage: features are strictly topological and activity count statistics.
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


# Node types
NODE_TYPE_PROCESS = 0
NODE_TYPE_FILE = 1
NODE_TYPE_SOCKET = 2
NODE_TYPE_OTHER = 3
NODE_TYPE_NAMES = ["Process", "File", "Socket", "Other"]

# Event types in CDM
EVENT_TYPE_SPAWN = "EVENT_FORK_CLONE_EXEC"
EVENT_TYPE_READ = "EVENT_READ"
EVENT_TYPE_WRITE = "EVENT_WRITE"
EVENT_TYPE_SEND = "EVENT_SENDTO_WRITE"
EVENT_TYPE_RECV = "EVENT_RECVFROM_READ"
EVENT_TYPE_CONNECT = "EVENT_CONNECT"


class DarpaTheiaGraphBuilder:
    """Deterministic, leakage-safe graph builder for DARPA TC E3 THEIA stream."""

    def __init__(self, name: str = "DARPA-TC-THEIA", topic: str = "ta1-theia-e3-official-1r"):
        self.name = name
        self.topic = topic
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.node_to_idx: Dict[str, int] = {}
        self.edges: List[Tuple[int, int]] = []
        self.ground_truth_malicious_ids: Set[str] = set()

    def add_node(self, node_id: str, node_type: int, name: str = "") -> int:
        if node_id not in self.node_to_idx:
            idx = len(self.nodes)
            self.node_to_idx[node_id] = idx
            self.nodes[node_id] = {
                "idx": idx,
                "type": node_type,
                "name": name,
                "in_degree": 0,
                "out_degree": 0,
                "read_count": 0,
                "write_count": 0,
                "spawn_count": 0,
                "net_send_count": 0,
                "net_recv_count": 0,
                "peers": set(),
                "event_types": set(),
                "timestamps": [],
            }
        return self.node_to_idx[node_id]

    def add_event(self, src_id: str, dst_id: str, src_type: int, dst_type: int,
                  event_type: str, timestamp: int, is_malicious_ground_truth: bool = False) -> None:
        src_idx = self.add_node(src_id, src_type)
        dst_idx = self.add_node(dst_id, dst_type)

        self.edges.append((src_idx, dst_idx))

        src_node = self.nodes[src_id]
        dst_node = self.nodes[dst_id]

        src_node["out_degree"] += 1
        dst_node["in_degree"] += 1

        src_node["peers"].add(dst_idx)
        dst_node["peers"].add(src_idx)

        src_node["event_types"].add(event_type)
        dst_node["event_types"].add(event_type)

        if timestamp > 0:
            src_node["timestamps"].append(timestamp)
            dst_node["timestamps"].append(timestamp)

        # Update event-specific counts
        if "READ" in event_type:
            src_node["read_count"] += 1
        elif "WRITE" in event_type:
            src_node["write_count"] += 1
        elif "FORK" in event_type or "CLONE" in event_type or "EXEC" in event_type or "SPAWN" in event_type:
            src_node["spawn_count"] += 1
        elif "SEND" in event_type or "WRITE" in event_type:
            src_node["net_send_count"] += 1
        elif "RECV" in event_type or "READ" in event_type:
            dst_node["net_recv_count"] += 1

        if is_malicious_ground_truth:
            self.ground_truth_malicious_ids.add(src_id)
            self.ground_truth_malicious_ids.add(dst_id)

    def mark_ground_truth_entity(self, entity_id: str) -> None:
        """Mark ground truth entity identified in DARPA official APT report."""
        if entity_id in self.node_to_idx:
            self.ground_truth_malicious_ids.add(entity_id)

    def extract_features(self) -> np.ndarray:
        """Extract 16 leakage-safe topological and activity count features per node.
        
        Feature schema (16 dims):
          0-3: Node type one-hot (Process, File, Socket, Other)
          4: log1p(in_degree)
          5: log1p(out_degree)
          6: log1p(read_count)
          7: log1p(write_count)
          8: log1p(spawn_count)
          9: log1p(net_send_count)
          10: log1p(net_recv_count)
          11: log1p(unique_counterparts)
          12: log1p(unique_event_types)
          13: log1p(total_events)
          14: log1p(active_duration_sec)
          15: log1p(event_frequency_per_hour)
        """
        n = len(self.nodes)
        feat = np.zeros((n, 16), dtype=np.float32)

        for node_id, info in self.nodes.items():
            idx = info["idx"]
            ntype = info["type"]
            # One-hot
            if 0 <= ntype <= 3:
                feat[idx, ntype] = 1.0
            else:
                feat[idx, 3] = 1.0

            in_deg = info["in_degree"]
            out_deg = info["out_degree"]
            tot_events = in_deg + out_deg

            feat[idx, 4] = math.log1p(in_deg)
            feat[idx, 5] = math.log1p(out_deg)
            feat[idx, 6] = math.log1p(info["read_count"])
            feat[idx, 7] = math.log1p(info["write_count"])
            feat[idx, 8] = math.log1p(info["spawn_count"])
            feat[idx, 9] = math.log1p(info["net_send_count"])
            feat[idx, 10] = math.log1p(info["net_recv_count"])
            feat[idx, 11] = math.log1p(len(info["peers"]))
            feat[idx, 12] = math.log1p(len(info["event_types"]))
            feat[idx, 13] = math.log1p(tot_events)

            ts = info["timestamps"]
            if len(ts) >= 2:
                dur = max(1, max(ts) - min(ts))
                freq = (tot_events / dur) * 3600.0
            else:
                dur = 1
                freq = float(tot_events)

            feat[idx, 14] = math.log1p(dur)
            feat[idx, 15] = math.log1p(freq)

        # Sanity check: replace non-finite
        feat = np.nan_to_num(feat, nan=0.0, posinf=100.0, neginf=-100.0)
        return feat

    def build_pyg_data(self) -> Tuple[Data, DefenseManifest]:
        """Construct PyG Data object and metadata manifest."""
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            raise ValueError("No nodes added to graph builder.")

        # Features
        x = torch.from_numpy(self.extract_features()).float()

        # Edges
        if self.edges:
            edge_arr = np.array(self.edges, dtype=np.int64).T
            # Ensure unique edges
            unique_edges = np.unique(edge_arr, axis=1)
            edge_index = torch.from_numpy(unique_edges).long()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Labels: 1 for ground truth attack entity, 0 otherwise
        y = torch.zeros(n_nodes, dtype=torch.long)
        for node_id in self.ground_truth_malicious_ids:
            if node_id in self.node_to_idx:
                y[self.node_to_idx[node_id]] = 1

        n_pos = int(y.sum().item())
        n_neg = n_nodes - n_pos

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = n_nodes
        data.dataset_name = "DARPA-TC-THEIA"

        # Manifest
        manifest = DefenseManifest(
            dataset_name="DARPA-TC-THEIA",
            official_source_name="DARPA Transparent Computing Engagement 3 (THEIA Stream)",
            source_citation="DARPA Transparent Computing Program, Engagement #3 Release (2018)",
            source_url_or_doi="https://github.com/darpa-i2o/transparent-computing",
            provenance_details=(
                f"Engagement 3 official topic: {self.topic}. "
                "Entities: Process, File, Network Socket. Edges: causal system calls (fork, read, write, connect, send, recv). "
                "Ground truth: ACT attack scenarios (Firefox backdoor, privilege escalation, data exfiltration)."
            ),
            num_nodes=n_nodes,
            num_edges=int(edge_index.size(1)),
            num_features=int(x.size(1)),
            num_positives=n_pos,
            num_negatives=n_neg,
            positive_ratio=float(n_pos / n_nodes) if n_nodes > 0 else 0.0,
            time_range_description="DARPA TC Engagement 3 execution period (April 2018)",
            node_definition="Causal provenance entities: Process (type 0), File (type 1), Socket/Network (type 2), Other (type 3).",
            edge_definition="Directed system events (Process->File, Process->Process, Process->Socket).",
            ground_truth_definition="Entities directly participating in ground-truth APT/malicious activities reported in official DARPA E3 ground truth.",
            negative_label_semantics="0 = not identified as attack in official DARPA E3 ground-truth report (not certified benign).",
            graph_sha256=sha256_tensor(edge_index),
            feature_sha256=sha256_tensor(x),
            label_sha256=sha256_tensor(y),
            split_strategy="stratified_node_transductive",
            is_temporal=False,
            metadata={
                "topic": self.topic,
                "node_types": NODE_TYPE_NAMES,
                "feature_names": [
                    "is_process", "is_file", "is_socket", "is_other",
                    "in_degree_log1p", "out_degree_log1p",
                    "read_count_log1p", "write_count_log1p", "spawn_count_log1p",
                    "net_send_count_log1p", "net_recv_count_log1p",
                    "unique_peers_log1p", "unique_event_types_log1p", "total_events_log1p",
                    "active_duration_log1p", "event_frequency_log1p"
                ]
            }
        )

        return data, manifest
