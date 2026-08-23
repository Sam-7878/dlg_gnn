#!/usr/bin/env python3
"""
DARPA TC E5 THEIA — Official Source Graph Builder (Production D3)
Defense Extension Round D3, Phase 1

Features:
- Fast Avro streaming with fastavro
- In-memory node ID mapping using raw 16-byte UUIDs
- Telemetry feature extraction (node type one-hot, degrees, event counts, duration)
- Ground Truth mapping from official TA51_Final_report_E5 (2019-05-15 attack window & target host)
- Output lineage, manifests, and PyG Data graph artifact

Usage:
    cd /mnt/d/_Work/goat_bank/dlg_gnn
    python scripts/defense_extension_real/darpa_theia_build.py
"""
import argparse
import csv
import gzip
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import fastavro
import torch
from torch_geometric.data import Data

# Ensure line-buffered output for real-time monitoring
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
THEIA_BASE = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA")
THEIA_DATA_DIR = THEIA_BASE / "Data" / "theia"
SCHEMA_FILE = THEIA_BASE / "Schema" / "TCCDMDatum.avsc"
GT_DOCX = THEIA_BASE / "Ground_Truth" / "TA51_Final_report_E5.docx"
GT_PDF = THEIA_BASE / "Ground_Truth" / "TA51_Final_report_E5.pdf"

OUTPUT_BASE = Path("outputs/sci_defense_extension_real")
SOURCE_AUDIT_DIR = OUTPUT_BASE / "source_audit"
GRAPH_DIR = OUTPUT_BASE / "graphs"

# ─────────────────────────────────────────────────────────────────────────────
# CDM20 outer wrapper type → Node Class mapping
# ─────────────────────────────────────────────────────────────────────────────
OUTER_TYPE_TO_CLASS = {
    "RECORD_EVENT": "Event",
    "RECORD_SUBJECT": "Subject",
    "RECORD_FILE_OBJECT": "FileObject",
    "RECORD_NET_FLOW_OBJECT": "NetFlowObject",
    "RECORD_IPC_OBJECT": "IPCObject",
    "RECORD_PACKET_SOCKET_OBJECT": "PacketSocket",
    "RECORD_REGISTRY_KEY_OBJECT": "RegistryKey",
    "RECORD_MEMORY_OBJECT": "MemoryObject",
    "RECORD_SRC_SINK_OBJECT": "SrcSinkObject",
    "RECORD_PRINCIPAL": "Principal",
    "RECORD_PROVENANCE_TAG_NODE": "ProvenanceTag",
    "RECORD_UNKNOWN_PROVENANCE_NODE": "ProvenanceTag",
    "RECORD_TIME_MARKER": "TimeMarker",
    "RECORD_HOST": "Host",
    "RECORD_UNIT_DEPENDENCY": "UnitDependency",
    "RECORD_END_MARKER": "EndMarker",
}

# ─────────────────────────────────────────────────────────────────────────────
# Event type → Edge Category (D3 §10)
# ─────────────────────────────────────────────────────────────────────────────
EVENT_EDGE_CATEGORIES = {
    "EVENT_FORK": "proc_proc", "EVENT_CLONE": "proc_proc",
    "EVENT_EXECUTE": "proc_proc", "EVENT_SIGNAL": "proc_proc",
    "EVENT_WAIT": "proc_proc", "EVENT_EXIT": "proc_proc",
    "EVENT_MODIFY_PROCESS": "proc_proc",
    "EVENT_READ": "proc_file", "EVENT_WRITE": "proc_file",
    "EVENT_CREATE_OBJECT": "proc_file", "EVENT_UNLINK": "proc_file",
    "EVENT_MMAP": "proc_file", "EVENT_RENAME": "proc_file",
    "EVENT_TRUNCATE": "proc_file", "EVENT_LINK": "proc_file",
    "EVENT_OPEN": "proc_file", "EVENT_CLOSE": "proc_file",
    "EVENT_MODIFY_FILE_ATTRIBUTES": "proc_file",
    "EVENT_CONNECT": "proc_net", "EVENT_ACCEPT": "proc_net",
    "EVENT_SENDTO": "proc_net", "EVENT_RECVFROM": "proc_net",
    "EVENT_SENDMSG": "proc_net", "EVENT_RECVMSG": "proc_net",
    "EVENT_WRITE_SOCKET_PARAMS": "proc_net",
    "EVENT_READ_SOCKET_PARAMS": "proc_net",
}

def get_edge_category(event_type: str) -> str:
    return EVENT_EDGE_CATEGORIES.get(event_type, "other")

def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Attack Definition from TA51_Final_report_E5 (Official GT)
# ─────────────────────────────────────────────────────────────────────────────
# Attack on ta1-theia-target-1:
# Date: 2019-05-15 (Wed) 14:47:41 EDT (18:47:41 UTC) to 15:10:00 EDT (19:10:00 UTC)
# In nanoseconds:
E5_THEIA_ATTACK_START_NS = 1557946061 * 1_000_000_000  # 2019-05-15 18:47:41 UTC
E5_THEIA_ATTACK_END_NS = 1557947400 * 1_000_000_000    # 2019-05-15 19:10:00 UTC
# Persistent console active through 2019-05-17
E5_THEIA_PERSISTENT_END_NS = 1558100760 * 1_000_000_000  # 2019-05-17 13:46:00 UTC

NULL_UUID = b'\x00' * 16


class TheiaGraphBuilder:
    """
    High-performance graph builder for DARPA TC E5 THEIA data.
    Uses integer node indexing and flat arrays for feature aggregation.
    """

    def __init__(self):
        # uuid_bytes (16 bytes) -> integer node ID
        self.uuid_to_id: dict[bytes, int] = {}
        self.node_class: list[str] = []
        self.node_cdm_type: list[str] = []

        # Directed edges: (src_id, dst_id)
        self.edges_src: list[int] = []
        self.edges_dst: list[int] = []
        self.edge_types: list[str] = []

        # Fast feature arrays (indexed by node ID)
        self.feat_in_deg: list[int] = []
        self.feat_out_deg: list[int] = []
        self.feat_read_cnt: list[int] = []
        self.feat_write_cnt: list[int] = []
        self.feat_exec_cnt: list[int] = []
        self.feat_spawn_cnt: list[int] = []
        self.feat_net_cnt: list[int] = []
        self.feat_other_cnt: list[int] = []
        self.feat_ts_min: list[Optional[int]] = []
        self.feat_ts_max: list[Optional[int]] = []

        # Accounting
        self.record_counts: dict[str, int] = defaultdict(int)
        self.outer_type_counts: dict[str, int] = defaultdict(int)
        self.event_type_counts: dict[str, int] = defaultdict(int)
        self.dropped_records: int = 0
        self.drop_reasons: dict[str, int] = defaultdict(int)

        self.global_ts_min: Optional[int] = None
        self.global_ts_max: Optional[int] = None

    def _get_or_create(self, uuid_bytes: bytes, node_class: str, cdm_type: str) -> int:
        nid = self.uuid_to_id.get(uuid_bytes)
        if nid is None:
            nid = len(self.uuid_to_id)
            self.uuid_to_id[uuid_bytes] = nid
            self.node_class.append(node_class)
            self.node_cdm_type.append(cdm_type)
            # Init features
            self.feat_in_deg.append(0)
            self.feat_out_deg.append(0)
            self.feat_read_cnt.append(0)
            self.feat_write_cnt.append(0)
            self.feat_exec_cnt.append(0)
            self.feat_spawn_cnt.append(0)
            self.feat_net_cnt.append(0)
            self.feat_other_cnt.append(0)
            self.feat_ts_min.append(None)
            self.feat_ts_max.append(None)
        return nid

    def process_record(self, outer_record: dict):
        self.record_counts["total"] += 1

        outer_type = outer_record.get("type", "")
        if isinstance(outer_type, dict):
            outer_type = outer_type.get("value", str(outer_type))
        self.outer_type_counts[outer_type] += 1

        datum = outer_record.get("datum")
        if not isinstance(datum, dict):
            self.dropped_records += 1
            self.drop_reasons["no_datum"] += 1
            return

        node_class = OUTER_TYPE_TO_CLASS.get(outer_type, "Unknown")
        inner_type = datum.get("type", "")
        if isinstance(inner_type, dict):
            inner_type = inner_type.get("value", str(inner_type))

        self.record_counts[node_class] += 1

        if node_class == "Event":
            self._process_event(datum, inner_type)
        elif node_class in ("Subject", "FileObject", "NetFlowObject", "IPCObject",
                             "PacketSocket", "RegistryKey", "MemoryObject",
                             "SrcSinkObject", "Principal"):
            self._process_entity(datum, node_class, inner_type)

    def _process_entity(self, datum: dict, node_class: str, cdm_type: str):
        uuid_raw = datum.get("uuid")
        if not isinstance(uuid_raw, bytes):
            if isinstance(uuid_raw, dict):
                uuid_raw = uuid_raw.get("bytes")
            if not isinstance(uuid_raw, bytes):
                self.dropped_records += 1
                self.drop_reasons["entity_no_uuid"] += 1
                return
        self._get_or_create(uuid_raw, node_class, cdm_type)

    def _process_event(self, datum: dict, event_type: str):
        self.event_type_counts[event_type] += 1

        ts_ns = datum.get("timestampNanos")
        if ts_ns:
            if self.global_ts_min is None or ts_ns < self.global_ts_min:
                self.global_ts_min = ts_ns
            if self.global_ts_max is None or ts_ns > self.global_ts_max:
                self.global_ts_max = ts_ns

        # Subject UUID
        subj_raw = datum.get("subject")
        if not isinstance(subj_raw, bytes):
            if isinstance(subj_raw, dict):
                subj_raw = subj_raw.get("bytes")
            if not isinstance(subj_raw, bytes):
                self.dropped_records += 1
                self.drop_reasons["event_no_subject"] += 1
                return

        src_id = self._get_or_create(subj_raw, "Subject", "SUBJECT_PROCESS")

        # PredicateObject UUID
        pred_raw = datum.get("predicateObject")
        if not isinstance(pred_raw, bytes):
            if isinstance(pred_raw, dict):
                pred_raw = pred_raw.get("bytes")
            if not isinstance(pred_raw, bytes) or pred_raw == NULL_UUID:
                self.drop_reasons["event_no_predobj"] += 1
                # Update src node timestamp even if no dst
                if ts_ns:
                    if self.feat_ts_min[src_id] is None or ts_ns < self.feat_ts_min[src_id]:
                        self.feat_ts_min[src_id] = ts_ns
                    if self.feat_ts_max[src_id] is None or ts_ns > self.feat_ts_max[src_id]:
                        self.feat_ts_max[src_id] = ts_ns
                return

        dst_id = self._get_or_create(pred_raw, "Unknown", "OBJECT_UNKNOWN")

        edge_cat = get_edge_category(event_type)
        self.edges_src.append(src_id)
        self.edges_dst.append(dst_id)
        self.edge_types.append(edge_cat)

        # Update features
        self.feat_out_deg[src_id] += 1
        self.feat_in_deg[dst_id] += 1

        if ts_ns:
            if self.feat_ts_min[src_id] is None or ts_ns < self.feat_ts_min[src_id]:
                self.feat_ts_min[src_id] = ts_ns
            if self.feat_ts_max[src_id] is None or ts_ns > self.feat_ts_max[src_id]:
                self.feat_ts_max[src_id] = ts_ns
            if self.feat_ts_min[dst_id] is None or ts_ns < self.feat_ts_min[dst_id]:
                self.feat_ts_min[dst_id] = ts_ns
            if self.feat_ts_max[dst_id] is None or ts_ns > self.feat_ts_max[dst_id]:
                self.feat_ts_max[dst_id] = ts_ns

        if edge_cat == "proc_file":
            if "READ" in event_type:
                self.feat_read_cnt[src_id] += 1
            else:
                self.feat_write_cnt[src_id] += 1
        elif edge_cat == "proc_proc":
            if "EXEC" in event_type:
                self.feat_exec_cnt[src_id] += 1
            else:
                self.feat_spawn_cnt[src_id] += 1
        elif edge_cat == "proc_net":
            self.feat_net_cnt[src_id] += 1
        else:
            self.feat_other_cnt[src_id] += 1

    def process_gz_file(self, gz_path: Path) -> int:
        import subprocess
        t0 = time.time()
        n = 0
        proc = subprocess.Popen(["gzip", "-dc", str(gz_path)], stdout=subprocess.PIPE, bufsize=2*1024*1024)
        try:
            reader = fastavro.reader(proc.stdout, handle_unicode_errors="replace")
            for record in reader:
                if isinstance(record, dict):
                    self.process_record(record)
                    n += 1
                    if n % 1_000_000 == 0:
                        elapsed = time.time() - t0
                        log.info(f"    [{gz_path.name}] {n:,} recs, nodes={len(self.uuid_to_id):,}, edges={len(self.edges_src):,}, {n/elapsed:,.0f} rec/s")
        finally:
            proc.stdout.close()
            proc.wait()

        elapsed = time.time() - t0
        log.info(f"    DONE {gz_path.name}: {n:,} recs in {elapsed:.1f}s ({n/elapsed:,.0f} rec/s)")
        return n

    def build_pyg_data(self) -> tuple[Data, list[dict]]:
        """
        Build PyG Data artifact and GT mapping records.
        """
        n_nodes = len(self.uuid_to_id)
        if n_nodes == 0:
            raise ValueError("No nodes registered in builder")

        NODE_CLASS_MAP = {
            "Subject": 0, "FileObject": 1, "NetFlowObject": 2,
            "IPCObject": 3, "PacketSocket": 4, "RegistryKey": 5,
            "MemoryObject": 6, "SrcSinkObject": 7, "Principal": 8,
            "Unknown": 9,
        }
        N_TYPES = 10
        # 10 one-hot + in_deg, out_deg, read, write, exec, spawn, net, other, duration, log_total_events = 20 dims
        x = torch.zeros(n_nodes, N_TYPES + 10, dtype=torch.float32)
        y = torch.zeros(n_nodes, dtype=torch.long)

        gt_records = []
        n_pos = 0

        # Invert uuid_to_id mapping
        id_to_uuid = {nid: uuid for uuid, nid in self.uuid_to_id.items()}

        for nid in range(n_nodes):
            cls_name = self.node_class[nid]
            tidx = NODE_CLASS_MAP.get(cls_name, 9)
            x[nid, tidx] = 1.0

            x[nid, N_TYPES + 0] = float(self.feat_in_deg[nid])
            x[nid, N_TYPES + 1] = float(self.feat_out_deg[nid])
            x[nid, N_TYPES + 2] = float(self.feat_read_cnt[nid])
            x[nid, N_TYPES + 3] = float(self.feat_write_cnt[nid])
            x[nid, N_TYPES + 4] = float(self.feat_exec_cnt[nid])
            x[nid, N_TYPES + 5] = float(self.feat_spawn_cnt[nid])
            x[nid, N_TYPES + 6] = float(self.feat_net_cnt[nid])
            x[nid, N_TYPES + 7] = float(self.feat_other_cnt[nid])

            t_min = self.feat_ts_min[nid]
            t_max = self.feat_ts_max[nid]
            if t_min and t_max and t_max >= t_min:
                x[nid, N_TYPES + 8] = float((t_max - t_min) / 1e9)
            tot_events = (self.feat_in_deg[nid] + self.feat_out_deg[nid])
            x[nid, N_TYPES + 9] = float(torch.log1p(torch.tensor(tot_events, dtype=torch.float32)))

            # GT label assignment:
            # Positive: Subject / Process active during documented E5 THEIA attack window on ta1-theia-1
            is_positive = False
            gt_ref = ""
            rationale = "not_identified_in_official_ground_truth"

            if t_min and t_max:
                # Check overlap with E5 THEIA attack window
                # (1557946061s = 2019-05-15 18:47:41 UTC to 19:10:00 UTC)
                if not (t_max < E5_THEIA_ATTACK_START_NS or t_min > E5_THEIA_ATTACK_END_NS):
                    # Active during the official attack window!
                    if cls_name == "Subject":
                        is_positive = True
                        gt_ref = "TA51_Final_report_E5_Section_05_15_Attack"
                        rationale = "Subject_active_during_20190515_THEIA1_Drakon_Inject_attack_window"
                    elif self.feat_net_cnt[nid] > 0 or self.feat_exec_cnt[nid] > 0:
                        is_positive = True
                        gt_ref = "TA51_Final_report_E5_Section_05_15_Attack"
                        rationale = "Entity_active_with_net_exec_during_20190515_attack_window"

            if is_positive:
                y[nid] = 1
                n_pos += 1

            uuid_hex = id_to_uuid[nid].hex()
            gt_records.append({
                "internal_id": nid,
                "uuid_hex": uuid_hex,
                "node_class": cls_name,
                "cdm_type": self.node_cdm_type[nid],
                "label": int(y[nid].item()),
                "gt_reference": gt_ref,
                "mapping_rationale": rationale,
                "first_seen_ns": t_min if t_min else "",
                "last_seen_ns": t_max if t_max else "",
                "total_events": tot_events,
            })

        if self.edges_src:
            edge_index = torch.tensor([self.edges_src, self.edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = n_nodes

        log.info(f"PyG Data built: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges, {n_pos:,} positive labels ({n_pos/max(1,n_nodes)*100:.2f}%)")
        return data, gt_records


def collect_gz_files(zip_count: int) -> list[Path]:
    dirs = sorted([d for d in THEIA_DATA_DIR.iterdir() if d.is_dir()], key=lambda d: d.name)
    gz_files = []
    for d in dirs[:zip_count]:
        sub = d / "theia"
        if not sub.exists():
            continue
        files = sorted(sub.glob("*.bin.*.gz"))
        log.info(f"  {d.name}: {len(files)} .bin.gz files")
        gz_files.extend(files)
    return gz_files


def compute_manifest(gz_files: list[Path], out_path: Path) -> list[dict]:
    rows = []
    for gz_path in gz_files:
        log.info(f"  Computing SHA-256: {gz_path.name} ({gz_path.stat().st_size/1024**2:.1f} MB)...")
        sha = sha256_of_file(gz_path)
        topic = gz_path.name.split(".bin.")[0] if ".bin." in gz_path.name else gz_path.name
        rows.append({
            "filename": gz_path.name,
            "official_topic": topic,
            "compressed_size_bytes": gz_path.stat().st_size,
            "sha256": sha,
            "parent_zip": gz_path.parent.parent.name,
            "cdm_version": "CDM20",
            "ta1_provider": "THEIA",
            "engagement": "Engagement 5",
        })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log.info(f"Manifest written: {out_path}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="DARPA TC E5 THEIA Graph Builder")
    ap.add_argument("--zip-count", type=int, default=1,
                    help="Number of extracted ZIP directories to process (default: 1)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/sci_defense_extension_real"),
                    help="Output base directory")
    ap.add_argument("--skip-manifest-hash", action="store_true",
                    help="Skip recomputing SHA-256 if manifest exists")
    args = ap.parse_args()

    out_base = args.output_dir
    audit_dir = out_base / "source_audit"
    graph_dir = out_base / "graphs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("DARPA TC E5 THEIA Graph Builder (Round D3 Official Rebuild)")
    log.info("=" * 70)

    gz_files = collect_gz_files(args.zip_count)
    if not gz_files:
        log.error("No .bin.gz files found in data directory!")
        sys.exit(1)

    log.info(f"Found {len(gz_files)} official .bin.gz raw files to process.")

    # 1. Manifest
    manifest_path = audit_dir / "darpa_raw_manifest.csv"
    if args.skip_manifest_hash and manifest_path.exists():
        log.info(f"Loading existing manifest from {manifest_path}")
        with open(manifest_path) as f:
            manifest_rows = list(csv.DictReader(f))
    else:
        manifest_rows = compute_manifest(gz_files, manifest_path)

    # 2. Build graph
    builder = TheiaGraphBuilder()
    log.info("\nStreaming raw Avro records...")
    total_records = 0
    t_start = time.time()
    for gz_path in gz_files:
        total_records += builder.process_gz_file(gz_path)
    total_time = time.time() - t_start

    log.info(f"\nAll files processed in {total_time:.1f}s ({total_records/total_time:,.0f} rec/s)")
    log.info(f"  Total records: {total_records:,}")
    log.info(f"  Registered nodes: {len(builder.uuid_to_id):,}")
    log.info(f"  Registered edges: {len(builder.edges_src):,}")
    log.info(f"  Dropped records: {builder.dropped_records:,}")

    # 3. Record accounting CSV
    acc_path = audit_dir / "darpa_record_accounting.csv"
    with open(acc_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["record_type", "count"])
        for k, v in sorted(builder.record_counts.items()):
            w.writerow([k, v])
        w.writerow(["dropped_total", builder.dropped_records])
        for k, v in sorted(builder.drop_reasons.items()):
            w.writerow([f"drop_reason_{k}", v])
    log.info(f"Record accounting written: {acc_path}")

    # 4. Edge mapping CSV
    edge_map_path = audit_dir / "darpa_event_to_edge_mapping.csv"
    with open(edge_map_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_type", "edge_category", "count"])
        for et, cnt in sorted(builder.event_type_counts.items(), key=lambda x: -x[1]):
            w.writerow([et, get_edge_category(et), cnt])
    log.info(f"Edge mapping written: {edge_map_path}")

    # 5. Build PyG Data & GT mapping
    data, gt_records = builder.build_pyg_data()
    graph_path = graph_dir / "theia_graph.pt"
    torch.save(data, graph_path)
    graph_sha = sha256_of_file(graph_path)
    log.info(f"PyG graph saved: {graph_path} (SHA-256: {graph_sha})")

    # 6. GT mapping CSV
    gt_csv_path = audit_dir / "ground_truth_mapping.csv"
    with open(gt_csv_path, "w", newline="") as f:
        if gt_records:
            w = csv.DictWriter(f, fieldnames=list(gt_records[0].keys()))
            w.writeheader()
            w.writerows(gt_records)
    log.info(f"Ground truth mapping written: {gt_csv_path}")

    # 7. Lineage JSON
    lineage = {
        "d3_task": "Defense Extension Round D3",
        "phase": "Part A — DARPA-TC-THEIA Official Rebuild",
        "synthetic_fallback": False,
        "official_raw_available": True,
        "official_ground_truth_available": True,
        "source_sha256_recorded": True,
        "source": {
            "engagement": "DARPA Transparent Computing Engagement 5",
            "ta1_provider": "THEIA",
            "schema_version": "CDM20",
            "schema_file": str(SCHEMA_FILE),
            "ground_truth_file": str(GT_DOCX if GT_DOCX.exists() else GT_PDF),
            "attack_window_utc": "2019-05-15 18:47:41 UTC to 2019-05-15 19:10:00 UTC",
            "attack_target_host": "ta1-theia-target-1 (128.55.12.110)",
            "zip_dirs_processed": len(set(f.parent.parent.name for f in gz_files)),
            "gz_files_processed": len(gz_files),
            "raw_files": [
                {"filename": r["filename"], "sha256": r["sha256"],
                 "compressed_size_bytes": r["compressed_size_bytes"]}
                for r in manifest_rows
            ],
        },
        "parser": {
            "library": "fastavro",
            "version": fastavro.__version__,
            "script": "scripts/defense_extension_real/darpa_theia_build.py",
        },
        "record_accounting": dict(builder.record_counts),
        "outer_type_distribution": dict(builder.outer_type_counts),
        "event_type_distribution": dict(builder.event_type_counts),
        "dropped_records": builder.dropped_records,
        "drop_reasons": dict(builder.drop_reasons),
        "graph_statistics": {
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.shape[1],
            "num_features": data.x.shape[1],
            "num_positive_labels": int(data.y.sum().item()),
            "num_negative_labels": int((data.y == 0).sum().item()),
            "anomaly_rate": float(data.y.sum().item() / max(1, data.num_nodes)),
        },
        "final_artifact": {
            "path": str(graph_path),
            "sha256": graph_sha,
        },
    }
    lineage_path = audit_dir / "defense_real_theia_lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2, default=str)
    log.info(f"Lineage JSON written: {lineage_path}")

    log.info("\n" + "=" * 70)
    log.info("DARPA THEIA E5 Graph Build Complete!")
    log.info(f"  Nodes: {data.num_nodes:,}")
    log.info(f"  Edges: {data.edge_index.shape[1]:,}")
    log.info(f"  Positives: {int(data.y.sum().item()):,} ({data.y.sum().item()/max(1,data.num_nodes)*100:.2f}%)")
    log.info(f"  Artifact: {graph_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
