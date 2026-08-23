#!/usr/bin/env python3
"""
LANL-RedTeam — Official Source Graph Builder (Production D3)
Defense Extension Round D3, Phase 2

Features:
- Streaming line-by-line reading of auth.txt.gz, proc.txt.gz, flows.txt.gz, dns.txt.gz
- Ground truth from official redteam.txt (positive = destination computer in red team events)
- Node universe accounting (auth, proc, flows, dns, redteam, union)
- 13 telemetry features per computer node
- Lineage JSON, manifest CSV, and PyG Data graph artifact

Usage:
    cd /mnt/d/_Work/goat_bank/dlg_gnn
    python scripts/defense_extension_real/lanl_build.py
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
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data

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

LANL_DIR = Path("/mnt/d/_Work/_data/DLG/LANL-RedTeam")

AUTH_FILE = LANL_DIR / "auth.txt.gz"
PROC_FILE = LANL_DIR / "proc.txt.gz"
FLOWS_FILE = LANL_DIR / "flows.txt.gz"
DNS_FILE = LANL_DIR / "dns.txt.gz"
REDTEAM_FILE = LANL_DIR / "redteam.txt"

OUTPUT_BASE = Path("outputs/sci_defense_extension_real")
SOURCE_AUDIT_DIR = OUTPUT_BASE / "source_audit"
GRAPH_DIR = OUTPUT_BASE / "graphs"

def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


class LanlGraphBuilder:
    """
    Builds a computer-to-computer authentication graph from LANL-RedTeam.
    """

    def __init__(self, max_ts: Optional[int] = None):
        self.max_ts = max_ts
        self.computer_to_id: dict[str, int] = {}

        # Edges: (src_id, dst_id) -> count
        self.edge_auth_count: dict[tuple, int] = defaultdict(int)

        # Per-node features
        self.feat: dict[str, dict] = defaultdict(lambda: {
            "auth_count_out": 0, "auth_count_in": 0,
            "success_count": 0, "failure_count": 0,
            "unique_users_set": set(), "unique_peers_out_set": set(),
            "unique_peers_in_set": set(),
            "proc_starts": 0, "proc_stops": 0, "unique_procs_set": set(),
            "flow_count_out": 0, "flow_count_in": 0,
            "dns_queries": 0,
        })

        self.record_counts: dict[str, int] = defaultdict(int)
        self.ts_min: Optional[int] = None
        self.ts_max: Optional[int] = None

        self.computers_in_auth: set[str] = set()
        self.computers_in_proc: set[str] = set()
        self.computers_in_flows: set[str] = set()
        self.computers_in_dns: set[str] = set()
        self.computers_in_redteam: set[str] = set()

    def _get_or_create(self, computer: str) -> int:
        nid = self.computer_to_id.get(computer)
        if nid is None:
            nid = len(self.computer_to_id)
            self.computer_to_id[computer] = nid
        return nid

    def process_auth(self) -> int:
        log.info(f"[auth] Streaming {AUTH_FILE} ...")
        t0 = time.time()
        n = 0
        skipped = 0
        with gzip.open(AUTH_FILE, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 9:
                    skipped += 1
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    skipped += 1
                    continue

                if self.max_ts and ts > self.max_ts:
                    continue

                if self.ts_min is None or ts < self.ts_min:
                    self.ts_min = ts
                if self.ts_max is None or ts > self.ts_max:
                    self.ts_max = ts

                src_user = parts[1]
                src_comp = parts[3]
                dst_comp = parts[4]
                success = parts[8].strip().upper() == "SUCCESS"

                self.computers_in_auth.add(src_comp)
                self.computers_in_auth.add(dst_comp)

                src_id = self._get_or_create(src_comp)
                dst_id = self._get_or_create(dst_comp)

                self.edge_auth_count[(src_id, dst_id)] += 1

                f_src = self.feat[src_comp]
                f_src["auth_count_out"] += 1
                f_src["unique_users_set"].add(src_user)
                f_src["unique_peers_out_set"].add(dst_comp)
                if success:
                    f_src["success_count"] += 1
                else:
                    f_src["failure_count"] += 1

                f_dst = self.feat[dst_comp]
                f_dst["auth_count_in"] += 1
                f_dst["unique_peers_in_set"].add(src_comp)

                n += 1
                if n % 10_000_000 == 0:
                    elapsed = time.time() - t0
                    log.info(f"  auth: {n:,} lines, nodes={len(self.computer_to_id):,}, edges={len(self.edge_auth_count):,}, {n/elapsed:,.0f} lines/s")

        elapsed = time.time() - t0
        log.info(f"  auth DONE: {n:,} lines ({skipped} skipped) in {elapsed:.1f}s")
        self.record_counts["auth"] = n
        return n

    def process_proc(self) -> int:
        log.info(f"[proc] Streaming {PROC_FILE} ...")
        t0 = time.time()
        n = 0
        with gzip.open(PROC_FILE, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 5:
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    continue
                if self.max_ts and ts > self.max_ts:
                    continue

                computer = parts[2]
                proc_name = parts[3]
                event_type = parts[4].strip()

                self.computers_in_proc.add(computer)
                self._get_or_create(computer)

                f = self.feat[computer]
                if event_type in ("Start", "start", "1"):
                    f["proc_starts"] += 1
                elif event_type in ("Stop", "stop", "2"):
                    f["proc_stops"] += 1
                f["unique_procs_set"].add(proc_name)

                n += 1
                if n % 10_000_000 == 0:
                    log.info(f"  proc: {n:,} lines in {time.time()-t0:.1f}s")

        log.info(f"  proc DONE: {n:,} lines in {time.time()-t0:.1f}s")
        self.record_counts["proc"] = n
        return n

    def process_flows(self) -> int:
        log.info(f"[flows] Streaming {FLOWS_FILE} ...")
        t0 = time.time()
        n = 0
        with gzip.open(FLOWS_FILE, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 9:
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    continue
                if self.max_ts and ts > self.max_ts:
                    continue

                src_comp = parts[2]
                dst_comp = parts[4]

                self.computers_in_flows.add(src_comp)
                self.computers_in_flows.add(dst_comp)
                self._get_or_create(src_comp)
                self._get_or_create(dst_comp)

                self.feat[src_comp]["flow_count_out"] += 1
                self.feat[dst_comp]["flow_count_in"] += 1
                n += 1
                if n % 10_000_000 == 0:
                    log.info(f"  flows: {n:,} lines in {time.time()-t0:.1f}s")

        log.info(f"  flows DONE: {n:,} lines in {time.time()-t0:.1f}s")
        self.record_counts["flows"] = n
        return n

    def process_dns(self) -> int:
        log.info(f"[dns] Streaming {DNS_FILE} ...")
        t0 = time.time()
        n = 0
        with gzip.open(DNS_FILE, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) < 3:
                    continue
                try:
                    ts = int(parts[0])
                except ValueError:
                    continue
                if self.max_ts and ts > self.max_ts:
                    continue

                src_comp = parts[1]
                self.computers_in_dns.add(src_comp)
                self._get_or_create(src_comp)
                self.feat[src_comp]["dns_queries"] += 1
                n += 1

        log.info(f"  dns DONE: {n:,} lines in {time.time()-t0:.1f}s")
        self.record_counts["dns"] = n
        return n

    def process_redteam(self) -> dict[str, list]:
        log.info(f"[redteam] Loading {REDTEAM_FILE} ...")
        gt: dict[str, list] = defaultdict(list)
        with open(REDTEAM_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 4:
                    continue
                ts = int(parts[0])
                user = parts[1]
                src_comp = parts[2]
                dst_comp = parts[3]
                gt[dst_comp].append({"time": ts, "user": user, "src": src_comp})
                self.computers_in_redteam.add(dst_comp)
                self.computers_in_redteam.add(src_comp)
        log.info(f"  redteam: {sum(len(v) for v in gt.values())} events, {len(gt)} positive destination computers")
        self.record_counts["redteam"] = sum(len(v) for v in gt.values())
        return dict(gt)

    def build_pyg_data(self, gt: dict) -> tuple[Data, list[dict]]:
        n_nodes = len(self.computer_to_id)
        if n_nodes == 0:
            raise ValueError("No nodes registered")

        # 13 features:
        # [auth_out, auth_in, success, failure, unique_users, unique_peers_out,
        #  unique_peers_in, proc_starts, proc_stops, unique_procs, flow_out, flow_in, dns]
        x = torch.zeros(n_nodes, 13, dtype=torch.float32)
        y = torch.zeros(n_nodes, dtype=torch.long)

        gt_rows = []
        n_pos = 0

        # Sort computers to ensure deterministic internal IDs
        sorted_computers = sorted(self.computer_to_id.keys())
        comp_to_new_id = {c: i for i, c in enumerate(sorted_computers)}

        for comp, nid in comp_to_new_id.items():
            f = self.feat[comp]
            x[nid, 0] = float(f["auth_count_out"])
            x[nid, 1] = float(f["auth_count_in"])
            x[nid, 2] = float(f["success_count"])
            x[nid, 3] = float(f["failure_count"])
            x[nid, 4] = float(len(f["unique_users_set"]))
            x[nid, 5] = float(len(f["unique_peers_out_set"]))
            x[nid, 6] = float(len(f["unique_peers_in_set"]))
            x[nid, 7] = float(f["proc_starts"])
            x[nid, 8] = float(f["proc_stops"])
            x[nid, 9] = float(len(f["unique_procs_set"]))
            x[nid, 10] = float(f["flow_count_out"])
            x[nid, 11] = float(f["flow_count_in"])
            x[nid, 12] = float(f["dns_queries"])

            if comp in gt:
                y[nid] = 1
                n_pos += 1
                events = gt[comp]
                gt_rows.append({
                    "computer_id": comp,
                    "internal_id": nid,
                    "redteam_line_count": len(events),
                    "first_compromise_time": min(e["time"] for e in events),
                    "last_compromise_time": max(e["time"] for e in events),
                    "users_involved": "|".join(set(e["user"] for e in events)),
                    "label": 1,
                    "gt_reference": "LANL_redteam.txt_official",
                    "mapping_rationale": "dstComputer_in_official_redteam_file",
                })
            else:
                gt_rows.append({
                    "computer_id": comp,
                    "internal_id": nid,
                    "redteam_line_count": 0,
                    "first_compromise_time": "",
                    "last_compromise_time": "",
                    "users_involved": "",
                    "label": 0,
                    "gt_reference": "LANL_redteam.txt_official",
                    "mapping_rationale": "not_in_official_redteam_file",
                })

        # Remap edges to new deterministic IDs
        edges_src = []
        edges_dst = []
        old_id_to_comp = {old_id: comp for comp, old_id in self.computer_to_id.items()}
        for (src_old, dst_old), _ in self.edge_auth_count.items():
            src_comp = old_id_to_comp[src_old]
            dst_comp = old_id_to_comp[dst_old]
            edges_src.append(comp_to_new_id[src_comp])
            edges_dst.append(comp_to_new_id[dst_comp])

        if edges_src:
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = n_nodes
        log.info(f"PyG Data built: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges, {n_pos:,} positive ({n_pos/n_nodes*100:.2f}%)")
        return data, gt_rows


def main():
    ap = argparse.ArgumentParser(description="LANL-RedTeam Graph Builder")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("outputs/sci_defense_extension_real"))
    ap.add_argument("--days", type=int, default=None,
                    help="Limit to first N days (default: full dataset)")
    ap.add_argument("--skip-proc-flows-dns", action="store_true",
                    help="Process only auth.txt.gz and redteam.txt")
    ap.add_argument("--skip-manifest-hash", action="store_true",
                    help="Skip recomputing SHA-256 for large files if manifest exists")
    args = ap.parse_args()

    out_base = args.output_dir
    audit_dir = out_base / "source_audit"
    graph_dir = out_base / "graphs"
    audit_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    max_ts = args.days * 86400 if args.days else None

    log.info("=" * 70)
    log.info("LANL-RedTeam Graph Builder (Round D3 Official Rebuild)")
    log.info("=" * 70)

    # 1. Manifest
    manifest_path = audit_dir / "lanl_real_manifest.csv"
    files = [AUTH_FILE, PROC_FILE, FLOWS_FILE, DNS_FILE, REDTEAM_FILE]
    if args.skip_manifest_hash and manifest_path.exists():
        log.info(f"Loading existing manifest from {manifest_path}")
        with open(manifest_path) as f:
            manifest_rows = list(csv.DictReader(f))
    else:
        manifest_rows = []
        for fpath in files:
            if not fpath.exists():
                log.error(f"Missing file: {fpath}")
                sys.exit(1)
            log.info(f"  Computing SHA-256: {fpath.name} ({fpath.stat().st_size/1024**2:.1f} MB)...")
            sha = sha256_of_file(fpath)
            manifest_rows.append({
                "filename": fpath.name,
                "source_url": "https://csr.lanl.gov/data/cyber1/",
                "compressed_size_bytes": fpath.stat().st_size,
                "sha256": sha,
            })
        with open(manifest_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            w.writeheader()
            w.writerows(manifest_rows)
        log.info(f"Manifest written: {manifest_path}")

    # 2. Build graph
    builder = LanlGraphBuilder(max_ts=max_ts)
    gt = builder.process_redteam()
    builder.process_auth()

    if not args.skip_proc_flows_dns:
        builder.process_proc()
        builder.process_flows()
        builder.process_dns()

    # 3. Node universe accounting
    universe_path = audit_dir / "lanl_node_universe.csv"
    union_computers = (builder.computers_in_auth
                       | builder.computers_in_proc
                       | builder.computers_in_flows
                       | builder.computers_in_dns
                       | builder.computers_in_redteam)
    with open(universe_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "count"])
        w.writerow(["official_reported_computers", 17684])
        w.writerow(["unique_computers_in_auth", len(builder.computers_in_auth)])
        w.writerow(["unique_computers_in_proc", len(builder.computers_in_proc)])
        w.writerow(["unique_computers_in_flows", len(builder.computers_in_flows)])
        w.writerow(["unique_computers_in_dns", len(builder.computers_in_dns)])
        w.writerow(["unique_computers_in_redteam", len(builder.computers_in_redteam)])
        w.writerow(["union_unique_computers", len(union_computers)])
        w.writerow(["final_graph_nodes", len(builder.computer_to_id)])
    log.info(f"Node universe written: {universe_path}")

    # 4. PyG Data
    data, gt_rows = builder.build_pyg_data(gt)
    graph_path = graph_dir / "lanl_graph.pt"
    torch.save(data, graph_path)
    graph_sha = sha256_of_file(graph_path)
    log.info(f"PyG graph saved: {graph_path} (SHA-256: {graph_sha})")

    # 5. GT mapping CSV
    gt_csv_path = audit_dir / "lanl_ground_truth_mapping.csv"
    with open(gt_csv_path, "w", newline="") as f:
        if gt_rows:
            w = csv.DictWriter(f, fieldnames=list(gt_rows[0].keys()))
            w.writeheader()
            w.writerows(gt_rows)
    log.info(f"GT mapping written: {gt_csv_path}")

    # 6. Lineage JSON
    lineage = {
        "d3_task": "Defense Extension Round D3",
        "phase": "Part B — LANL-RedTeam Official Rebuild",
        "synthetic_fallback": False,
        "source": {
            "dataset": "LANL Unified Host and Network Dataset",
            "url": "https://csr.lanl.gov/data/cyber1/",
            "time_limit_days": args.days,
            "raw_files": manifest_rows,
        },
        "record_accounting": dict(builder.record_counts),
        "node_universe": {
            "official_reported_computers": 17684,
            "auth": len(builder.computers_in_auth),
            "proc": len(builder.computers_in_proc),
            "flows": len(builder.computers_in_flows),
            "dns": len(builder.computers_in_dns),
            "redteam": len(builder.computers_in_redteam),
            "union": len(union_computers),
            "final_graph_nodes": data.num_nodes,
        },
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
    lineage_path = audit_dir / "defense_real_lanl_lineage.json"
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2, default=str)
    log.info(f"Lineage JSON written: {lineage_path}")

    log.info("\n" + "=" * 70)
    log.info("LANL-RedTeam Graph Build Complete!")
    log.info(f"  Nodes: {data.num_nodes:,}")
    log.info(f"  Edges: {data.edge_index.shape[1]:,}")
    log.info(f"  Positives: {int(data.y.sum().item()):,} ({data.y.sum().item()/max(1,data.num_nodes)*100:.2f}%)")
    log.info(f"  Artifact: {graph_path}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
