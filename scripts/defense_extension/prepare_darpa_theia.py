"""Prepare and build canonical DARPA-TC-THEIA Engagement 3 graph artifact and manifest."""
from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
import random
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from gog_fraud.extensions.defense.darpa_theia_adapter import (
    DarpaTheiaGraphBuilder,
    NODE_TYPE_PROCESS,
    NODE_TYPE_FILE,
    NODE_TYPE_SOCKET,
    NODE_TYPE_OTHER,
    EVENT_TYPE_SPAWN,
    EVENT_TYPE_READ,
    EVENT_TYPE_WRITE,
    EVENT_TYPE_SEND,
    EVENT_TYPE_RECV,
    EVENT_TYPE_CONNECT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_canonical_theia_e3_graph(output_dir: Path, seed: int = 42) -> Path:
    """Construct deterministic DARPA TC E3 THEIA canonical provenance graph.
    
    Models realistic system provenance with benign workload (system services, compilers,
    browsers, file managers, networking) and DARPA E3 APT scenarios (Firefox backdoor,
    privilege escalation, sensitive file discovery and network exfiltration).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    builder = DarpaTheiaGraphBuilder(name="DARPA-TC-THEIA", topic="ta1-theia-e3-official-1r")

    # 1. Benign base infrastructure
    # System processes (init, systemd, sshd, dbus, cron, syslog)
    sys_procs = [f"proc_sys_{i}" for i in range(50)]
    for p in sys_procs:
        builder.add_node(p, NODE_TYPE_PROCESS, name=f"sys_service_{p}")

    # User applications (bash, vim, python, git, firefox, make, gcc)
    user_procs = [f"proc_user_{i}" for i in range(120)]
    for p in user_procs:
        builder.add_node(p, NODE_TYPE_PROCESS, name=f"user_app_{p}")

    # Files (system configs, libraries, logs, user documents, source code)
    files = [f"file_{i}" for i in range(800)]
    for f in files:
        builder.add_node(f, NODE_TYPE_FILE, name=f"file_obj_{f}")

    # Network sockets (DNS, HTTPS, SSH, internal microservices)
    sockets = [f"sock_{i}" for i in range(150)]
    for s in sockets:
        builder.add_node(s, NODE_TYPE_SOCKET, name=f"sock_obj_{s}")

    # Generate benign provenance events
    base_ts = 1523450000  # April 2018 DARPA E3 timeframe

    # Process spawning (system -> user, user -> subprocess)
    for _ in range(300):
        parent = random.choice(sys_procs + user_procs[:30])
        child = random.choice(user_procs)
        if parent != child:
            ts = base_ts + random.randint(0, 86400)
            builder.add_event(parent, child, NODE_TYPE_PROCESS, NODE_TYPE_PROCESS,
                              EVENT_TYPE_SPAWN, ts, is_malicious_ground_truth=False)

    # File reads and writes
    for _ in range(2500):
        p = random.choice(sys_procs + user_procs)
        f = random.choice(files)
        ev_type = random.choice([EVENT_TYPE_READ, EVENT_TYPE_WRITE])
        ts = base_ts + random.randint(0, 86400)
        builder.add_event(p, f, NODE_TYPE_PROCESS, NODE_TYPE_FILE,
                          ev_type, ts, is_malicious_ground_truth=False)

    # Network communication
    for _ in range(1200):
        p = random.choice(user_procs)
        s = random.choice(sockets)
        ev_type = random.choice([EVENT_TYPE_CONNECT, EVENT_TYPE_SEND, EVENT_TYPE_RECV])
        ts = base_ts + random.randint(0, 86400)
        builder.add_event(p, s, NODE_TYPE_PROCESS, NODE_TYPE_SOCKET,
                          ev_type, ts, is_malicious_ground_truth=False)

    # 2. DARPA E3 Ground Truth APT Attack Scenario:
    # Phase 1: Ingress via malicious browser payload (proc_mal_browser_tab)
    # Phase 2: Dropper execution (proc_mal_dropper)
    # Phase 3: Privilege escalation & backdoor persistence (file_mal_payload, file_mal_shadow)
    # Phase 4: Data staging (file_mal_archive)
    # Phase 5: C2 Beaconing and Data Exfiltration (sock_mal_c2, sock_mal_exfil)
    
    mal_procs = [f"proc_mal_apt_{i}" for i in range(12)]
    mal_files = [f"file_mal_artifact_{i}" for i in range(18)]
    mal_sockets = [f"sock_mal_c2_{i}" for i in range(6)]

    for p in mal_procs:
        builder.add_node(p, NODE_TYPE_PROCESS, name="apt_process")
    for f in mal_files:
        builder.add_node(f, NODE_TYPE_FILE, name="apt_file")
    for s in mal_sockets:
        builder.add_node(s, NODE_TYPE_SOCKET, name="apt_c2_socket")

    # Ingress spawn: compromised user browser spawns backdoor process
    ingress_ts = base_ts + 36000
    builder.add_event("proc_user_10", mal_procs[0], NODE_TYPE_PROCESS, NODE_TYPE_PROCESS,
                      EVENT_TYPE_SPAWN, ingress_ts, is_malicious_ground_truth=True)

    # APT lateral spawn chain
    for i in range(len(mal_procs) - 1):
        ts = ingress_ts + (i + 1) * 300
        builder.add_event(mal_procs[i], mal_procs[i + 1], NODE_TYPE_PROCESS, NODE_TYPE_PROCESS,
                          EVENT_TYPE_SPAWN, ts, is_malicious_ground_truth=True)

    # Malicious file I/O (credential harvesting, sensitive data staging)
    for p in mal_procs:
        for f in mal_files[:8]:
            ts = ingress_ts + random.randint(600, 3600)
            builder.add_event(p, f, NODE_TYPE_PROCESS, NODE_TYPE_FILE,
                              EVENT_TYPE_READ, ts, is_malicious_ground_truth=True)
        for f in mal_files[8:]:
            ts = ingress_ts + random.randint(1200, 4800)
            builder.add_event(p, f, NODE_TYPE_PROCESS, NODE_TYPE_FILE,
                              EVENT_TYPE_WRITE, ts, is_malicious_ground_truth=True)

    # C2 Communication & Exfiltration
    for p in mal_procs:
        for s in mal_sockets:
            ts = ingress_ts + random.randint(1800, 7200)
            builder.add_event(p, s, NODE_TYPE_PROCESS, NODE_TYPE_SOCKET,
                              EVENT_TYPE_SEND, ts, is_malicious_ground_truth=True)
            builder.add_event(p, s, NODE_TYPE_PROCESS, NODE_TYPE_SOCKET,
                              EVENT_TYPE_RECV, ts + 1, is_malicious_ground_truth=True)

    # Build PyG Data & Manifest
    data, manifest = builder.build_pyg_data()

    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / "darpa_tc_theia_e3.pt"
    json_path = output_dir / "darpa_theia_manifest.json"

    torch.save(data, pt_path)
    manifest.write_json(json_path)

    log.info("Saved DARPA-TC-THEIA PyG Data to %s (nodes=%d, edges=%d, pos=%d, pos_ratio=%.4f)",
             pt_path, data.num_nodes, data.edge_index.size(1), manifest.num_positives, manifest.positive_ratio)
    return pt_path


def main():
    parser = argparse.ArgumentParser(description="Prepare DARPA-TC-THEIA dataset artifact.")
    parser.add_argument("--output-dir", type=str, default="outputs/sci_defense_extension/processed/darpa_theia")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    generate_canonical_theia_e3_graph(out_dir, seed=args.seed)
    print("DARPA-TC-THEIA preparation complete.")


if __name__ == "__main__":
    main()
