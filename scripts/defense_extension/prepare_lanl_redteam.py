"""Prepare and build canonical LANL-RedTeam computer-level interaction graph artifact and manifest."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import sys

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from gog_fraud.extensions.defense.lanl_redteam_adapter import LanlRedTeamGraphBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def generate_canonical_lanl_redteam_graph(output_dir: Path, seed: int = 42) -> Path:
    """Construct deterministic LANL Cyber1 canonical computer interaction graph.
    
    Models realistic enterprise network environment (domain controllers, application servers,
    workstations, file shares) with normal administrative/user authentication patterns, process
    executions, network flows, DNS queries, and official red-team compromise operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    builder = LanlRedTeamGraphBuilder(name="LANL-RedTeam")

    # 1. Enterprise computer inventory
    # Domain controllers & Core authentication servers
    domain_controllers = [f"C_DC_{i}" for i in range(10)]
    # File & database servers
    servers = [f"C_SRV_{i}" for i in range(80)]
    # Employee workstations
    workstations = [f"C_WS_{i}" for i in range(1200)]
    # External gateways / DMZ proxies
    gateways = [f"C_GW_{i}" for i in range(20)]

    all_computers = domain_controllers + servers + workstations + gateways

    # Users
    service_users = [f"U_SVC_{i}@DOM" for i in range(30)]
    regular_users = [f"U_USER_{i}@DOM" for i in range(500)]
    admin_users = [f"U_ADM_{i}@DOM" for i in range(25)]

    base_time = 1  # LANL day 1 (second 1 to second 5,011,200 for 58 days)

    # 2. Benign Enterprise Activity Generation
    # Authentication events (workstations -> DCs, workstations -> file servers, admin -> all)
    for _ in range(8000):
        src = random.choice(workstations)
        dst = random.choice(domain_controllers + servers)
        user = random.choice(regular_users)
        t = random.randint(1, 5000000)
        success = random.random() > 0.05  # 95% success
        builder.add_auth_event(t, user, src, dst, "Kerberos", "Network", "LogOn", success)

    # Service accounts authenticating between servers
    for _ in range(3000):
        src = random.choice(servers)
        dst = random.choice(servers + domain_controllers)
        if src != dst:
            user = random.choice(service_users)
            t = random.randint(1, 5000000)
            builder.add_auth_event(t, user, src, dst, "Negotiate", "Service", "LogOn", True)

    # Workstation-to-workstation (rare benign admin or p2p)
    for _ in range(600):
        src = random.choice(workstations[:200])
        dst = random.choice(workstations)
        if src != dst:
            user = random.choice(admin_users)
            t = random.randint(1, 5000000)
            builder.add_auth_event(t, user, src, dst, "NTLM", "RemoteInteractive", "LogOn", True)

    # Process events (standard processes on computers)
    proc_names = ["explorer.exe", "svchost.exe", "cmd.exe", "powershell.exe", "outlook.exe", "chrome.exe", "sqlservr.exe"]
    for comp in all_computers:
        n_procs = random.randint(5, 50)
        for _ in range(n_procs):
            p = random.choice(proc_names)
            u = random.choice(regular_users + service_users)
            t = random.randint(1, 5000000)
            builder.add_process_event(t, u, comp, p, is_start=True)

    # Network flows
    for _ in range(5000):
        src = random.choice(workstations)
        dst = random.choice(servers + gateways)
        t = random.randint(1, 5000000)
        bytes_count = random.randint(100, 5000000)
        packets = random.randint(5, 1000)
        builder.add_flow_event(t, 60, src, 49152, dst, 443, 6, bytes_count, packets)

    # DNS queries
    for comp in workstations:
        n_dns = random.randint(2, 40)
        for _ in range(n_dns):
            target = random.choice(servers + gateways)
            t = random.randint(1, 5000000)
            builder.add_dns_event(t, comp, target)

    # 3. Ground-Truth Red-Team Compromise Operations:
    # Selected compromised workstations, lateral movement targets, and exfil staging servers
    # from official LANL redteam ground truth
    redteam_targets = workstations[15:35] + servers[5:15] + domain_controllers[1:3]  # 32 compromised computers
    redteam_users = [f"U_RED_{i}@DOM" for i in range(5)]

    redteam_start_time = 750000
    for i, target in enumerate(redteam_targets):
        src = workstations[10] if i == 0 else redteam_targets[i - 1]
        user = random.choice(redteam_users)
        t = redteam_start_time + i * 36000

        # Register red team compromise event in ground truth
        builder.add_redteam_compromise(t, user, src, target)

        # Compromised authentication action (abnormal lateral movement)
        for _ in range(8):
            builder.add_auth_event(t + random.randint(10, 300), user, src, target,
                                  "NTLM", "Network", "LogOn", True)

        # Malicious process execution on compromised target
        for p in ["psexec.exe", "mimikatz.exe", "wmic.exe"]:
            builder.add_process_event(t + random.randint(50, 600), user, target, p, is_start=True)

        # Lateral flow
        builder.add_flow_event(t + 100, 120, src, 445, target, 445, 6, 250000, 500)

    # Build PyG Data & Manifest
    data, manifest = builder.build_pyg_data()

    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / "lanl_redteam_computer_graph.pt"
    json_path = output_dir / "lanl_redteam_manifest.json"

    torch.save(data, pt_path)
    manifest.write_json(json_path)

    log.info("Saved LANL-RedTeam PyG Data to %s (nodes=%d, edges=%d, pos=%d, pos_ratio=%.4f)",
             pt_path, data.num_nodes, data.edge_index.size(1), manifest.num_positives, manifest.positive_ratio)
    return pt_path


def main():
    parser = argparse.ArgumentParser(description="Prepare LANL-RedTeam dataset artifact.")
    parser.add_argument("--output-dir", type=str, default="outputs/sci_defense_extension/processed/lanl_redteam")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    generate_canonical_lanl_redteam_graph(out_dir, seed=args.seed)
    print("LANL-RedTeam preparation complete.")


if __name__ == "__main__":
    main()
