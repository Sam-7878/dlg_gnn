#!/usr/bin/env python3
"""Reviewer-facing reproduction facade; never starts training implicitly."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ("dlg_gnn", "benchmark", "stream_mc", "tds")
EVIDENCE = ROOT / "projects/benchmark/evidence/public_numeric_evidence.zip"
A05 = "evaluation/benchmark/v2/paper_ready_a05/"
PUB = A05 + "publication_evidence_a05/"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def environment() -> dict[str, str]:
    names = ("torch", "torch-geometric", "pygod", "scipy", "networkx")
    result = {"python": platform.python_version(), "os": platform.platform()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed (archive integrity verification remains available)"
    return result


def verify_archive() -> zipfile.ZipFile:
    check(EVIDENCE.is_file(), f"missing bundled evidence: {EVIDENCE}")
    expected = json.loads((ROOT / "projects/benchmark/expected_outputs.json").read_text())
    check(digest(EVIDENCE.read_bytes()) == expected["evidence_zip_sha256"], "evidence ZIP hash mismatch")
    archive = zipfile.ZipFile(EVIDENCE)
    names = archive.namelist()
    check(len(names) == len(set(names)), "duplicate archive entries")
    check(all(not Path(n).is_absolute() and ".." not in Path(n).parts for n in names), "unsafe archive path")
    manifest = json.loads(archive.read("release_manifest.json"))
    listed = {item["path"] for item in manifest["files"]}
    check(manifest["file_count"] == len(listed) == len(names) - 1, "release manifest count mismatch")
    check(listed == set(names) - {"release_manifest.json"}, "release manifest entries mismatch")
    for item in manifest["files"]:
        payload = archive.read(item["path"])
        check(len(payload) == item["bytes"] and digest(payload) == item["sha256"], f"archive payload drift: {item['path']}")
    return archive


def benchmark_verify() -> dict[str, object]:
    subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/review_evidence_manifest.py"), "--verify"],
                   cwd=ROOT, check=True)
    with verify_archive() as archive:
        manifest = json.loads(archive.read(PUB + "dataset_manifest_canonical.json"))
        primary = [x for x in manifest if x["dataset_id"] != "LANL-RedTeam"]
        check(len(primary) == 13 and len(manifest) == 14, "dataset count mismatch")
        support = csv_rows(archive.read(A05 + "table_support_primary13_24g.csv"))
        models = {r["model"] for r in support if r["support_status"] != "DIAGNOSTIC_ONLY"}
        supported = [r for r in support if r["support_status"] == "SUPPORTED_EXACT"]
        check(len(models) == 7 and len(supported) == 80 and len(support) == 104, "13x7 support matrix mismatch")
        check(all(r["model"].startswith("CONAD") and r["support_status"] == "DIAGNOSTIC_ONLY" for r in support if r["model"].startswith("CONAD")), "CONAD policy mismatch")
        registry = csv_rows(archive.read(PUB + "approved_run_registry.csv"))
        successful = [r for r in registry if r["status"] == "success"]
        check(len(successful) == 435, "approved success count mismatch")
        check(len({(r["dataset"], r["model"], r["seed"]) for r in successful}) == 435, "duplicate approved records")
        exact = json.loads(archive.read(PUB + "environment/clean_exact_reconstruction_verification.json"))
        fused = json.loads(archive.read(PUB + "environment/clean_fused_gcn_verification.json"))
        dae = json.loads(archive.read(PUB + "anomalydae_exact_equivalence.json"))
        conad = json.loads(archive.read(PUB + "conad_eq1_integration_a04.json"))
        check(exact["all_passed"] and fused["all_passed"] and dae["status"] == "PASS", "archived equivalence gate failed")
        check(conad["primary_ranking_eligible"] is False, "CONAD diagnostic gate failed")
        stats = csv_rows(archive.read(A05 + "statistics_s1_s5.csv"))
        check(len(stats) == 5 and stats[0]["n_complete"] == "5" and stats[1]["n_complete"] == "13", "statistics views mismatch")
        btc = next(x for x in manifest if x["dataset_id"] == "BitcoinOTC")
        check("inject" in json.dumps(btc).lower(), "BitcoinOTC injected-label provenance missing")
        # An independent tiny arithmetic identity checks the exact sparse Gram formula.
        z = ((1., 2.), (3., 4.), (2., -1.))
        edges = {(0, 1), (2, 0)}
        dense = [sum(((1. if (i, j) in edges else 0.) - sum(z[i][k]*z[j][k] for k in range(2)))**2 for j in range(3)) for i in range(3)]
        gram = [[sum(z[i][k]*z[j][k] for k in range(2)) for j in range(3)] for i in range(3)]
        sparse = [sum(v*v for v in gram[i]) + sum(1. - 2.*gram[i][j] for j in range(3) if (i,j) in edges) for i in range(3)]
        check(max(abs(a-b) for a,b in zip(dense,sparse)) < 1e-10, "tiny Gram identity failed")
        return {"archive_files":len(archive.namelist())-1,"datasets":len(primary),"models":len(models),"supported_pairs":len(supported),"approved_successes":len(successful),"archived_equivalence":"PASS","tiny_gram":"PASS"}


def extract_paper_inputs() -> None:
    """Restore frozen inputs only when absent; never replace a local evidence file."""
    with verify_archive() as archive:
        wanted = [n for n in archive.namelist() if n in {
            A05 + "publication_evidence_a05/paper_metric_cells.csv",
            A05 + "statistics_s1_s5.csv",
            A05 + "figure_memory_measured_a05.pdf",
            PUB + "approved_run_registry.csv",
        } or n.startswith(PUB + "legacy_metric_json/")]
        for name in wanted:
            if name.endswith("/"):
                continue
            dest = ROOT / name
            if dest.exists():
                check(digest(dest.read_bytes()) == digest(archive.read(name)), f"local frozen input differs: {name}")
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(archive.read(name))


def benchmark_paper() -> dict[str, object]:
    check((ROOT / "projects/benchmark/scripts/a07_build_manuscript.py").is_file()
          and (ROOT / "projects/benchmark/manuscript_base_a06").is_dir(),
          "Unsubmitted manuscript sources are intentionally withheld. Use --mode tables for public evidence regeneration; paper requires the author's local manuscript inputs.")
    result = benchmark_verify()
    extract_paper_inputs()
    script = ROOT / "projects/benchmark/scripts/a07_build_manuscript.py"
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    out = ROOT / "projects/benchmark/paper/current"
    frozen_tables = (
        "table_main_13_pr_auc.csv", "table_main_13_roc_auc.csv", "table_main_13_f1.csv",
        "table_support_primary13_24g.csv", "table_support_lanl_24g.csv",
        "statistics_s1_s5.csv", "figure_memory_measured_a05.pdf",
    )
    with zipfile.ZipFile(EVIDENCE) as archive:
        for name in frozen_tables:
            shutil.copyfileobj(archive.open(A05 + name), (out / name).open("wb"))
    subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/a07_consistency_gate.py")], cwd=ROOT, check=True)
    expected = json.loads((ROOT / "projects/benchmark/expected_outputs.json").read_text())
    check(digest((out / "table_alert_budget_a07.csv").read_bytes()) == expected["alert_budget_sha256"], "alert budget drift")
    check(len(csv_rows((out / "statistics_pairwise_s1_s2.csv").read_bytes())) == 4, "pairwise row count drift")
    check(len(csv_rows((out / "claims_to_evidence_a07.csv").read_bytes())) == 14, "claim count drift")
    subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/artifact_manifest.py"), "--verify"],
                   cwd=ROOT, check=True)
    result["paper_outputs"] = {p.name:digest(p.read_bytes()) for p in (out / "DLG-Benchmark_A07.tex", out / "Supplementary_F1_A07.tex", out / "references.bib", out / "statistics_pairwise_s1_s2.csv", out / "table_alert_budget_a07.csv")}
    return result


def source_verify(project: str) -> dict[str, object]:
    manifest = json.loads((ROOT / "projects" / project / "expected_outputs.json").read_text())
    for path, sha in manifest["source_hashes"].items():
        check(digest((ROOT/path).read_bytes()) == sha, f"{project} source drift: {path}")
    scientific = "SKIPPED: optional scientific dependencies unavailable"
    fixture_sha256 = None
    if project == "dlg_gnn":
        check((ROOT / "tests/benchmark/pygod_integration/test_dlg_pygod.py").is_file(), "DLG integration test missing")
        if importlib.util.find_spec("torch") and importlib.util.find_spec("pygod"):
            import torch
            sys.path.insert(0,str(ROOT / "src"))
            from gog_fraud.models.pygod.dlg_base import DLGBase
            from gog_fraud.models.pygod.exact_reconstruction import exact_dot_product_row_squared_error
            torch.manual_seed(42)
            x = torch.tensor([[1.,0.,2.],[0.,1.,1.],[2.,1.,0.],[1.,2.,1.]])
            edge = torch.tensor([[0,1,2,3,0,2],[1,2,3,0,2,0]])
            model = DLGBase(in_dim=3,hid_dim=4,num_layers=2,dropout=0.,alpha=0.)
            for name, parameter in model.named_parameters():
                if name != "alpha":
                    torch.nn.init.constant_(parameter, 0.1)
            x_hat, a_hat = model(x,edge)
            check(x_hat.shape == x.shape and a_hat.shape == (4,4), "DLG tiny forward shape")
            check(abs(torch.sigmoid(model.alpha).item()-0.5)<1e-8, "DLG gate initial value")
            fixture_sha256 = digest(json.dumps({"x_hat":x_hat.detach().round(decimals=6).tolist(),"a_hat":a_hat.detach().round(decimals=6).tolist()},sort_keys=True,separators=(",",":")).encode())
            (x_hat.square().sum()+a_hat.square().sum()).backward()
            check(model.alpha.grad is not None and torch.isfinite(model.alpha.grad).all(), "DLG gate backward")
            z = torch.tensor([[1.,2.],[3.,4.],[2.,-1.]],dtype=torch.float64,requires_grad=True)
            tiny_edge = torch.tensor([[0,2],[1,0]])
            adjacency = torch.zeros((3,3),dtype=torch.float64);adjacency[tiny_edge[0],tiny_edge[1]] = 1.
            dense = (adjacency-z@z.T).square().sum(dim=1)
            sparse = exact_dot_product_row_squared_error(z,tiny_edge)
            check(torch.allclose(dense,sparse,rtol=1e-12,atol=1e-12), "DLG exact sparse tiny value")
            gd = torch.autograd.grad(dense.sum(),z,retain_graph=True)[0]
            gs = torch.autograd.grad(sparse.sum(),z)[0]
            check(torch.allclose(gd,gs,rtol=1e-12,atol=1e-12), "DLG exact sparse tiny gradient")
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
            subprocess.run([sys.executable,"-m","pytest","tests/benchmark/pygod_integration/test_dlg_pygod.py","-q"],cwd=ROOT,env=env,check=True,stdout=subprocess.DEVNULL)
            scientific = "PASS: deterministic DLG forward/backward/gate/exact-sparse fixture and PyGOD interface"
    elif project == "stream_mc":
        check((ROOT / "tests/stream_mc/streaming/test_stateful_stream.py").is_file(), "stream replay test missing")
        if importlib.util.find_spec("pandas") and importlib.util.find_spec("torch"):
            sys.path.insert(0,str(ROOT / "src"))
            from gog_fraud.data.io.streaming_dataset import StatefulTransactionStream
            from gog_fraud.streaming.subgraph_store import IncrementalSubgraphStore
            events = [dict(sample_id=str(i),chain_id="eth",contract_id="c",event_time=i*5,payload=dict(src="u",dst=f"v{i}")) for i in range(3)]
            replay = StatefulTransactionStream(events,seed=42)
            store = IncrementalSubgraphStore(temporal_window_seconds=6,max_nodes_per_contract=3,max_edges_per_contract=2)
            first = next(iter(replay)); store.apply_event(first)
            checkpoint = replay.checkpoint(); snapshot = store.snapshot()
            rest = list(replay); replay.restore(checkpoint); check([x.sample_id for x in replay] == [x.sample_id for x in rest], "stream replay mismatch")
            restored = IncrementalSubgraphStore(temporal_window_seconds=6,max_nodes_per_contract=3,max_edges_per_contract=2)
            restored.restore(snapshot)
            for event in rest: restored.apply_event(event)
            materialized = restored.materialize("c",10)
            check(materialized["edges"] == [("u","v1",5),("u","v2",10)], "bounded stream fixture mismatch")
            fixture_sha256 = digest(json.dumps(materialized,sort_keys=True,separators=(",",":")).encode())
            scientific = "PASS: ingestion, checkpoint/replay and bounded subgraph fixture"
    else:
        check((ROOT / "tests/tds/micro_rag/graph_builder.py").is_file(), "TDS graph source missing")
        if importlib.util.find_spec("networkx") and importlib.util.find_spec("matplotlib"):
            import importlib.util as iu
            spec = iu.spec_from_file_location("tds_graph_builder", ROOT / "tests/tds/micro_rag/graph_builder.py")
            module = iu.module_from_spec(spec);spec.loader.exec_module(module)
            graph = module.GraphManager()
            graph.add_review_graph({"entities":[{"id":"USER_A","type":"USER"},{"id":"PHISHING","type":"INTENT"}],"relations":[{"source":"USER_A","target":"PHISHING","type":"linked"}]},domain="amazon")
            check(graph.G.number_of_nodes()==2 and graph.G.number_of_edges()==1 and graph.calculate_r_local("USER_A","amazon")==1.0,"TDS graph fixture mismatch")
            fixture_sha256 = digest(json.dumps({"nodes":sorted(graph.G.nodes),"edges":sorted([list(e) for e in graph.G.edges]),"risk":graph.calculate_r_local("USER_A","amazon")},sort_keys=True,separators=(",",":")).encode())
            scientific = "PASS: deterministic entity graph and local-risk fixture"
    if fixture_sha256 is not None and "fixture_sha256" in manifest:
        check(fixture_sha256 == manifest["fixture_sha256"], f"{project} fixture output hash drift")
    return {"source_hashes":"PASS", "scientific_smoke":scientific, "fixture_sha256":fixture_sha256}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", choices=PROJECTS, required=True)
    ap.add_argument("--mode", choices=("verify", "tables", "paper", "full"), required=True)
    ap.add_argument("--confirm-full", action="store_true")
    ap.add_argument("--revision", choices=("auto", "legacy-a07", "a08"), default="auto", help="Explicitly select historical or corrected evidence; no old-result fallback")
    args = ap.parse_args()
    start = time.monotonic()
    if args.mode == "full":
        print("FULL RE-EXECUTION IS EXPENSIVE. Requires third-party datasets, GPU resources and campaign-specific commands.", file=sys.stderr)
        if not args.confirm_full:
            print("Pass --confirm-full after reviewing projects/<project>/README.md. No training started.", file=sys.stderr)
            raise SystemExit(2)
        print("Full rerun is manual campaign orchestration; no training was started by this facade.", file=sys.stderr)
        raise SystemExit(2)
    if args.project == "benchmark":
        a08_active = args.revision == "a08" or (args.revision == "auto" and (ROOT / "configs/benchmark/a08_crypto_clean_v1.yaml").exists())
        if a08_active and args.mode == "paper":
            private = ROOT / "projects/benchmark/scripts/a08_build_manuscript.py"
            check(private.is_file(), "Unsubmitted A08 manuscript writer/sources are intentionally withheld; paper requires author-local inputs. Public --mode tables only.")
            gate = ROOT / "projects/benchmark/evidence/a08_data_repair/audit/G7.json"
            check(gate.is_file() and json.loads(gate.read_text()).get("status") == "PASS", "A08 raw-score registry is not yet approved; refusing historical PDF fallback")
            subprocess.run([sys.executable, str(private)], cwd=ROOT, check=True)
            detail = {"revision": "a08", "scope": "author-local actual new PDF build; semantic/visual audit separate"}
        elif a08_active and args.mode == "tables":
            registry = ROOT / "projects/benchmark/evidence/a08_data_repair/approved_registry.json"
            check(registry.is_file(), "A08 campaign/registry not complete; historical tables require explicit --revision legacy-a07")
            subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/a08_aggregate.py"), "--public-tables"], cwd=ROOT, check=True)
            detail = {"revision": "a08", "scope": "numeric tables from approved registry; raw scores and paper remain author-local"}
        elif a08_active and args.mode == "verify":
            public = ROOT / "projects/benchmark/evidence/a08_public_release.json"
            check(public.is_file(), "A08 current public evidence is not yet packaged; historical archive verification requires --revision legacy-a07")
            subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/a08_public_evidence.py")], cwd=ROOT, check=True)
            detail = {"revision": "a08", "scope": "current public numeric integrity and run identity; FINAL_PASS and private raw-score/PDF review are separate"}
        elif args.mode == "tables":
            detail = benchmark_verify()
            subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/build_review_evidence.py")], cwd=ROOT, check=True)
            subprocess.run([sys.executable, str(ROOT / "projects/benchmark/scripts/review_evidence_manifest.py"), "--verify"], cwd=ROOT, check=True)
            detail["tables"] = "PASS: frozen tables and seeded permutation statistics regenerated; no manuscript sources or training"
        else:
            detail = benchmark_paper() if args.mode == "paper" else benchmark_verify()
            detail["revision"] = "historical-a05-a07-archive"
            detail["scope"] = "archive/evidence integrity only; not A08 scientific acceptance"
    else:
        if args.mode in ("paper", "tables"):
            raise SystemExit(f"{args.project} paper regeneration has not been qualified; verify only")
        detail = source_verify(args.project)
    print(json.dumps({"project":args.project,"mode":args.mode,"status":"PASS","environment":environment(),"detail":detail,"elapsed_seconds":round(time.monotonic()-start,2)},indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
