#!/usr/bin/env python3
"""Cross-check A07 manuscript claims, frozen evidence, and reviewer docs."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[3]
ZIP = ROOT / "projects/benchmark/evidence/public_numeric_evidence.zip"
P = "evaluation/benchmark/v2/paper_ready_a05/"
M = ROOT / "projects/benchmark/paper/current/DLG-Benchmark_A07.tex"


def must(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    with zipfile.ZipFile(ZIP) as z:
        manifest = json.loads(z.read(P + "publication_evidence_a05/dataset_manifest_canonical.json"))
        support = list(csv.DictReader(io.StringIO(z.read(P + "table_support_primary13_24g.csv").decode())))
        registry = list(csv.DictReader(io.StringIO(z.read(P + "publication_evidence_a05/approved_run_registry.csv").decode())))
    primary = [x for x in manifest if x["dataset_id"] != "LANL-RedTeam"]
    models = {x["model"] for x in support if x["support_status"] != "DIAGNOSTIC_ONLY"}
    supported = [x for x in support if x["support_status"] == "SUPPORTED_EXACT"]
    successes = [x for x in registry if x["status"] == "success"]
    must((len(primary),len(models),len(supported),len(successes)) == (13,7,80,435), "frozen counts changed")
    readme = (ROOT / "README.md").read_text()
    benchmark = (ROOT / "projects/benchmark/README.md").read_text()
    datasets = (ROOT / "projects/benchmark/DATASETS.md").read_text()
    manuscript = M.read_text()
    for name, text in (("root README",readme),("benchmark README",benchmark)):
        for needle in ("13 primary", "seven functioning", "80/91", "435", "CONAD", "BitcoinOTC"):
            must(needle.lower() in text.lower(), f"{name}: missing {needle}")
        for stale in ("71/80", "355 primary", "10 primary datasets"):
            must(stale.lower() not in text.lower(), f"{name}: stale {stale}")
    must("13 primary" in manuscript and "Eighty of 91" in manuscript and "435" in manuscript, "manuscript counts missing")
    must("controlled node-level injection" in manuscript and "two provider-native graphs" in manuscript
         and "unresolved construction provenance" in manuscript, "label provenance audit missing")
    must("DOMINANT / DLG-Aug (tied)" in manuscript, "S2 tied mean-rank leaders missing")
    must("protocol-incomparable diagnostics" in manuscript, "crypto threshold-policy caveat missing")
    must("will be inserted after archive publication" not in manuscript, "future publication placeholder remains")
    must("Selected Holm-Adjusted Pairwise Comparisons" in manuscript, "Holm table missing")
    must("BSC is the clearest negative" in manuscript, "BSC discussion missing")
    must("DR-GAD" not in manuscript, "unverified citation present")
    must("Access limitation" in datasets and "public upstream download" in datasets, "dataset access limitation missing")
    must("independent full reruns of those three cases require separate data access" in manuscript, "contract graph access limitation missing")
    must("benchmark-a04-cuda.lock.txt" in (ROOT / "projects/benchmark/ENVIRONMENT.md").read_text(), "A04/A05 environment missing")
    must("Historical Round5" in (ROOT / "projects/benchmark/ENVIRONMENT.md").read_text(), "legacy environment disclosure missing")
    tags = re.findall(r"tag \\texttt\{([^}]+)\}, commit \\texttt\{([^}]+)\}",manuscript)
    for tag,commit in tags:
        resolved = subprocess.check_output(["git","rev-parse",f"refs/tags/{tag}^{{commit}}"],cwd=ROOT,text=True).strip()
        must(resolved == commit, "manuscript tag/commit mismatch")
    print(json.dumps({"status":"PASS","datasets":13,"models":7,"supported_pairs":"80/91","approved_successes":435,"release_identity_in_manuscript":bool(tags)},indent=2))


if __name__ == "__main__":
    main()
