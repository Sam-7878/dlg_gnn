"""Resume-safe execution order for the Round 4C completion task."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gog_fraud.pipelines.analyze_sci_round4c_completion import finalize
from gog_fraud.pipelines.classify_sci_round4c_support import build_ledger
from gog_fraud.pipelines.run_sci_round4c import (
    _external_failure_record, _timeout_record, ensure_layout, result_path, run_matrix,
)
from gog_fraud.pipelines.run_sci_round4c_guard_resume import resume_guard


FAST_REDDIT_MODELS = ["DOMINANT", "CONAD", "DLG-Base", "DLG-Aug", "CoLA", "OCGNN"]


def _run_fresh_cell(config: dict, config_path: Path, output: Path,
                    dataset: str, model: str, seed: int) -> dict:
    path = result_path(config, output, dataset, model, seed)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    command = [sys.executable, "-m", "gog_fraud.pipelines.run_sci_round4c",
               "--config", str(config_path.resolve()), "--stage", "cell",
               "--dataset", dataset, "--model", model, "--seed", str(seed)]
    log = output / "logs" / f"{dataset}__{model}__seed{seed}__completion.log"
    timeout_sec = float(config["execution"]["max_run_wall_hours"]) * 3600
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        try:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                       timeout=timeout_sec, check=False)
            if completed.returncode != 0 and not path.exists():
                _external_failure_record(config, output, dataset, model, seed,
                                         completed.returncode, time.perf_counter() - started)
        except subprocess.TimeoutExpired:
            _timeout_record(config, output, dataset, model, seed, timeout_sec)
    return json.loads(path.read_text(encoding="utf-8"))


def run_completion(config_path: Path, *, prior_dgraph_active_sec: float) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_root"])
    ensure_layout(output)
    run_matrix(config, config_path, output, resume=True, retry_failed=False,
               datasets=["Reddit"], models=FAST_REDDIT_MODELS)
    resume_guard(config_path, dataset="DGraphFin", seed=42,
                 prior_active_sec=prior_dgraph_active_sec)
    for model in ("GADNR", "AnomalyDAE"):
        seed42 = _run_fresh_cell(config, config_path, output, "Reddit", model, 42)
        if seed42["status"] == "success":
            _run_fresh_cell(config, config_path, output, "Reddit", model, 43)
    ledger = build_ledger(config, output)
    ledger_path = output / "manifests/support_reclassification.json"
    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    return finalize(config, output, Path("outputs/sci_round4b"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prior-dgraph-active-sec", type=float, default=63064.0,
                        help="conservative last verified cumulative active runtime")
    args = parser.parse_args()
    gate = run_completion(Path(args.config), prior_dgraph_active_sec=args.prior_dgraph_active_sec)
    print(json.dumps(gate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
