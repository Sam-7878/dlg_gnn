"""Resume a checkpointed AnomalyDAE cell under a cumulative 24-hour guard."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from gog_fraud.pipelines.run_sci_round4c import (
    _timeout_record, cell_key, ensure_layout, hashes, result_path,
)


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _progress_epochs(output: Path, config: dict, dataset: str, model: str, seed: int) -> int:
    config_hash, backend_hash = hashes(config)
    key = cell_key(dataset, model, seed, config_hash, backend_hash)
    path = output / "checkpoints" / f"{key}.pt.progress.json"
    if not path.exists():
        return 0
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["completed_epochs"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return 0


def resume_guard(config_path: Path, *, dataset: str, seed: int,
                 prior_active_sec: float | None = None) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = Path(config["experiment"]["output_root"])
    ensure_layout(output)
    model = "AnomalyDAE"
    guard_sec = float(config["execution"]["max_run_wall_hours"]) * 3600
    state_path = output / "resources/dgraphfin_anomalydae_guard.json"
    existing = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    prior = max(float(existing.get("cumulative_active_sec", 0)), float(prior_active_sec or 0))
    remaining = max(0.0, guard_sec - prior)
    state = {
        "dataset": dataset, "model": model, "seed": int(seed),
        "guard_hours": guard_sec / 3600, "prior_verified_active_sec": prior,
        "cumulative_active_sec": prior, "remaining_guard_sec": remaining,
        "observed_epochs": _progress_epochs(output, config, dataset, model, seed),
        "production_projection_hours": 211.4, "guard_completed": remaining == 0,
        "status": "unsupported_operational" if remaining == 0 else "guard_resume_starting",
        "reason": "exact nonlinear all-pairs reconstruction under cumulative production guard",
    }
    _atomic_json(state_path, state)
    if remaining == 0:
        _timeout_record(config, output, dataset, model, seed, guard_sec)
        return state

    command = [sys.executable, "-m", "gog_fraud.pipelines.run_sci_round4c",
               "--config", str(config_path.resolve()), "--stage", "cell",
               "--dataset", dataset, "--model", model, "--seed", str(seed)]
    log_path = output / "logs" / f"{dataset}__{model}__seed{seed}__guard_resume.log"
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            cumulative = min(guard_sec, prior + elapsed)
            state.update({
                "cumulative_active_sec": cumulative,
                "remaining_guard_sec": max(0.0, guard_sec - cumulative),
                "observed_epochs": _progress_epochs(output, config, dataset, model, seed),
                "status": "running_under_predeclared_guard",
            })
            _atomic_json(state_path, state)
            if cumulative >= guard_sec:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                break
            time.sleep(min(60.0, max(1.0, guard_sec - cumulative)))

    elapsed = time.perf_counter() - started
    cumulative = min(guard_sec, prior + elapsed)
    result = result_path(config, output, dataset, model, seed)
    if result.exists() and json.loads(result.read_text(encoding="utf-8")).get("status") == "success":
        status = "success"
        completed = True
    elif cumulative >= guard_sec:
        _timeout_record(config, output, dataset, model, seed, guard_sec)
        status = "unsupported_operational"
        completed = True
    else:
        status = "interrupted_before_guard"
        completed = False
    state.update({
        "cumulative_active_sec": cumulative,
        "remaining_guard_sec": max(0.0, guard_sec - cumulative),
        "observed_epochs": _progress_epochs(output, config, dataset, model, seed),
        "guard_completed": completed, "status": status,
        "evidence_log": str(log_path.relative_to(output)),
    })
    _atomic_json(state_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="DGraphFin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior-active-sec", type=float)
    args = parser.parse_args()
    state = resume_guard(Path(args.config), dataset=args.dataset, seed=args.seed,
                         prior_active_sec=args.prior_active_sec)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
