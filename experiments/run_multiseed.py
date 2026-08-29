"""Route multi-seed evaluation to a real-checkpoint or explicit simulation path.

The default is fail-closed: callers must provide ``--real-checkpoint`` or opt
into the non-paper ``--simulation-study`` path.  Label-derived proxy scores are
kept physically outside this paper-facing entry point.

Output contract: the delegated real runner records ``gnn_source``, consumes
event-level ``raw_predictions``, and preserves provenance in
``run_manifests``/checkpoint manifests.  AUC-PR is computed with
``average_precision_score`` and ``roc_auc_score`` and is threshold-free;
classification metrics use
the documented fixed policy ``threshold = 0.5`` (never test-optimized).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output")
    parser.add_argument("--seeds", default="7,17,27,37,47")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--real-checkpoint",
        type=Path,
        help="A trained checkpoint or directory containing the five seed checkpoints.",
    )
    mode.add_argument(
        "--simulation-study",
        action="store_true",
        help="Run the legacy label-informed framework simulation (never paper eligible).",
    )
    args = parser.parse_args()

    if args.simulation_study:
        command = [
            sys.executable,
            "experiments/simulation/run_multiseed_simulation.py",
            "--config",
            args.config,
            "--seeds",
            args.seeds,
        ]
        if args.output:
            command.extend(("--output", args.output))
        return subprocess.run(command, cwd=ROOT, check=False).returncode

    checkpoint = args.real_checkpoint.resolve()
    if not checkpoint.exists():
        parser.error(f"real checkpoint path does not exist: {checkpoint}")
    checkpoint_dir = checkpoint if checkpoint.is_dir() else checkpoint.parent
    command = [
        sys.executable,
        "experiments/round3/run_full_pipeline.py",
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--seeds",
        args.seeds,
    ]
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
