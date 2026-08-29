"""
experiments/run_all_paper_experiments.py

Master runner for all SCI paper experiments.

Executes the following pipelines in sequence:
    1. Dataset leakage audit (prerequisite)
    2. Main model evaluation (multi-seed)
    3. Ablation study (8 variants)
    4. MC sensitivity sweep (T = 1, 5, 10, 20, 30)
    5. Privacy-utility tradeoff (5 representations)
    6. Leakage attack (attribute inference)
    7. Latency profiling (component breakdown)

All results written to results/<experiment_name>/<timestamp>/

Usage:
    cd /mnt/d/_Work/goat_bank/dlg_gnn
    source /mnt/d/_Work/goat_bank/.venv/bin/activate
    python experiments/run_all_paper_experiments.py --config configs/base.yaml
"""

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def run_step(name: str, cmd: list, cwd: Path) -> bool:
    """Run a subprocess step and log result."""
    log.info(f"\n{'='*60}")
    log.info(f"  STEP: {name}")
    log.info(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        log.info(f"  ✅ DONE [{elapsed:.1f}s]")
        return True
    else:
        log.error(f"  ❌ FAILED (exit code {result.returncode}) [{elapsed:.1f}s]")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run all paper experiments for dlg_gnn GraphRAG R1")
    parser.add_argument("--config", required=True, help="Base config YAML path")
    parser.add_argument("--skip-audit",   action="store_true")
    parser.add_argument("--skip-main",    action="store_true")
    parser.add_argument("--skip-ablation",action="store_true")
    parser.add_argument("--skip-mc",      action="store_true")
    parser.add_argument("--skip-privacy", action="store_true")
    parser.add_argument("--skip-leakage", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--paper-ready",  action="store_true", help="Run full Round 2 validated paper suite")
    parser.add_argument("--dry-run",      action="store_true", help="Print commands without running")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.resolve()
    python = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = root / "results" / f"paper_experiments_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Results directory: {output_dir}")
    log.info(f"Config: {args.config}")
    if args.paper_ready:
        log.info("Mode: --paper-ready (Full Round 2 validation suite enabled)")

    steps_ok = []

    def maybe_run(skip: bool, name: str, script: str, extra_args: list = None, pass_output: bool = True):
        if skip:
            log.info(f"  SKIPPED: {name}")
            return
        cmd = [python, f"experiments/{script}", "--config", args.config]
        if pass_output:
            cmd += ["--output", str(output_dir / script.replace(".py", ""))]
        if extra_args:
            cmd += extra_args
        if args.dry_run:
            log.info(f"  [DRY RUN] {' '.join(cmd)}")
        else:
            ok = run_step(name, cmd, root)
            steps_ok.append((name, ok))

    # ── Steps ──────────────────────────────────────────────────────────────
    maybe_run(args.skip_leakage, "1. Dataset Leakage & Inference Attack", "run_leakage.py")
    maybe_run(args.skip_main,    "2. Main Model (multi-seed)",            "run_multiseed.py")
    maybe_run(args.skip_ablation,"3. Ablation Study (8 variants)",        "run_ablation.py",
              ["--ablation-config", "configs/ablation.yaml"])
    maybe_run(args.skip_mc,      "4. MC Sensitivity Sweep",               "run_mc_sensitivity.py",
              ["--mc-config", "configs/mc.yaml"])
    maybe_run(args.skip_privacy, "5. Privacy-Utility Tradeoff",           "run_privacy_utility.py",
              ["--privacy-config", "configs/privacy.yaml"])
    maybe_run(args.skip_latency, "6. Module Latency Profiling",           "run_latency.py")

    # Round 2 additions
    if args.paper_ready:
        maybe_run(False, "7. Context Lexical Baselines",                  "run_context_baselines.py", pass_output=False)
        maybe_run(False, "8. Uncertainty Subgroup Analysis",              "run_uncertainty_subgroup.py", pass_output=False)
        maybe_run(False, "9. True End-to-End Latency by T",               "run_e2e_latency.py", pass_output=False)
        maybe_run(False, "10. Robustness & Missing Context Stress Test",   "run_robustness.py", pass_output=False)
        maybe_run(False, "11. Dataset Manifest Generation",               "generate_dataset_manifest.py", pass_output=False)
        maybe_run(False, "12. LaTeX Tables Generation",                   "generate_latex_tables.py", pass_output=False)

    maybe_run(False,             "13. Publication Figures & Summary CSVs", "generate_figures.py", pass_output=False)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("  EXPERIMENT SUMMARY")
    log.info(f"{'='*60}")
    for name, ok in steps_ok:
        status = "✅" if ok else "❌"
        log.info(f"  {status}  {name}")

    failed = [n for n, ok in steps_ok if not ok]
    if failed:
        log.error(f"\n{len(failed)} step(s) FAILED: {failed}")
        sys.exit(1)
    log.info(f"\n✅ All steps completed. Results at: {output_dir}")


if __name__ == "__main__":
    main()
