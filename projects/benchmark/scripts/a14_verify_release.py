#!/usr/bin/env python3
"""
A14-G5: Fast Benchmark Reproduction and Verification Entrypoint.
Allows external reviewers and users to reproduce all tables, figures, and manuscripts
directly from a fresh repository clone without local author-specific paths.

Usage:
    python projects/benchmark/scripts/a14_verify_release.py --mode tables
    python projects/benchmark/scripts/a14_verify_release.py --mode figures
    python projects/benchmark/scripts/a14_verify_release.py --mode manuscript
    python projects/benchmark/scripts/a14_verify_release.py --mode all
"""

import sys
import argparse
import subprocess
from pathlib import Path

# Repository-relative root discovery
BASE_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BASE_DIR / 'scripts'

def run_step(cmd_list, description):
    print(f"\n[RUNNING] {description}...")
    res = subprocess.run(cmd_list, text=True)
    if res.returncode != 0:
        print(f"[FAILED] {description} exited with code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] {description}")

def mode_tables():
    python_bin = sys.executable
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_select_canonical_registry.py')], "Canonical Approved-Run Selection")
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_compute_deltas.py')], "Source-Derived 14-Dataset Delta Table Generation")

def mode_figures():
    python_bin = sys.executable
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_generate_figures.py')], "Figure Generation (Fig 1, Fig 2, Fig 3)")

def mode_manuscript():
    python_bin = sys.executable
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_build_manuscripts.py')], "Building A14 Preprints and MDPI Manuscripts")
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_compile_manuscripts.py')], "Compiling A14 LaTeX Manuscripts to PDF")
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_validate_final_manuscripts.py')], "Whole-Document Numerical Validation & Negative Controls")

def mode_all():
    mode_tables()
    mode_figures()
    mode_manuscript()
    python_bin = sys.executable
    run_step([python_bin, str(SCRIPTS_DIR / 'a14_package_submission.py')], "Packaging Submissions and Clean Unpack Compilation")

def main():
    parser = argparse.ArgumentParser(description="A14 Benchmark Fast Reproduction & Verification Tool")
    parser.add_argument(
        '--mode',
        choices=['tables', 'figures', 'manuscript', 'all'],
        default='all',
        help="Execution mode: 'tables', 'figures', 'manuscript', or 'all' (default: 'all')"
    )
    args = parser.parse_args()

    print("================================================================")
    print(" DLG-GNN Benchmark A14 Fast Reproduction Entrypoint")
    print(f" Execution Mode: {args.mode}")
    print(f" Repository Root: {BASE_DIR.parent.parent}")
    print("================================================================")

    if args.mode == 'tables':
        mode_tables()
    elif args.mode == 'figures':
        mode_figures()
    elif args.mode == 'manuscript':
        mode_manuscript()
    elif args.mode == 'all':
        mode_all()

    print("\n================================================================")
    print(" A14 VERIFICATION COMPLETED SUCCESSFULLY!")
    print("================================================================")

if __name__ == '__main__':
    main()
