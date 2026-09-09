"""
experiments/generate_latex_tables.py

Generates publication-ready LaTeX booktabs tables from results CSVs:
  - results/tables/main_results.tex
  - results/tables/context_baselines.tex
  - results/tables/uncertainty_subgroup.tex
  - results/tables/e2e_latency.tex
  - results/tables/robustness.tex
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Format DataFrame as a clean, publication-ready LaTeX booktabs table."""
    latex_table = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "r" * (len(df.columns) - 1),
    )
    wrapper = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        f"{latex_table}"
        "\\end{table}\n"
    )
    return wrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--tables-dir", default="results/tables")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    res_dir = root / args.results_dir
    tbl_dir = root / args.tables_dir
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main Results Table
    main_csv = res_dir / "main_results.csv"
    if main_csv.exists():
        df_main = pd.read_csv(main_csv)
        tex = df_to_latex(
            df_main,
            caption="Overall detection performance of proposed method (5 seeds)",
            label="tab:main_results",
        )
        (tbl_dir / "main_results.tex").write_text(tex, encoding="utf-8")
        log.info("Generated results/tables/main_results.tex")

    # 2. Context Baselines Table
    ctx_csv = res_dir / "context_baselines.csv"
    if ctx_csv.exists():
        df_ctx = pd.read_csv(ctx_csv)
        tex = df_to_latex(
            df_ctx,
            caption="Context-only lexical baselines comparison",
            label="tab:context_baselines",
        )
        (tbl_dir / "context_baselines.tex").write_text(tex, encoding="utf-8")
        log.info("Generated results/tables/context_baselines.tex")

    # 3. Uncertainty Subgroup Table
    unc_csv = res_dir / "uncertainty_subgroup.csv"
    if unc_csv.exists():
        df_unc = pd.read_csv(unc_csv)
        tex = df_to_latex(
            df_unc,
            caption="Performance stratification by uncertainty quartile",
            label="tab:uncertainty_subgroup",
        )
        (tbl_dir / "uncertainty_subgroup.tex").write_text(tex, encoding="utf-8")
        log.info("Generated results/tables/uncertainty_subgroup.tex")

    # 4. E2E Latency Table
    lat_csv = res_dir / "e2e_latency_by_T.csv"
    if lat_csv.exists():
        df_lat = pd.read_csv(lat_csv)
        tex = df_to_latex(
            df_lat,
            caption="True End-to-End Latency vs. MC Samples ($T$)",
            label="tab:e2e_latency",
        )
        (tbl_dir / "e2e_latency.tex").write_text(tex, encoding="utf-8")
        log.info("Generated results/tables/e2e_latency.tex")

    # 5. Robustness Table
    rob_csv = res_dir / "robustness.csv"
    if rob_csv.exists():
        df_rob = pd.read_csv(rob_csv)
        tex = df_to_latex(
            df_rob,
            caption="Robustness under missing and noisy context conditions",
            label="tab:robustness",
        )
        (tbl_dir / "robustness.tex").write_text(tex, encoding="utf-8")
        log.info("Generated results/tables/robustness.tex")


if __name__ == "__main__":
    main()
