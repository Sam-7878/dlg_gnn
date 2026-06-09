import os
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PaperTableExporter:
    """
    Exports benchmark results into LaTeX and Markdown tables for paper inclusion.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def export_latex_table(self, df: pd.DataFrame, file_prefix: str, caption: str, label: str):
        """Converts DataFrame to a professional LaTeX table code block."""
        tex_path = os.path.join(self.output_dir, f"{file_prefix}.tex")
        
        # Format floating numbers
        formatted_df = df.copy()
        for col in formatted_df.columns:
            if formatted_df[col].dtype in [float, 'float64']:
                formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.4f}" if abs(x) < 1.0 else f"{x:.2f}")

        latex_code = formatted_df.to_latex(
            index=False,
            caption=caption,
            label=f"tab:{label}",
            column_format="l" + "c" * (len(df.columns) - 1),
            position="t"
        )
        
        # Add basic table enhancements
        enhanced_latex = (
            "\\begin{table}[t]\n"
            "\\centering\n"
            "\\small\n"
            + latex_code +
            "\\end{table}\n"
        )
        
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(enhanced_latex)
            
        logger.info(f"LaTeX table saved to {tex_path}")

    def export_markdown_table(self, df: pd.DataFrame, file_prefix: str):
        """Converts DataFrame to a clean Markdown table."""
        md_path = os.path.join(self.output_dir, f"{file_prefix}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
        logger.info(f"Markdown table saved to {md_path}")

    def export_all_reports(self, ablation_df: pd.DataFrame, baseline_df: pd.DataFrame = None):
        """Generates all standard tables for the paper draft."""
        # 1. Ablation Study Table
        self.export_markdown_table(ablation_df, "ablation_study")
        self.export_latex_table(
            ablation_df,
            "ablation_study",
            "Ablation study showing contributions of GraphRAG, Monte Carlo (MC), and Privacy Vector",
            "ablation"
        )
        
        # 2. Main Performance Table (extracted from ablation or baseline summary)
        main_perf_df = ablation_df[["Setting", "AUC-PR", "F1-Score", "Recall", "ECE", "Avg Latency (ms)"]].copy()
        self.export_markdown_table(main_perf_df, "main_performance")
        self.export_latex_table(
            main_perf_df,
            "main_performance",
            "Performance and reliability comparison of different pipeline stages",
            "main_perf"
        )

        # 3. Latency & Resource Profiling Table
        if "Comm (Bytes)" in ablation_df.columns:
            latency_df = ablation_df[["Setting", "Avg Latency (ms)", "Comm (Bytes)"]].copy()
            self.export_markdown_table(latency_df, "latency_profiling")
            self.export_latex_table(
                latency_df,
                "latency_profiling",
                "Computational latency and network communication overhead profiling",
                "latency"
            )
            
        logger.info("All paper-ready tables successfully generated.")
