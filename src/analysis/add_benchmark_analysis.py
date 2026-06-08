# -*- coding: utf-8 -*-
"""
DLG-GNN Benchmark Expansion Analysis Main Coordinator
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, friedmanchisquare

# Add local path to sys for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DOMAIN_MAP, DOMAIN_GROUP_MAP, MODEL_FAMILY_MAP, normalize_columns
import plot_benchmark_analysis as pba

def parse_args():
    parser = argparse.ArgumentParser(description="DLG-GNN Benchmark Expansion")
    parser.add_argument('--input_csv', type=str, required=True, help="Path to input benchmark CSV")
    parser.add_argument('--metadata_csv', type=str, required=True, help="Path to topology metadata CSV")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory for CSV results")
    parser.add_argument('--table_dir', type=str, required=True, help="Output directory for LaTeX/Markdown tables")
    parser.add_argument('--figure_dir', type=str, required=True, help="Output directory for PNG/PDF figures")
    return parser.parse_args()

def corr_test(x, y):
    """Compute Pearson and Spearman correlation coefficient along with p-values."""
    valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
    x_valid = x[valid]
    y_valid = y[valid]
    
    if len(x_valid) < 3:
        return {
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "n": len(x_valid)
        }
        
    pr = pearsonr(x_valid, y_valid)
    sr = spearmanr(x_valid, y_valid)
    return {
        "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
        "spearman_rho": sr.statistic, "spearman_p": sr.pvalue,
        "n": len(x_valid)
    }

def save_latex_and_md(df, file_name, table_dir, title_cols=None, format_dict=None, bold_best_col=None, bold_lowest=False):
    """Helper to save clean Markdown and publication-quality LaTeX tables."""
    os.makedirs(table_dir, exist_ok=True)
    md_path = os.path.join(table_dir, f"{file_name}.md")
    tex_path = os.path.join(table_dir, f"{file_name}.tex")
    
    # Write markdown
    df.to_markdown(md_path, index=False)
    
    # Construct booktabs LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("  \\centering")
    
    cols = df.columns
    col_align = "l" * len(cols)
    latex_lines.append(f"  \\begin{{tabular}}{{{col_align}}}")
    latex_lines.append("    \\toprule")
    
    # Headers
    header_line = " & ".join([str(c).replace('_', '\\_').replace('%', '\\%') for c in cols]) + " \\\\"
    latex_lines.append(f"    {header_line}")
    latex_lines.append("    \\midrule")
    
    # Rows
    for _, row in df.iterrows():
        row_vals = []
        for col in cols:
            val = row[col]
            
            # Format numeric values
            if format_dict and col in format_dict and isinstance(val, (int, float)) and not pd.isna(val):
                formatted = format_dict[col].format(val)
            else:
                formatted = str(val).replace('_', '\\_').replace('%', '\\%')
            
            # Highlight best values
            if bold_best_col and col == bold_best_col:
                # Highlight logic (assumed grouped formatting done beforehand for simplicity)
                pass
            row_vals.append(formatted)
            
        line_str = " & ".join(row_vals) + " \\\\"
        latex_lines.append(f"    {line_str}")
        
    latex_lines.append("    \\bottomrule")
    latex_lines.append("  \\end{tabular}")
    latex_lines.append(f"  \\caption{{Benchmark Table: {file_name.replace('_', ' ')}}}")
    latex_lines.append(f"  \\label{{tab:{file_name}}}")
    latex_lines.append("\\end{table}")
    
    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines))

def main():
    args = parse_args()
    
    # Ensure all output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.table_dir, exist_ok=True)
    os.makedirs(args.figure_dir, exist_ok=True)
    
    # ── 1. Load and Clean Data ──
    print("Loading benchmark results...")
    df_raw = pd.read_csv(args.input_csv)
    df = normalize_columns(df_raw)
    
    # Inject mappings
    df['Domain'] = df['Dataset'].map(DOMAIN_MAP)
    df['DomainGroup'] = df['Dataset'].map(DOMAIN_GROUP_MAP)
    df['ModelFamily'] = df['Model'].map(MODEL_FAMILY_MAP)
    
    # Verify mappings
    missing_domains = df[df['DomainGroup'].isna()]['Dataset'].unique()
    if len(missing_domains) > 0:
        print(f"    [Warning] Datasets missing domain group mappings: {missing_domains}")
        
    # ── 2. Add Metrics Rankings ──
    # Ranks are calculated per dataset, descending for scores, ascending for resources
    df['rank_roc'] = df.groupby('Dataset')['ROC-AUC'].rank(ascending=False, method='average')
    df['rank_pr'] = df.groupby('Dataset')['PR-AUC'].rank(ascending=False, method='average')
    df['rank_f1'] = df.groupby('Dataset')['F1-Score'].rank(ascending=False, method='average')
    df['rank_composite'] = df[['rank_roc', 'rank_pr', 'rank_f1']].mean(axis=1)
    
    # Calculate fraud-oriented weighted score and rank
    df['fraud_composite_score'] = 0.25 * df['ROC-AUC'] + 0.40 * df['PR-AUC'] + 0.35 * df['F1-Score']
    df['fraud_composite_rank'] = 0.25 * df['rank_roc'] + 0.40 * df['rank_pr'] + 0.35 * df['rank_f1']
    
    # ── 3. Domain-Wise Ranking Analysis ──
    print("Computing domain-wise rankings...")
    domain_rank_agg = df.groupby(['DomainGroup', 'Model']).agg(
        mean_rank_roc=('rank_roc', 'mean'),
        mean_rank_pr=('rank_pr', 'mean'),
        mean_rank_f1=('rank_f1', 'mean'),
        mean_rank_composite=('rank_composite', 'mean'),
        mean_roc=('ROC-AUC', 'mean'),
        mean_pr=('PR-AUC', 'mean'),
        mean_f1=('F1-Score', 'mean'),
        n_datasets=('Dataset', 'nunique')
    ).reset_index().sort_values(by=['DomainGroup', 'mean_rank_composite'], ascending=[True, True])
    
    domain_rank_fine = df.groupby(['Domain', 'Model']).agg(
        mean_rank_roc=('rank_roc', 'mean'),
        mean_rank_pr=('rank_pr', 'mean'),
        mean_rank_f1=('rank_f1', 'mean'),
        mean_rank_composite=('rank_composite', 'mean'),
        mean_roc=('ROC-AUC', 'mean'),
        mean_pr=('PR-AUC', 'mean'),
        mean_f1=('F1-Score', 'mean'),
        n_datasets=('Dataset', 'nunique')
    ).reset_index().sort_values(by=['Domain', 'mean_rank_composite'], ascending=[True, True])
    
    # Save CSV files
    domain_rank_agg.to_csv(os.path.join(args.output_dir, "domain_wise_ranking_aggregated.csv"), index=False)
    domain_rank_fine.to_csv(os.path.join(args.output_dir, "domain_wise_ranking_fine_grained.csv"), index=False)
    
    # Generate tables
    fmt_rank = {
        'mean_rank_roc': '{:.2f}', 'mean_rank_pr': '{:.2f}', 'mean_rank_f1': '{:.2f}', 
        'mean_rank_composite': '{:.2f}', 'mean_roc': '{:.4f}', 'mean_pr': '{:.4f}', 'mean_f1': '{:.4f}'
    }
    save_latex_and_md(domain_rank_agg, "table_domain_wise_ranking", args.table_dir, format_dict=fmt_rank)
    
    # ── 4. Metric-Wise Analysis ──
    print("Performing metric-wise evaluation summary...")
    metric_summary_all = df.groupby('Model').agg(
        mean_roc=('ROC-AUC', 'mean'),
        std_roc=('ROC-AUC', 'std'),
        mean_pr=('PR-AUC', 'mean'),
        std_pr=('PR-AUC', 'std'),
        mean_f1=('F1-Score', 'mean'),
        std_f1=('F1-Score', 'std'),
        mean_time=('Time (s)', 'mean'),
        mean_ram=('Peak RAM (MB)', 'mean'),
        mean_vram=('Peak VRAM (MB)', 'mean')
    ).reset_index()
    
    # Assign ranks
    metric_summary_all['rank_mean_roc'] = metric_summary_all['mean_roc'].rank(ascending=False, method='average')
    metric_summary_all['rank_mean_pr'] = metric_summary_all['mean_pr'].rank(ascending=False, method='average')
    metric_summary_all['rank_mean_f1'] = metric_summary_all['mean_f1'].rank(ascending=False, method='average')
    metric_summary_all['rank_efficiency_time'] = metric_summary_all['mean_time'].rank(ascending=True, method='average')
    metric_summary_all['rank_efficiency_vram'] = metric_summary_all['mean_vram'].rank(ascending=True, method='average')
    
    metric_summary_all.to_csv(os.path.join(args.output_dir, "metric_wise_summary_all.csv"), index=False)
    
    # Fraud only summary
    fraud_df = df[df['DomainGroup'] == "Financial & Blockchain Fraud"]
    metric_summary_fraud = fraud_df.groupby('Model').agg(
        mean_roc=('ROC-AUC', 'mean'),
        std_roc=('ROC-AUC', 'std'),
        mean_pr=('PR-AUC', 'mean'),
        std_pr=('PR-AUC', 'std'),
        mean_f1=('F1-Score', 'mean'),
        std_f1=('F1-Score', 'std'),
        mean_time=('Time (s)', 'mean'),
        mean_ram=('Peak RAM (MB)', 'mean'),
        mean_vram=('Peak VRAM (MB)', 'mean')
    ).reset_index()
    
    metric_summary_fraud['rank_mean_roc'] = metric_summary_fraud['mean_roc'].rank(ascending=False, method='average')
    metric_summary_fraud['rank_mean_pr'] = metric_summary_fraud['mean_pr'].rank(ascending=False, method='average')
    metric_summary_fraud['rank_mean_f1'] = metric_summary_fraud['mean_f1'].rank(ascending=False, method='average')
    metric_summary_fraud['rank_efficiency_time'] = metric_summary_fraud['mean_time'].rank(ascending=True, method='average')
    metric_summary_fraud['rank_efficiency_vram'] = metric_summary_fraud['mean_vram'].rank(ascending=True, method='average')
    
    metric_summary_fraud.to_csv(os.path.join(args.output_dir, "metric_wise_summary_fraud_only.csv"), index=False)
    
    # Fraud composite rank summary
    fraud_composite_summary = df.groupby('Model').agg(
        mean_fraud_composite_score=('fraud_composite_score', 'mean'),
        mean_fraud_composite_rank=('fraud_composite_rank', 'mean')
    ).reset_index().sort_values(by='mean_fraud_composite_rank', ascending=True)
    
    fraud_composite_summary.to_csv(os.path.join(args.output_dir, "fraud_oriented_composite_rank.csv"), index=False)
    
    # Generate LaTeX tables for metrics
    fmt_metric = {
        'mean_roc': '{:.4f}', 'std_roc': '{:.4f}', 'mean_pr': '{:.4f}', 'std_pr': '{:.4f}',
        'mean_f1': '{:.4f}', 'std_f1': '{:.4f}', 'mean_time': '{:.2f}', 'mean_ram': '{:.1f}', 'mean_vram': '{:.1f}'
    }
    save_latex_and_md(metric_summary_all, "table_metric_wise_summary_all", args.table_dir, format_dict=fmt_metric)
    save_latex_and_md(metric_summary_fraud, "table_metric_wise_summary_fraud_only", args.table_dir, format_dict=fmt_metric)
    save_latex_and_md(fraud_composite_summary, "table_fraud_oriented_composite_rank", args.table_dir, 
                      format_dict={'mean_fraud_composite_score': '{:.4f}', 'mean_fraud_composite_rank': '{:.2f}'})
    
    # ── 5. Domain Degradation (Citation vs. Fraud) ──
    print("Computing performance degradation from citation to fraud domains...")
    citation_mean = df[df['DomainGroup'] == "Citation/Homophilous"].groupby('Model')[['ROC-AUC', 'PR-AUC', 'F1-Score']].mean()
    fraud_mean = df[df['DomainGroup'] == "Financial & Blockchain Fraud"].groupby('Model')[['ROC-AUC', 'PR-AUC', 'F1-Score']].mean()
    
    degradation = citation_mean - fraud_mean
    degradation.columns = ['roc_degradation', 'pr_degradation', 'f1_degradation']
    degradation = degradation.reset_index().sort_values(by='f1_degradation', ascending=True)
    
    degradation.to_csv(os.path.join(args.output_dir, "domain_degradation_citation_to_fraud.csv"), index=False)
    save_latex_and_md(degradation, "table_domain_degradation", args.table_dir, 
                      format_dict={'roc_degradation': '{:.4f}', 'pr_degradation': '{:.4f}', 'f1_degradation': '{:.4f}'})
    
    # ── 6. Homophily-Performance Correlation Tests ──
    print("Conducting homophily correlation test...")
    if not os.path.exists(args.metadata_csv):
        print(f"    [Warning] Metadata file {args.metadata_csv} not found! Skipping correlation tests.")
        corr_df = pd.DataFrame()
    else:
        metadata = pd.read_csv(args.metadata_csv)
        # Verify topology columns
        metadata = normalize_columns(metadata)
        if 'EdgeHomophily' not in metadata.columns:
            # Fallback mapper
            print("    [Warning] EdgeHomophily not in metadata columns. Checking alternatives.")
            
        perf_meta = df.merge(metadata, on="Dataset", how="left")
        
        # Check for missing homophily values
        missing_homo = perf_meta[perf_meta['EdgeHomophily'].isna()]['Dataset'].unique()
        if len(missing_homo) > 0:
            print(f"    [Warning] Datasets missing homophily values in metadata: {missing_homo}")
        
        # Model-wise correlation
        model_perf_by_dataset = perf_meta.groupby(["Dataset", "Model", "EdgeHomophily", "AnomalyRatio", "DomainGroup"]).agg(
            roc=("ROC-AUC", "mean"),
            pr=("PR-AUC", "mean"),
            f1=("F1-Score", "mean"),
            composite_score=("fraud_composite_score", "mean"),
            composite_rank=("fraud_composite_rank", "mean")
        ).reset_index()
        
        corr_records = []
        for model, sub in model_perf_by_dataset.groupby("Model"):
            for metric_col in ["roc", "pr", "f1", "composite_score"]:
                res = corr_test(
                    sub["EdgeHomophily"].to_numpy(dtype=float),
                    sub[metric_col].to_numpy(dtype=float)
                )
                corr_records.append({
                    "Model": model,
                    "X": "EdgeHomophily",
                    "Y": metric_col,
                    **res
                })
        
        corr_df = pd.DataFrame(corr_records)
        corr_df.to_csv(os.path.join(args.output_dir, "homophily_performance_correlation_by_model.csv"), index=False)
        save_latex_and_md(corr_df, "table_homophily_correlation", args.table_dir,
                          format_dict={'pearson_r': '{:.4f}', 'pearson_p': '{:.4f}', 'spearman_rho': '{:.4f}', 'spearman_p': '{:.4f}'})
        
        # Pooled correlation
        pooled_records = []
        for metric_col in ["roc", "pr", "f1", "composite_score"]:
            res = corr_test(
                model_perf_by_dataset["EdgeHomophily"].to_numpy(dtype=float),
                model_perf_by_dataset[metric_col].to_numpy(dtype=float)
            )
            pooled_records.append({
                "X": "EdgeHomophily",
                "Y": metric_col,
                **res
            })
        pooled_df = pd.DataFrame(pooled_records)
        pooled_df.to_csv(os.path.join(args.output_dir, "homophily_performance_correlation_pooled.csv"), index=False)
        
        # Family-wise correlation
        model_perf_by_dataset['ModelFamily'] = model_perf_by_dataset['Model'].map(MODEL_FAMILY_MAP)
        family_records = []
        for family, sub in model_perf_by_dataset.groupby("ModelFamily"):
            for metric_col in ["roc", "pr", "f1", "composite_score"]:
                res = corr_test(
                    sub["EdgeHomophily"].to_numpy(dtype=float),
                    sub[metric_col].to_numpy(dtype=float)
                )
                family_records.append({
                    "ModelFamily": family,
                    "X": "EdgeHomophily",
                    "Y": metric_col,
                    **res
                })
        family_df = pd.DataFrame(family_records)
        family_df.to_csv(os.path.join(args.output_dir, "homophily_performance_correlation_by_family.csv"), index=False)

    # ── 7. Friedman & Nemenyi Statistical Tests ──
    print("Running Friedman statistical ranking tests...")
    friedman_records = []
    for m_label, m_col in [("ROC-AUC", "ROC-AUC"), ("PR-AUC", "PR-AUC"), ("F1-Score", "F1-Score"), ("Fraud Composite Rank", "fraud_composite_rank")]:
        # Pivot dataset
        pivot = df.pivot(index="Dataset", columns="Model", values=m_col)
        # Friedman test needs clean dataset without missing entries. Handled via dropna()
        pivot_clean = pivot.dropna()
        if pivot_clean.shape[1] > 1 and pivot_clean.shape[0] >= 3:
            try:
                stat, p = friedmanchisquare(*[pivot_clean[col].to_numpy() for col in pivot_clean.columns])
                friedman_records.append({
                    "Metric": m_label,
                    "Statistic": stat,
                    "p-value": p,
                    "n_datasets": pivot_clean.shape[0],
                    "Significant": p < 0.05
                })
            except Exception as e:
                print(f"      [Error] Friedman test failed for {m_label}: {e}")
        else:
            print(f"      [Warning] Insufficient data for Friedman test on {m_label}")
            
    friedman_df = pd.DataFrame(friedman_records)
    friedman_df.to_csv(os.path.join(args.output_dir, "friedman_test_results.csv"), index=False)
    
    # Nemenyi Post-hoc Test (Optional)
    try:
        import scikit_posthocs as sp
        print("Running Nemenyi pairwise post-hoc test...")
        for m_label, m_col in [("PR-AUC", "PR-AUC"), ("F1-Score", "F1-Score")]:
            pivot = df.pivot(index="Dataset", columns="Model", values=m_col).dropna()
            if pivot.shape[0] >= 3:
                nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
                nemenyi.columns = pivot.columns
                nemenyi.index = pivot.columns
                out_name = f"nemenyi_{m_label.lower().replace('-', '_')}.csv"
                nemenyi.to_csv(os.path.join(args.output_dir, out_name))
    except ImportError:
        print("    [Info] scikit-posthocs is not installed. Skipping Nemenyi test.")
        
    # ── 8. Call Visualizations ──
    if not corr_df.empty:
        print("Generating Figures...")
        # Create output figure prefix paths
        m_all_prefix = os.path.join(args.figure_dir, "metric_wise_comparison_all")
        m_fraud_prefix = os.path.join(args.figure_dir, "metric_wise_comparison_fraud_only")
        h_all_prefix = os.path.join(args.figure_dir, "rank_heatmap_all")
        h_fraud_prefix = os.path.join(args.figure_dir, "rank_heatmap_fraud_only")
        sc_pr_all_prefix = os.path.join(args.figure_dir, "homophily_vs_pr_auc_all_models")
        sc_pr_sel_prefix = os.path.join(args.figure_dir, "homophily_vs_pr_auc_selected_models")
        sc_f1_all_prefix = os.path.join(args.figure_dir, "homophily_vs_f1_all_models")
        sc_f1_sel_prefix = os.path.join(args.figure_dir, "homophily_vs_f1_selected_models")
        corr_heatmap_prefix = os.path.join(args.figure_dir, "homophily_correlation_heatmap_spearman")
        
        # Draw all plots
        pba.plot_metric_comparison(metric_summary_all, m_all_prefix, "Metric-Wise Performance Comparison (All Datasets)")
        pba.plot_metric_comparison(metric_summary_fraud, m_fraud_prefix, "Metric-Wise Performance Comparison (Financial & Blockchain Fraud)")
        
        # Heatmap ranks
        # We need model-wise mean ranks for Figure 2
        mean_ranks_all = df.groupby('Model').agg(
            mean_rank_roc=('rank_roc', 'mean'),
            mean_rank_pr=('rank_pr', 'mean'),
            mean_rank_f1=('rank_f1', 'mean'),
            mean_rank_composite=('rank_composite', 'mean')
        ).reset_index()
        mean_ranks_fraud = fraud_df.groupby('Model').agg(
            mean_rank_roc=('rank_roc', 'mean'),
            mean_rank_pr=('rank_pr', 'mean'),
            mean_rank_f1=('rank_f1', 'mean'),
            mean_rank_composite=('rank_composite', 'mean')
        ).reset_index()
        
        pba.plot_rank_heatmap(mean_ranks_all, h_all_prefix, "Model Ranks Heatmap (All Datasets)")
        pba.plot_rank_heatmap(mean_ranks_fraud, h_fraud_prefix, "Model Ranks Heatmap (Financial & Blockchain Fraud)")
        
        # Homophily scatter plots
        pba.plot_homophily_scatter(perf_meta, 'PR-AUC', sc_pr_all_prefix, "Edge Homophily vs. PR-AUC (All Models)")
        pba.plot_homophily_scatter(perf_meta, 'PR-AUC', sc_pr_sel_prefix, "Edge Homophily vs. PR-AUC (Selected Models)", selected_only=True)
        pba.plot_homophily_scatter(perf_meta, 'F1-Score', sc_f1_all_prefix, "Edge Homophily vs. F1-Score (All Models)")
        pba.plot_homophily_scatter(perf_meta, 'F1-Score', sc_f1_sel_prefix, "Edge Homophily vs. F1-Score (Selected Models)", selected_only=True)
        
        # Correlation heatmap
        pba.plot_correlation_heatmap(corr_df, corr_heatmap_prefix, "Spearman Correlation between Edge Homophily and Performance")
        
    print("Done! Analysis completed successfully.")

if __name__ == '__main__':
    main()
