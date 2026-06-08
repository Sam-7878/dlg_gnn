# -*- coding: utf-8 -*-
"""
DLG-GNN Benchmark: Visualization Generator (PNG and PDF output)
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_style():
    """Configure styling matching publication standards."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

def plot_metric_comparison(summary_df, output_prefix, title):
    """Figure 1: Metric-wise model comparison (Bar Chart)."""
    set_style()
    df_melted = summary_df.melt(
        id_vars=['Model'], 
        value_vars=['mean_roc', 'mean_pr', 'mean_f1'],
        var_name='Metric', 
        value_name='Score'
    )
    
    # Clean Metric labels
    df_melted['Metric'] = df_melted['Metric'].map({
        'mean_roc': 'ROC-AUC',
        'mean_pr': 'PR-AUC',
        'mean_f1': 'F1-Score'
    })
    
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_melted, 
        x='Model', 
        y='Score', 
        hue='Metric',
        palette='muted',
        edgecolor='black',
        linewidth=0.75
    )
    
    plt.title(title, pad=15)
    plt.ylabel('Mean Performance Score')
    plt.xlabel('Model')
    plt.ylim(0.0, 1.05)
    plt.legend(loc='upper right', frameon=True)
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    plt.close()

def plot_rank_heatmap(rank_df, output_prefix, title):
    """Figure 2: Model Rank Heatmap."""
    set_style()
    # Pivot ranks into heatmap-ready format
    heatmap_data = rank_df.set_index('Model')[[
        'mean_rank_roc', 'mean_rank_pr', 'mean_rank_f1', 'mean_rank_composite'
    ]]
    
    # Rename columns to human-readable format
    heatmap_data.columns = ['ROC-AUC Rank', 'PR-AUC Rank', 'F1 Rank', 'Composite Rank']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="YlGnBu_r", # Reverse since lower rank is better
        cbar_kws={'label': 'Mean Rank (Lower is Better)'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title(title, pad=15)
    plt.ylabel('Model')
    plt.xticks(rotation=15)
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    plt.close()

def plot_homophily_scatter(perf_meta, metric, output_prefix, title, selected_only=False):
    """Figure 3 & 4: Homophily vs. Performance Scatter Plot."""
    set_style()
    
    df_plot = perf_meta.copy()
    if selected_only:
        selected_models = ['DOMINANT', 'AnomalyDAE', 'CoLA', 'OCGNN', 'DLG']
        df_plot = df_plot[df_plot['Model'].isin(selected_models)]
        
    plt.figure(figsize=(9, 6))
    
    # Mappings for styling
    domain_markers = {
        "Financial & Blockchain Fraud": "o",
        "Social/Sybil": "s",
        "Citation/Homophilous": "^"
    }
    
    family_colors = {
        "Reconstruction": "#1f77b4",
        "Contrastive": "#ff7f0e",
        "Contrastive/Augmented": "#2ca02c",
        "Neighborhood Reconstruction": "#d62728",
        "One-Class": "#9467bd",
        "Decoupled": "#bcbd22"
    }
    
    # Add temporary styling columns
    from utils import MODEL_FAMILY_MAP
    df_plot['ModelFamily'] = df_plot['Model'].map(MODEL_FAMILY_MAP)
    
    # Draw scatter plot manually for exact marker/color control
    for group, marker in domain_markers.items():
        sub = df_plot[df_plot['DomainGroup'] == group]
        if sub.empty:
            continue
        
        # Determine colors from family mappings
        colors = sub['ModelFamily'].map(family_colors).fillna('#7f7f7f').tolist()
        
        plt.scatter(
            sub['EdgeHomophily'], 
            sub[metric], 
            marker=marker, 
            s=80, 
            c=colors, 
            label=group, 
            edgecolors='black', 
            linewidths=0.75, 
            alpha=0.85
        )
        
    plt.title(title, pad=15)
    plt.xlabel('Edge Homophily (h)')
    plt.ylabel(metric)
    plt.xlim(-0.05, 1.05)
    plt.ylim(0.0, 1.05)
    
    # Legend for domain groups (markers)
    legend1 = plt.legend(title="Dataset Domain", loc="upper left", frameon=True)
    plt.gca().add_artist(legend1)
    
    # Custom legend for Model families (colors)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=col, label=fam, markersize=10, markeredgecolor='black')
        for fam, col in family_colors.items() if fam in df_plot['ModelFamily'].unique()
    ]
    plt.legend(handles=legend_elements, title="Model Family", loc="lower left", frameon=True)
    
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    plt.close()

def plot_correlation_heatmap(corr_df, output_prefix, title):
    """Figure 5: Correlation Heatmap (Spearman rho + significance marks)."""
    set_style()
    
    # Pivot Spearman rho & p-value
    rho_pivot = corr_df.pivot(index='Model', columns='Y', values='spearman_rho')
    p_pivot = corr_df.pivot(index='Model', columns='Y', values='spearman_p')
    
    # Reorder columns
    cols_order = ['roc', 'pr', 'f1', 'composite_score']
    cols_order = [c for c in cols_order if c in rho_pivot.columns]
    rho_pivot = rho_pivot[cols_order]
    p_pivot = p_pivot[cols_order]
    
    # Rename columns for heatmap
    label_map = {
        'roc': 'ROC-AUC',
        'pr': 'PR-AUC',
        'f1': 'F1-Score',
        'composite_score': 'Fraud Composite'
    }
    rho_pivot.columns = [label_map[c] for c in rho_pivot.columns]
    p_pivot.columns = [label_map[c] for c in p_pivot.columns]
    
    # Generate labels containing Spearman rho and significance stars
    annot_data = rho_pivot.copy().astype(str)
    for r_idx in rho_pivot.index:
        for c_idx in rho_pivot.columns:
            rho_val = rho_pivot.loc[r_idx, c_idx]
            p_val = p_pivot.loc[r_idx, c_idx]
            
            if pd.isna(rho_val):
                annot_data.loc[r_idx, c_idx] = 'N/A'
                continue
                
            stars = ''
            if p_val < 0.001:
                stars = '***'
            elif p_val < 0.01:
                stars = '**'
            elif p_val < 0.05:
                stars = '*'
            annot_data.loc[r_idx, c_idx] = f"{rho_val:.2f}{stars}"
            
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        rho_pivot, 
        annot=annot_data.values, 
        fmt="", 
        cmap="coolwarm", 
        vmin=-1.0, 
        vmax=1.0,
        cbar_kws={'label': 'Spearman Correlation ($\\rho$)'},
        linewidths=0.5,
        linecolor='black'
    )
    plt.title(title, pad=15)
    plt.ylabel('Model')
    plt.tight_layout()
    
    plt.savefig(f"{output_prefix}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_prefix}.pdf", bbox_inches='tight')
    plt.close()
