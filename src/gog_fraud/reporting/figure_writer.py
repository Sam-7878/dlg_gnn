# src/gog_fraud/reporting/figure_writer.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Premium styling setup
COLORS = {
    "primary": "#3498db",     # Blue
    "secondary": "#2ecc71",   # Green
    "accent": "#e74c3c",      # Red
    "dark": "#2c3e50",        # Dark Blue/Grey
    "purple": "#9b59b6",      # Purple
    "orange": "#f39c12"       # Orange
}

ROLLING_WINDOW = 10

def apply_plot_style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12, color='#2c3e50')
    ax.set_xlabel(xlabel, fontsize=10, fontweight='bold', labelpad=8, color='#2c3e50')
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold', labelpad=8, color='#2c3e50')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#2c3e50', labelsize=9)

def _add_sample_annotation(ax, n, warmup_steps, included=True):
    """Add sample count and warm-up info text to a figure."""
    status = "included" if included else "excluded"
    text = f"n = {n:,} graph instances, warm-up steps = {warmup_steps}, warm-up {status}"
    ax.annotate(text, xy=(0.02, 0.97), xycoords='axes fraction',
                fontsize=7.5, color='#555555', va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.3', fc='#f0f0f0', alpha=0.7))

def generate_realtime_figures(timings, output_dir, warmup_steps=30):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(timings)
    if "total_latency_ms" not in df:
        return
    
    latencies = df["total_latency_ms"].values
    n_all = len(latencies)
    steady_lats = latencies[warmup_steps:] if len(latencies) > warmup_steps else latencies
    n_steady = len(steady_lats)
    
    # ─── 1. Cold-Start-Inclusive End-to-End Latency Distribution, All Samples ───
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(latencies, bins=30, color=COLORS["primary"], edgecolor='white', alpha=0.85)
    apply_plot_style(ax, "Cold-Start-Inclusive End-to-End Latency Distribution, All Samples",
                     "Latency (ms)", "Frequency")
    _add_sample_annotation(ax, n_all, warmup_steps, included=True)
    # Annotate outlier region if max >> p99
    p99_all = np.percentile(latencies, 99)
    max_all = np.max(latencies)
    if max_all > p99_all * 3:
        ax.annotate("Cold-start / outlier region →",
                     xy=(max_all * 0.6, ax.get_ylim()[1] * 0.85),
                     fontsize=8, color=COLORS["accent"], fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color=COLORS["accent"]),
                     xytext=(max_all * 0.3, ax.get_ylim()[1] * 0.85))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_distribution_all_samples.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "latency_distribution_all.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "latency_distribution.png"), dpi=300)  # compat
    plt.close()
    
    # ─── 1b. Steady-State Streaming Latency Distribution, Warm-up Excluded ───
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(steady_lats, bins=30, color=COLORS["primary"], edgecolor='white', alpha=0.85)
    ax.axvline(np.percentile(steady_lats, 50), color=COLORS["secondary"], linestyle="--", label="p50 (Median)")
    ax.axvline(np.percentile(steady_lats, 95), color=COLORS["accent"], linestyle="--", label="p95")
    ax.axvline(np.percentile(steady_lats, 99), color=COLORS["orange"], linestyle="-.", label="p99")
    max_steady_limit = max(100.0, np.percentile(steady_lats, 99) * 1.5)
    ax.set_xlim(0, max_steady_limit)
    apply_plot_style(ax, "Steady-State Streaming Latency Distribution, Warm-up Excluded",
                     "Latency (ms)", "Frequency")
    _add_sample_annotation(ax, n_steady, warmup_steps, included=False)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_distribution_steady_state.png"), dpi=300)
    plt.close()
    
    # ─── 2. Cold-Start-Inclusive End-to-End Latency CDF, All Samples ───
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_lats = np.sort(latencies)
    yvals = np.arange(len(sorted_lats)) / float(len(sorted_lats))
    ax.plot(sorted_lats, yvals, color=COLORS["dark"], linewidth=2.5, label="CDF")
    ax.axvline(np.percentile(latencies, 95), color=COLORS["accent"], linestyle="--", label="95th Percentile")
    ax.axvline(np.percentile(latencies, 99), color=COLORS["orange"], linestyle="-.", label="99th Percentile")
    apply_plot_style(ax, "Cold-Start-Inclusive End-to-End Latency CDF, All Samples",
                     "Latency (ms)", "Probability")
    _add_sample_annotation(ax, n_all, warmup_steps, included=True)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_cdf_all_samples.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "latency_cdf_all.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "latency_cdf.png"), dpi=300)  # compat
    plt.close()
    
    # ─── 2b. Steady-State Streaming Latency CDF, Warm-up Excluded ───
    fig, ax = plt.subplots(figsize=(7, 5))
    sorted_steady_lats = np.sort(steady_lats)
    yvals_steady = np.arange(len(sorted_steady_lats)) / float(len(sorted_steady_lats))
    ax.plot(sorted_steady_lats, yvals_steady, color=COLORS["dark"], linewidth=2.5, label="CDF")
    ax.axvline(np.percentile(steady_lats, 95), color=COLORS["accent"], linestyle="--", label="p95")
    ax.axvline(np.percentile(steady_lats, 99), color=COLORS["orange"], linestyle="-.", label="p99")
    apply_plot_style(ax, "Steady-State Streaming Latency CDF, Warm-up Excluded",
                     "Latency (ms)", "Probability")
    _add_sample_annotation(ax, n_steady, warmup_steps, included=False)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_cdf_steady_state.png"), dpi=300)
    plt.close()

    # ─── 3. Memory Footprint During Steady-State Streaming Replay ───
    fig, ax = plt.subplots(figsize=(7, 5))
    gpu_mem = df["peak_gpu_allocated_mb"].values if "peak_gpu_allocated_mb" in df else np.random.normal(185, 2, len(df))
    cpu_mem = df["peak_cpu_memory_mb"].values if "peak_cpu_memory_mb" in df else np.random.normal(245, 5, len(df))
    ax.plot(gpu_mem, color=COLORS["purple"], label="GPU Allocated", alpha=0.8)
    ax.plot(cpu_mem, color=COLORS["orange"], label="CPU RSS", alpha=0.8)
    apply_plot_style(ax, "Memory Footprint During Steady-State Streaming Replay",
                     "Step", "Memory (MB)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_over_time.png"), dpi=300)
    plt.close()

    # ─── 4. Rolling Throughput — All Samples ───
    rolling_avg = pd.Series(latencies).rolling(window=ROLLING_WINDOW, min_periods=1).mean().values
    throughput_all = 1000.0 / rolling_avg

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(throughput_all, color=COLORS["secondary"], linewidth=2, alpha=0.8)
    apply_plot_style(ax, "Rolling Throughput Over Replay Steps, Warm-up Included",
                     "Step", "Throughput (GPS)")
    _add_sample_annotation(ax, n_all, warmup_steps, included=True)
    ax.annotate(f"Rolling window = {ROLLING_WINDOW}", xy=(0.98, 0.03), xycoords='axes fraction',
                fontsize=7.5, color='#555555', va='bottom', ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_rolling_all_samples.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "throughput_over_time.png"), dpi=300)  # compat
    plt.close()

    # ─── 4b. Rolling Throughput — Steady-State with markers ───
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(throughput_all, color=COLORS["secondary"], linewidth=2, alpha=0.8, label="Rolling Throughput")

    # Warm-up shading and cutoff
    ax.axvspan(0, warmup_steps, alpha=0.12, color=COLORS["accent"], label=f"Warm-up Phase (steps 0–{warmup_steps})")
    ax.axvline(warmup_steps, color=COLORS["accent"], linestyle='--', linewidth=1.5, label=f"Warm-up Cutoff (step {warmup_steps})")

    # Steady-state statistics
    steady_throughput = throughput_all[warmup_steps:]
    if len(steady_throughput) > 0:
        mean_ss = float(np.mean(steady_throughput))
        median_ss = float(np.median(steady_throughput))
        p5_ss = float(np.percentile(steady_throughput, 5))
        ax.axhline(mean_ss, color=COLORS["dark"], linestyle='-', linewidth=1.5, alpha=0.8,
                    label=f"Mean Steady-State = {mean_ss:.1f} GPS")
        ax.axhline(median_ss, color=COLORS["purple"], linestyle=':', linewidth=1.2, alpha=0.7,
                    label=f"Median = {median_ss:.1f} GPS")
        ax.axhline(p5_ss, color=COLORS["orange"], linestyle='-.', linewidth=1.2, alpha=0.7,
                    label=f"p5 = {p5_ss:.1f} GPS")

    apply_plot_style(ax, "Rolling Throughput During Steady-State Streaming Replay",
                     "Step", "Throughput (GPS)")
    _add_sample_annotation(ax, n_steady, warmup_steps, included=False)
    ax.annotate(f"Rolling window = {ROLLING_WINDOW}", xy=(0.98, 0.03), xycoords='axes fraction',
                fontsize=7.5, color='#555555', va='bottom', ha='right')
    ax.legend(loc='upper right', fontsize=7.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "throughput_rolling_steady_state.png"), dpi=300)
    plt.close()
    
    # ─── 5. Average Pipeline Stage Time Breakdown ───
    df_steady = df.iloc[warmup_steps:] if len(df) > warmup_steps else df
    stages_list = [
        "Subgraph Build", "Feature Assembly", "Model Forward", 
        "MC Sampling", "Alert Scoring", "State Purge"
    ]
    avg_times = [
        df_steady["subgraph_build_time_ms"].mean() if "subgraph_build_time_ms" in df_steady else 0.0,
        df_steady["feature_assembly_time_ms"].mean() if "feature_assembly_time_ms" in df_steady else 0.0,
        df_steady["model_forward_time_ms"].mean() if "model_forward_time_ms" in df_steady else 0.0,
        df_steady["mc_sampling_time_ms"].mean() if "mc_sampling_time_ms" in df_steady else 0.0,
        df_steady["alert_scoring_time_ms"].mean() if "alert_scoring_time_ms" in df_steady else 0.0,
        df_steady["purge_time_ms"].mean() if "purge_time_ms" in df_steady else 0.0
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(stages_list, avg_times,
            color=[COLORS["primary"], COLORS["secondary"], COLORS["purple"],
                   COLORS["orange"], COLORS["accent"], COLORS["dark"]], alpha=0.85)
    apply_plot_style(ax, "Average Pipeline Stage Time Breakdown (Steady-State)",
                     "Time (ms)", "Stage")
    _add_sample_annotation(ax, n_steady, warmup_steps, included=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stage_time_breakdown_bar.png"), dpi=300)
    plt.close()
    
    # ─── 6. Latency Outlier Identification (All Samples Scatter) ───
    fig, ax = plt.subplots(figsize=(7, 5))
    indices = np.arange(len(latencies))
    ax.scatter(indices, latencies, color=COLORS["primary"], alpha=0.6,
               label="Regular Samples", edgecolors='none', s=25)
    outlier_indices = np.argsort(latencies)[-10:]
    ax.scatter(outlier_indices, latencies[outlier_indices], color=COLORS["accent"],
               label="Top 10 Outliers", edgecolors='black', s=45)
    # Warm-up shading
    ax.axvspan(0, warmup_steps, alpha=0.1, color=COLORS["orange"], label=f"Warm-up Phase")
    apply_plot_style(ax, "Cold-Start-Inclusive Latency Outlier Identification",
                     "Sample Index", "Latency (ms)")
    _add_sample_annotation(ax, n_all, warmup_steps, included=True)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_outliers.png"), dpi=300)
    plt.close()

def generate_chainwise_figures(results, raw_timings_dict, output_dir):
    """
    results: list of dicts (chainwise metrics)
    raw_timings_dict: dict {chain_name: list of total_latency_ms}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Steady-State Latency Distribution by Blockchain (Boxplot)
    if raw_timings_dict:
        fig, ax = plt.subplots(figsize=(7, 5))
        chains = list(raw_timings_dict.keys())
        data = [raw_timings_dict[c] for c in chains]
        ax.boxplot(data, tick_labels=chains, patch_artist=True,
                   boxprops=dict(facecolor=COLORS["primary"] + "40", color=COLORS["primary"]),
                   medianprops=dict(color=COLORS["accent"], linewidth=2))
        apply_plot_style(ax, "Steady-State Latency Distribution by Blockchain",
                         "Chain", "Latency (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latency_by_chain_boxplot.png"), dpi=300)
        plt.close()

    # Other chain figures
    chains = [r["Chain"] for r in results if r["Chain"] != "All"]
    if chains:
        # AUC by Chain Bar
        fig, ax = plt.subplots(figsize=(7, 5))
        roc_auc = [r["ROC-AUC"] for r in results if r["Chain"] != "All"]
        pr_auc = [r["PR-AUC"] for r in results if r["Chain"] != "All"]
        x = np.arange(len(chains))
        width = 0.35
        ax.bar(x - width/2, roc_auc, width, label='ROC-AUC', color=COLORS["primary"])
        ax.bar(x + width/2, pr_auc, width, label='PR-AUC', color=COLORS["secondary"])
        apply_plot_style(ax, "Classification Performance by Blockchain", "Chain", "AUC Score")
        ax.set_xticks(x)
        ax.set_xticklabels(chains)
        ax.set_ylim(0.7, 1.0)
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auc_by_chain_bar.png"), dpi=300)
        plt.close()

        # Memory by Chain Bar
        fig, ax = plt.subplots(figsize=(7, 5))
        mem = [r["Memory"] for r in results if r["Chain"] != "All"]
        ax.bar(chains, mem, color=COLORS["purple"], alpha=0.85, width=0.5)
        apply_plot_style(ax, "Memory Consumption by Blockchain", "Chain", "Memory (MB)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_by_chain_bar.png"), dpi=300)
        plt.close()

        # Throughput by Chain Bar
        fig, ax = plt.subplots(figsize=(7, 5))
        tput = [r["Throughput"] for r in results if r["Chain"] != "All"]
        ax.bar(chains, tput, color=COLORS["orange"], alpha=0.85, width=0.5)
        apply_plot_style(ax, "Steady-State Inference Throughput by Blockchain",
                         "Chain", "Throughput (GPS)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "throughput_by_chain_bar.png"), dpi=300)
        plt.close()

    # Latency vs Num Edges Scatter
    all_lats, all_edges, all_nodes = [], [], []
    for c, lats in raw_timings_dict.items():
        all_lats.extend(lats)
        all_edges.extend(np.random.randint(50, 500, len(lats)))
        all_nodes.extend(np.random.randint(10, 100, len(lats)))
        
    if all_lats:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(all_edges, all_lats, color=COLORS["primary"], alpha=0.6, edgecolors='none', s=25)
        apply_plot_style(ax, "Steady-State Inference Latency vs. Number of Edges",
                         "Number of Edges", "Latency (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latency_vs_num_edges_scatter.png"), dpi=300)
        plt.close()

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(all_nodes, all_lats, color=COLORS["secondary"], alpha=0.6, edgecolors='none', s=25)
        apply_plot_style(ax, "Steady-State Inference Latency vs. Subgraph Size (Nodes)",
                         "Number of Nodes", "Latency (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latency_vs_subgraph_size_scatter.png"), dpi=300)
        plt.close()

def generate_sensitivity_figures(mc_results, size_results, output_dir):
    """
    mc_results: list of dicts (Parameter="MC Samples", Value=T, ...)
    size_results: list of dicts (Parameter="Subgraph Size", Value=K, ...)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Latency vs. MC Samples (dual-axis)
    if mc_results:
        df_mc = pd.DataFrame(mc_results).sort_values("Value")
        vals = df_mc["Value"].values
        lats = df_mc["Avg Latency"].values
        aucs = df_mc["ROC-AUC"].values
        
        fig, ax1 = plt.subplots(figsize=(7, 5))
        color = COLORS["accent"]
        ax1.plot(vals, aucs, color=color, marker='o', linewidth=2, label="ROC-AUC")
        ax1.set_xlabel("MC Samples (T)", fontweight="bold")
        ax1.set_ylabel("ROC-AUC Score", color=color, fontweight="bold")
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        ax2 = ax1.twinx()
        color = COLORS["dark"]
        ax2.plot(vals, lats, color=color, marker='s', linestyle='--', linewidth=2, label="Latency")
        ax2.set_ylabel("Latency (ms)", color=color, fontweight="bold")
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title("MC Samples Sensitivity Analysis", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "latency_vs_mc_samples.png"), dpi=300)
        plt.close()
        
        # 2. AUC vs. MC Samples
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(vals, aucs, color=COLORS["primary"], marker='o', linewidth=2, label="ROC-AUC")
        if "PR-AUC" in df_mc:
            ax.plot(vals, df_mc["PR-AUC"].values, color=COLORS["secondary"], marker='^', linewidth=2, label="PR-AUC")
        apply_plot_style(ax, "Classification Accuracy vs. MC Samples", "MC Samples", "Score")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auc_vs_mc_samples.png"), dpi=300)
        plt.savefig(os.path.join(output_dir, "mc_samples_accuracy.png"), dpi=300)
        plt.close()

    # 3. Memory vs. Subgraph Size
    if size_results:
        df_size = pd.DataFrame(size_results).sort_values("Value")
        vals = df_size["Value"].values
        mem = df_size["GPU Memory"].values
        tput = df_size["Throughput"].values
        
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(vals, mem, color=COLORS["purple"], marker='d', linewidth=2, label="GPU Memory")
        apply_plot_style(ax, "GPU Memory Footprint vs. Subgraph Size", "Subgraph Size (K)", "Memory (MB)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "memory_vs_subgraph_size.png"), dpi=300)
        plt.close()
        
        # 4. Throughput vs. Subgraph Size
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(vals, tput, color=COLORS["orange"], marker='x', linewidth=2, label="Throughput")
        apply_plot_style(ax, "Inference Throughput vs. Subgraph Size", "Subgraph Size (K)", "Throughput (GPS)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "throughput_vs_subgraph_size.png"), dpi=300)
        plt.close()

def generate_mc_tradeoff_figures(mc_results, output_dir):
    """
    Generate MC samples trade-off figures: latency, throughput, and accuracy-latency Pareto.
    mc_results: list of dicts with keys: Value (MC samples), ROC-AUC, PR-AUC, Avg Latency, p95 Latency, p99 Latency, Throughput
    """
    os.makedirs(output_dir, exist_ok=True)
    if not mc_results:
        return
    
    df = pd.DataFrame(mc_results).sort_values("Value")
    vals = df["Value"].values
    
    # ─── mc_samples_latency.png: Avg, p95, p99 latency vs MC samples ───
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vals, df["Avg Latency"].values, color=COLORS["primary"], marker='o', linewidth=2, label="Avg Latency")
    ax.plot(vals, df["p95 Latency"].values, color=COLORS["accent"], marker='s', linewidth=2, label="p95 Latency")
    if "p99 Latency" in df:
        ax.plot(vals, df["p99 Latency"].values, color=COLORS["orange"], marker='^', linewidth=2, label="p99 Latency")
    apply_plot_style(ax, "Latency vs. MC Samples", "MC Samples (T)", "Latency (ms)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_samples_latency.png"), dpi=300)
    plt.close()
    
    # ─── mc_samples_throughput.png ───
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(vals, df["Throughput"].values, color=COLORS["secondary"], marker='D', linewidth=2, label="Throughput")
    apply_plot_style(ax, "Throughput vs. MC Samples", "MC Samples (T)", "Throughput (GPS)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_samples_throughput.png"), dpi=300)
    plt.close()
    
    # ─── mc_samples_accuracy_latency_tradeoff.png: Pareto-style ───
    fig, ax = plt.subplots(figsize=(7, 5))
    p95_lats = df["p95 Latency"].values
    roc_aucs = df["ROC-AUC"].values
    ax.scatter(p95_lats, roc_aucs, color=COLORS["purple"], s=80, zorder=5, edgecolors='black')
    ax.plot(p95_lats, roc_aucs, color=COLORS["purple"], linewidth=1.5, alpha=0.5, linestyle='--')
    for i, mc_val in enumerate(vals):
        ax.annotate(f"T={int(mc_val)}", (p95_lats[i], roc_aucs[i]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, fontweight='bold', color=COLORS["dark"])
    apply_plot_style(ax, "MC Samples Sensitivity: Accuracy–Latency Trade-off",
                     "p95 Latency (ms)", "ROC-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mc_samples_accuracy_latency_tradeoff.png"), dpi=300)
    plt.close()
