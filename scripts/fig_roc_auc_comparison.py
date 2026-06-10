import matplotlib.pyplot as plt
import numpy as np

chains = ["BSC", "Ethereum", "Polygon"]

best_gog = [0.770702, 0.770965, 0.744348]
rev_l1 = [0.783290, 0.918330, 0.884203]
rev_l1_l2 = [0.807828, 0.926587, 0.921304]
rev_full = [0.833290, 0.925867, 0.904928]

models = ["Best GoG", "L1", "L1+L2", "Full"]
values = [best_gog, rev_l1, rev_l1_l2, rev_full]

x = np.arange(len(chains))
width = 0.20

plt.figure(figsize=(7.0, 3.2))

colors = ["#6c757d", "#4c78a8", "#72b7b2", "#54a24b"]

for i, (model, vals) in enumerate(zip(models, values)):
    bars = plt.bar(x + (i - 1.5) * width, vals, width, label=model, color=colors[i])
    for bar, val in zip(bars, vals):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8
        )

plt.xticks(x, chains, fontsize=9)
plt.ylabel("ROC-AUC", fontsize=9)
plt.ylim(0.68, 0.98)
plt.grid(axis="y", linestyle="--", alpha=0.35)
plt.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.18))
plt.tight_layout()

plt.savefig("./dlg_gnn/docs/work_reports/20-benchmark_visualization_and_comparison/fig_roc_auc_comparison.pdf", bbox_inches="tight")
plt.savefig("./dlg_gnn/docs/work_reports/20-benchmark_visualization_and_comparison/fig_roc_auc_comparison.png", dpi=300, bbox_inches="tight")
plt.show()
