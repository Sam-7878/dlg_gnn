"""Build the chain-stratified empirical risk--coverage figure."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    evidence = pd.read_csv("results/sci_v3_submission/canonical/canonical_risk_control.csv")
    evidence = evidence[evidence.metric_defined.astype(bool)]
    output = Path("results/sci_v3_submission_r2/manuscript/figures")
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(6.3, 4.0))
    for chain, group in evidence.groupby("chain"):
        summary = group.groupby("target_alpha")[["coverage", "observed_FNR", "exact_upper_bound"]].mean().reset_index()
        axis.plot(summary.coverage, summary.observed_FNR, marker="o", label=f"{chain}: observed")
        axis.plot(summary.coverage, summary.exact_upper_bound, linestyle="--", alpha=.75, label=f"{chain}: exact upper bound")
    axis.set(xlabel="Direct-exit coverage", ylabel="Fraud miss risk", ylim=(0, None))
    axis.legend(fontsize=7, ncol=2); figure.tight_layout()
    figure.savefig(output / "figure_risk_coverage.pdf", bbox_inches="tight")
    figure.savefig(output / "figure_risk_coverage.png", dpi=200, bbox_inches="tight")
    print(f"wrote {len(evidence)} risk-control rows")


if __name__ == "__main__":
    main()
