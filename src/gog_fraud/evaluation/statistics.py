from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon


def paired_bootstrap_difference(y_true, score_a, score_b, metric: Callable, *, iterations: int = 2000, seed: int = 42, confidence: float = 0.95) -> dict[str, float]:
    y = np.asarray(y_true); a = np.asarray(score_a); b = np.asarray(score_b)
    if not (len(y) == len(a) == len(b)) or not len(y):
        raise ValueError("paired non-empty arrays are required")
    rng = np.random.default_rng(seed); differences = []
    for _ in range(iterations):
        index = rng.integers(0, len(y), len(y))
        try: differences.append(float(metric(y[index], a[index]) - metric(y[index], b[index])))
        except ValueError: continue
    if not differences:
        raise ValueError("metric was undefined for every bootstrap sample")
    values = np.asarray(differences); alpha = (1 - confidence) / 2
    return {"difference": float(metric(y, a) - metric(y, b)), "ci_low": float(np.quantile(values, alpha)), "ci_high": float(np.quantile(values, 1 - alpha)), "p_two_sided": float(2 * min(np.mean(values <= 0), np.mean(values >= 0))), "iterations_valid": int(len(values)), "seed": seed}


def paired_effect_size(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    differences = np.asarray(values_a, dtype=float) - np.asarray(values_b, dtype=float)
    if len(differences) < 2 or differences.std(ddof=1) == 0:
        return 0.0
    return float(differences.mean() / differences.std(ddof=1))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float); order = np.argsort(values); adjusted = np.empty(len(values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index]); adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def aggregate_seed_results(frame, *, metric: str, value_name: str = "value"):
    """Return one observation per dataset/model before cross-dataset tests."""
    import pandas as pd

    required = {"dataset", "model", "seed", metric}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns for seed aggregation: {sorted(missing)}")
    clean = frame.loc[frame[metric].notna(), ["dataset", "model", "seed", metric]].copy()
    duplicate = clean.duplicated(["dataset", "model", "seed"], keep=False)
    if duplicate.any():
        raise ValueError("duplicate dataset/model/seed observations would cause pseudoreplication")
    return clean.groupby(["dataset", "model"], as_index=False)[metric].mean().rename(columns={metric: value_name})


def friedman_dataset_test(frame, *, metric: str) -> dict[str, object]:
    """Friedman test on dataset-level seed means with complete model blocks."""
    aggregate = aggregate_seed_results(frame, metric=metric)
    matrix = aggregate.pivot(index="dataset", columns="model", values="value").dropna(axis=0)
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        raise ValueError("Friedman test requires >=2 complete datasets and >=3 models")
    statistic, p_value = friedmanchisquare(*(matrix[column].to_numpy() for column in matrix.columns))
    ranks = matrix.rank(axis=1, method="average", ascending=False)
    return {
        "test": "friedman",
        "metric": metric,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_datasets": int(matrix.shape[0]),
        "n_models": int(matrix.shape[1]),
        "models": list(matrix.columns),
        "datasets": list(matrix.index),
        "average_ranks": {name: float(value) for name, value in ranks.mean().items()},
    }


def paired_model_tests(frame, *, metric: str, correction: str = "holm"):
    """Pairwise dataset-blocked Wilcoxon tests after seed aggregation."""
    import pandas as pd

    aggregate = aggregate_seed_results(frame, metric=metric)
    matrix = aggregate.pivot(index="dataset", columns="model", values="value")
    records: list[dict[str, object]] = []
    for model_a, model_b in combinations(matrix.columns, 2):
        pair = matrix[[model_a, model_b]].dropna()
        a, b = pair[model_a].to_numpy(), pair[model_b].to_numpy()
        if len(pair) < 2:
            statistic, p_value = float("nan"), float("nan")
        elif np.allclose(a, b):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
        records.append({
            "comparison": f"{model_a} vs {model_b}", "model_a": model_a,
            "model_b": model_b, "metric": metric, "test": "wilcoxon",
            "statistic": float(statistic), "p_value": float(p_value),
            "n_datasets": int(len(pair)), "effect_size_dz": paired_effect_size(a, b),
        })
    valid = [record["p_value"] for record in records if np.isfinite(record["p_value"])]
    adjusted = holm_adjust(valid) if correction == "holm" else valid
    cursor = 0
    for record in records:
        if np.isfinite(record["p_value"]):
            record["adjusted_p"] = float(adjusted[cursor]); cursor += 1
        else:
            record["adjusted_p"] = float("nan")
        record["significant"] = bool(np.isfinite(record["adjusted_p"]) and record["adjusted_p"] < 0.05)
        record["correction"] = correction
    return pd.DataFrame.from_records(records)


def spearman_with_bootstrap(
    x: Sequence[float], y: Sequence[float], *, iterations: int = 2000,
    seed: int = 42, confidence: float = 0.95,
) -> dict[str, float | int]:
    """Spearman association with a dataset-resampling percentile interval."""
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr, y_arr = x_arr[valid], y_arr[valid]
    if len(x_arr) < 3 or np.unique(x_arr).size < 2 or np.unique(y_arr).size < 2:
        raise ValueError("Spearman analysis requires >=3 finite, non-constant pairs")
    observed = spearmanr(x_arr, y_arr)
    rng, boot = np.random.default_rng(seed), []
    for _ in range(int(iterations)):
        index = rng.integers(0, len(x_arr), len(x_arr))
        if np.unique(x_arr[index]).size < 2 or np.unique(y_arr[index]).size < 2:
            continue
        rho = float(spearmanr(x_arr[index], y_arr[index]).statistic)
        if np.isfinite(rho):
            boot.append(rho)
    if len(boot) < 10:
        raise ValueError("too few valid bootstrap resamples")
    alpha = (1.0 - confidence) / 2.0
    return {
        "rho": float(observed.statistic), "p_value": float(observed.pvalue),
        "ci95_low": float(np.quantile(boot, alpha)),
        "ci95_high": float(np.quantile(boot, 1.0 - alpha)),
        "n_datasets": int(len(x_arr)), "bootstrap_valid": int(len(boot)),
    }
