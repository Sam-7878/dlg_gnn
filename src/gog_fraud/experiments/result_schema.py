from __future__ import annotations

from typing import Any, Mapping


REQUIRED_RESULT_COLUMNS = (
    "experiment_id", "git_sha", "dataset_version", "chain", "fold", "seed",
    "model", "variant", "split", "mc_samples", "dropout_p", "router",
    "tau_b", "tau_f", "tau_u", "legacy_augmentation", "l1_backend",
    "l2_relation", "fusion", "num_samples", "num_pos", "num_neg",
    "roc_auc", "pr_auc", "best_f1", "fraud_recall", "fnr", "ece10",
    "ece20", "brier", "nll", "direct_exit_rate", "benign_direct_rate",
    "fraud_direct_rate", "deep_route_rate", "review_rate", "selective_risk",
    "mean_latency_ms", "p50_latency_ms", "p95_latency_ms", "p99_latency_ms",
    "throughput_samples_s", "throughput_tx_s", "peak_vram_mb", "peak_rss_mb",
    "memory_slope_mb_per_10k", "cache_hit_rate", "queue_wait_p95_ms", "status",
)


def validate_result_record(record: Mapping[str, Any]) -> list[str]:
    errors = [f"missing column: {name}" for name in REQUIRED_RESULT_COLUMNS if name not in record]
    if record.get("split") not in {"train", "validation", "test", "rolling_test"}:
        errors.append("split must identify the evaluated protocol")
    if record.get("status") not in {"success", "failed", "oom", "timeout", "skipped"}:
        errors.append("status is invalid")
    for name in ("roc_auc", "pr_auc", "fraud_recall", "fnr", "direct_exit_rate", "deep_route_rate"):
        value = record.get(name)
        if value is not None and not 0.0 <= float(value) <= 1.0:
            errors.append(f"{name} must be within [0, 1]")
    if record.get("num_samples") is not None and record.get("num_pos") is not None and record.get("num_neg") is not None:
        if int(record["num_pos"]) + int(record["num_neg"]) != int(record["num_samples"]):
            errors.append("num_pos + num_neg must equal num_samples")
    return errors
