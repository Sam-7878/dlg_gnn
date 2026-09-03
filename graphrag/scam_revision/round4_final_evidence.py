"""Fail-closed validation primitives for scam revision Round 4.

The functions in this module are intentionally independent from model code.
They encode the evidence rules that must remain true even when an experiment
implementation changes: proxy GoG links are not real scam-wallet matches,
unreviewed controls are not benign labels, and missing on-chain scores are not
imputed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ROUND4_VERSION = "scam-r4-final-evidence-v1.0"
ALLOWED_ANNOTATIONS = {"BENIGN", "SCAM", "AMBIGUOUS", "INSUFFICIENT_EVIDENCE"}
TRANSACTION_COLUMNS = (
    "chain", "transaction_hash", "block_number", "block_timestamp",
    "from_address", "to_address", "value", "token_contract",
    "transaction_type", "source", "retrieved_at",
)
CHECKPOINT_SEEDS = (7, 17, 27, 37, 47)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_hash(values: Iterable[str]) -> str:
    return sha256_text(json.dumps(sorted(set(map(str, values))), separators=(",", ":")))


def parse_source_rows(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item for item in str(value).split(";") if item]


def gog_consistency(manifest: pd.DataFrame) -> dict[str, object]:
    required = {
        "campaign_id", "registry_wallet", "wallet_chain", "gog_entity_id",
        "match_type", "match_key", "exact_match", "legacy_proxy",
        "source_file", "source_row",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"GoG match manifest missing columns: {missing}")
    exact = manifest[manifest.exact_match.fillna(False).astype(bool)]
    proxies = manifest[manifest.legacy_proxy.fillna(False).astype(bool)]
    invalid_exact = exact[
        exact.registry_wallet.fillna("").eq("")
        | exact.gog_entity_id.fillna("").eq("")
        | exact.registry_wallet.str.lower().ne(exact.gog_entity_id.str.lower())
    ]
    real_campaigns = set(exact.campaign_id.dropna()) - {""}
    return {
        "exact_match_count": int(len(exact)),
        "legacy_proxy_count": int(len(proxies)),
        "campaigns_with_real_gog_wallet_evidence": int(len(real_campaigns)),
        "invalid_exact_rows": int(len(invalid_exact)),
        "pass": bool(len(invalid_exact) == 0 and (len(exact) > 0 or len(real_campaigns) == 0)),
    }


def adjudicated_benign_only(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only independent BENIGN consensus rows.

    Empty annotations never become labels.  Both primary annotations and the
    final adjudicated label must agree on BENIGN.
    """
    required = {"annotation_1", "annotation_2", "final_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"annotation frame missing columns: {sorted(missing)}")
    normalized = frame[list(required)].fillna("").apply(lambda column: column.astype(str).str.strip().str.upper())
    mask = (
        normalized.annotation_1.eq("BENIGN")
        & normalized.annotation_2.eq("BENIGN")
        & normalized.final_label.eq("BENIGN")
    )
    return frame.loc[mask].copy()


def annotation_summary(frame: pd.DataFrame) -> dict[str, object]:
    a1 = frame.get("annotation_1", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.upper()
    a2 = frame.get("annotation_2", pd.Series(index=frame.index, dtype=object)).fillna("").astype(str).str.upper()
    both = a1.isin(ALLOWED_ANNOTATIONS) & a2.isin(ALLOWED_ANNOTATIONS)
    if not both.any():
        return {
            "candidate_n": int(len(frame)), "double_annotated_n": 0,
            "agreement": None, "cohen_kappa": None, "disagreement_n": 0,
            "consensus_benign_n": 0, "pass": False,
        }
    left, right = a1[both], a2[both]
    agreement = float((left == right).mean())
    labels = sorted(ALLOWED_ANNOTATIONS)
    observed = agreement
    expected = sum(float((left == label).mean() * (right == label).mean()) for label in labels)
    kappa = None if np.isclose(1 - expected, 0) else float((observed - expected) / (1 - expected))
    benign = adjudicated_benign_only(frame)
    return {
        "candidate_n": int(len(frame)), "double_annotated_n": int(both.sum()),
        "agreement": agreement, "cohen_kappa": kappa,
        "disagreement_n": int((left != right).sum()),
        "consensus_benign_n": int(len(benign)),
        "pass": bool(both.sum() >= 300 and len(benign) > 0),
    }


def standardized_mean_difference(positive: pd.Series, negative: pd.Series) -> float:
    positive = pd.to_numeric(positive, errors="coerce").dropna().astype(float)
    negative = pd.to_numeric(negative, errors="coerce").dropna().astype(float)
    if positive.empty or negative.empty:
        return float("nan")
    pooled = np.sqrt((positive.var(ddof=1) + negative.var(ddof=1)) / 2)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return 0.0 if np.isclose(positive.mean(), negative.mean()) else float("inf")
    return float((positive.mean() - negative.mean()) / pooled)


def balance_diagnostics(frame: pd.DataFrame, covariates: Iterable[str]) -> pd.DataFrame:
    if "label" not in frame:
        raise ValueError("common-support frame requires label")
    positive = frame[frame.label.astype(int) == 1]
    negative = frame[frame.label.astype(int) == 0]
    rows = []
    for column in covariates:
        rows.append({
            "covariate": column,
            "positive_mean": pd.to_numeric(positive[column], errors="coerce").mean(),
            "negative_mean": pd.to_numeric(negative[column], errors="coerce").mean(),
            "smd": standardized_mean_difference(positive[column], negative[column]),
        })
    return pd.DataFrame(rows)


def common_support_balance_pass(diagnostics: pd.DataFrame, threshold: float = 0.1) -> bool:
    if diagnostics.empty or "smd" not in diagnostics:
        return False
    values = pd.to_numeric(diagnostics.smd, errors="coerce")
    return bool(values.notna().all() and np.isfinite(values).all() and (values.abs() < threshold).all())


def validate_same_corpus_retrieval(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "query_id", "method", "candidate_corpus_sha256", "relevance_set_sha256",
        "primary_metric", "n_candidates", "ranking_uses_gold_membership",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"retrieval frame missing columns: {sorted(missing)}")
    methods_per_query = frame.groupby("query_id").method.nunique()
    corpus_per_query = frame.groupby("query_id").candidate_corpus_sha256.nunique()
    relevance_per_query = frame.groupby("query_id").relevance_set_sha256.nunique()
    candidate_counts = frame.groupby("query_id").n_candidates.nunique()
    expected_methods = {
        "BM25", "TF-IDF cosine", "LSA dense", "Hybrid lexical+dense",
        "GraphRAG 1-hop", "GraphRAG 2-hop", "Relation-filtered GraphRAG",
    }
    actual = set(frame.method.unique())
    passed = (
        expected_methods.issubset(actual)
        and methods_per_query.min() >= len(expected_methods)
        and corpus_per_query.max() == 1
        and relevance_per_query.max() == 1
        and candidate_counts.max() == 1
        and frame.primary_metric.eq("nDCG@10").all()
        and not frame.ranking_uses_gold_membership.fillna(True).astype(bool).any()
    )
    return {
        "pass": bool(passed), "query_n": int(frame.query_id.nunique()),
        "methods": sorted(actual), "same_corpus": bool(corpus_per_query.max() == 1 and candidate_counts.max() == 1),
        "fixed_gold": bool(relevance_per_query.max() == 1),
    }


def semantic_alignment_supported(frame: pd.DataFrame) -> dict[str, object]:
    required = {"seed", "method", "auc_pr", "control_status"}
    if required - set(frame.columns):
        return {"test_complete": False, "claim_supported": False, "reason": "missing columns"}
    if frame.empty or not frame.control_status.eq("adjudicated_common_support").all():
        return {
            "test_complete": False, "claim_supported": False,
            "reason": "independently adjudicated common-support controls unavailable",
        }
    pivot = frame.pivot(index="seed", columns="method", values="auc_pr")
    needed = {"GraphRAG real text", "GraphRAG shuffled text", "GraphRAG no text"}
    if not needed.issubset(pivot.columns) or len(pivot) < 5:
        return {"test_complete": False, "claim_supported": False, "reason": "missing ablation seeds"}
    real = pivot["GraphRAG real text"]
    shuffled = pivot["GraphRAG shuffled text"]
    no_text = pivot["GraphRAG no text"]
    return {
        "test_complete": True,
        "claim_supported": bool((real > shuffled).all() and real.mean() > no_text.mean()),
        "mean_real_minus_shuffled": float((real - shuffled).mean()),
        "mean_real_minus_no_text": float((real - no_text).mean()),
    }


def valid_real_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(TRANSACTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"transaction manifest missing columns: {sorted(missing)}")
    transaction_hash = frame.transaction_hash.fillna("").astype(str).str.strip()
    timestamp = pd.to_datetime(frame.block_timestamp, utc=True, errors="coerce")
    chain = frame.chain.fillna("").astype(str).str.strip()
    source = frame.source.fillna("").astype(str).str.strip()
    hash_like = transaction_hash.str.match(r"^(?:0x)?[0-9a-fA-F]{64}$")
    return frame[hash_like & timestamp.notna() & chain.ne("") & source.ne("")].copy()


def validate_checkpoint_isolation(frame: pd.DataFrame) -> dict[str, object]:
    required = {
        "seed", "mode", "checkpoint_path", "checkpoint_sha256", "train_id_hash",
        "val_id_hash", "test_id_hash", "label_hash",
    }
    missing = required - set(frame.columns)
    if missing or frame.empty:
        return {"pass": False, "reason": f"missing checkpoint evidence: {sorted(missing)}"}
    expected = {(seed, mode) for seed in CHECKPOINT_SEEDS for mode in ("observed", "permuted")}
    actual = set(zip(frame.seed.astype(int), frame["mode"].astype(str)))
    unique_paths = frame.checkpoint_path.nunique() == len(frame)
    unique_hashes = frame.checkpoint_sha256.nunique() == len(frame)
    return {
        "pass": bool(actual == expected and unique_paths and unique_hashes),
        "expected_runs": len(expected), "actual_runs": len(actual),
        "unique_paths": bool(unique_paths), "unique_hashes": bool(unique_hashes),
    }


def validate_permutation_chance(frame: pd.DataFrame, prevalence: float, tolerance: float = 0.10) -> bool:
    if frame.empty or not {"seed", "mode", "roc_auc", "auc_pr"}.issubset(frame.columns):
        return False
    permuted = frame[frame["mode"].eq("permuted")]
    if set(permuted.seed.astype(int)) != set(CHECKPOINT_SEEDS):
        return False
    return bool(
        ((permuted.roc_auc.astype(float) - 0.5).abs() <= tolerance).all()
        and ((permuted.auc_pr.astype(float) - prevalence).abs() <= tolerance).all()
    )


def complete_cross_layer_cases(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "label", "p_rag", "p_gnn", "onchain_transaction_hash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cross-layer frame missing columns: {sorted(missing)}")
    return frame[
        frame.p_rag.notna() & frame.p_gnn.notna()
        & frame.onchain_transaction_hash.fillna("").astype(str).str.strip().ne("")
    ].copy()


def evaluate_gate_v5(checks: Mapping[str, object]) -> dict[str, object]:
    gate_a_keys = (
        "independent_benign_adjudication_pass", "common_support_balance_pass",
        "degree_shortcut_not_near_perfect", "fixed_retrieval_gold",
        "global_retrieval_baselines_complete", "semantic_alignment_test_complete",
        "cross_source_two_class_adjudicated", "claim_metric_consistency",
    )
    gate_b_extra = (
        "real_scam_wallet_transactions_available", "real_transaction_hash_timestamp_lineage",
        "onchain_coverage_sufficient", "real_dlg_gnn_checkpoint_5seed",
        "dlg_permutation_sanity_pass", "cross_layer_complete_sample_support",
    )
    normalized = {key: bool(checks.get(key, False)) for key in gate_a_keys + gate_b_extra}
    gate_a = all(normalized[key] for key in gate_a_keys)
    gate_b = gate_a and all(normalized[key] for key in gate_b_extra)
    if gate_b:
        outcome = "A_FULL_CROSS_LAYER"
    elif gate_a:
        outcome = "B_GRAPHRAG_ONLY"
    elif not normalized["independent_benign_adjudication_pass"]:
        outcome = "D_BENIGN_ADJUDICATION_UNAVAILABLE"
    else:
        outcome = "C_EVIDENCE_GATE_FAILED"
    return {
        "version": ROUND4_VERSION, **normalized,
        "gate_a_graphrag_paper": gate_a,
        "gate_b_full_cross_layer": gate_b,
        "paper_ready": gate_a,
        "full_cross_layer_paper_ready": gate_b,
        "outcome": outcome,
        "fail_closed": True,
    }
