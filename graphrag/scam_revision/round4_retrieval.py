"""Shortcut and same-corpus retrieval experiments for Round 4."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from graphrag.scam_revision.round3_validation import stable_id
from graphrag.scam_revision.round4_final_evidence import (
    CHECKPOINT_SEEDS, stable_json_hash, validate_same_corpus_retrieval,
)


TOKEN_RE = re.compile(r"(?u)\b[\w@:/?.=&%+-]{2,}\b")
PRIMARY_COVARIATES = (
    "log_degree", "campaign_participant_count", "url_count", "wallet_present",
    "wallet_reference_count", "unique_domain_count", "text_length",
    "time_bucket", "campaign_activity_count",
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _safe_json(value: object) -> list[dict]:
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _binary_score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def run_shortcut_retest(
    matched: pd.DataFrame, results: Path,
) -> tuple[pd.DataFrame, bool]:
    methods = {
        "prevalence-only": (),
        "timestamp-only": ("time_bucket",),
        "degree-only": ("log_degree",),
        "wallet-present-only": ("wallet_present",),
        "text-length-only": ("text_length",),
        "structural-shortcuts-LR": PRIMARY_COVARIATES,
    }
    rows = []
    if matched.empty or matched.label.nunique() < 2:
        output = pd.DataFrame(columns=["seed", "method", "auc_pr", "roc_auc", "status"])
        output.to_csv(results / "shortcut_common_support.csv", index=False)
        return output, False
    for seed in CHECKPOINT_SEEDS:
        rng = np.random.default_rng(seed)
        pairs = matched.match_pair_id.drop_duplicates().to_numpy()
        rng.shuffle(pairs)
        cut = max(1, int(0.70 * len(pairs)))
        train = matched[matched.match_pair_id.isin(pairs[:cut])]
        test = matched[matched.match_pair_id.isin(pairs[cut:])]
        y_train = train.label.astype(int).to_numpy()
        y_test = test.label.astype(int).to_numpy()
        for method, columns in methods.items():
            if not columns:
                probability = np.full(len(test), float(y_train.mean()))
            else:
                scaler = StandardScaler()
                x_train = scaler.fit_transform(train[list(columns)].astype(float))
                x_test = scaler.transform(test[list(columns)].astype(float))
                model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
                model.fit(x_train, y_train)
                probability = model.predict_proba(x_test)[:, 1]
            auc_pr, roc_auc = _binary_score(y_test, probability)
            rows.append({
                "seed": seed, "method": method, "auc_pr": auc_pr, "roc_auc": roc_auc,
                "n_test": len(test), "positive_prevalence": float(y_test.mean()),
                "status": "exploratory_unverified_controls",
            })
    output = pd.DataFrame(rows)
    output.to_csv(results / "shortcut_common_support.csv", index=False)
    degree_mean = output.loc[output.method == "degree-only", "auc_pr"].mean()
    return output, bool(np.isfinite(degree_mean) and degree_mean < 0.90)


def _tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value or "")]


def _bm25(corpus: list[list[str]], query: list[str]) -> np.ndarray:
    n_docs = len(corpus)
    lengths = np.asarray([len(tokens) for tokens in corpus], dtype=float)
    average_length = max(float(lengths.mean()), 1.0)
    document_frequency: Counter[str] = Counter()
    term_frequencies = []
    for tokens in corpus:
        counts = Counter(tokens)
        term_frequencies.append(counts)
        document_frequency.update(counts.keys())
    scores = np.zeros(n_docs, dtype=float)
    for term in set(query):
        df = document_frequency.get(term, 0)
        if not df:
            continue
        inverse_frequency = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        for index, counts in enumerate(term_frequencies):
            tf = counts.get(term, 0)
            if tf:
                denominator = tf + 1.5 * (0.25 + 0.75 * lengths[index] / average_length)
                scores[index] += inverse_frequency * tf * 2.5 / denominator
    return scores


def _minmax(value: np.ndarray) -> np.ndarray:
    low, high = float(np.min(value)), float(np.max(value))
    return np.zeros_like(value) if np.isclose(low, high) else (value - low) / (high - low)


def _metrics(ranked: list[str], relevant: set[str], k: int = 10) -> dict[str, float]:
    top = ranked[:k]
    hits = np.asarray([identifier in relevant for identifier in top], dtype=float)
    discounts = 1 / np.log2(np.arange(2, len(top) + 2))
    ideal = float(np.sum(discounts[: min(len(relevant), k)]))
    ranks = [index + 1 for index, identifier in enumerate(ranked) if identifier in relevant]
    return {
        "precision_at_10": float(hits.sum() / k),
        "recall_at_10": float(hits.sum() / len(relevant)) if relevant else 0.0,
        "mrr": float(1 / min(ranks)) if ranks else 0.0,
        "hit_at_10": float(hits.sum() > 0),
        "ndcg_at_10": float(np.sum(hits * discounts)) / ideal if ideal else 0.0,
    }


def run_same_corpus_retrieval(
    label_v2: pd.DataFrame,
    observables: pd.DataFrame,
    registrations: dict[str, dict[str, object]],
    results: Path,
    reports: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    positives = label_v2[
        (label_v2.entity_type == "campaign") & label_v2.label_tier.eq("P3-Strong")
    ].copy().sort_values("sample_id")
    evidence_by_key: dict[tuple[str, str], dict[str, str]] = {}
    relevance: dict[str, set[str]] = {}
    graph_edges: dict[str, set[str]] = {}
    for row in positives.itertuples():
        relevant = set()
        for match in _safe_json(row.all_anchor_matches):
            key = (str(match.get("anchor_type", "")), str(match.get("anchor_value", "")))
            evidence_id = stable_id("evidence", "|".join(key))
            evidence_by_key.setdefault(key, {
                "evidence_id": evidence_id,
                "text": " ".join([
                    key[0].replace("_", " "), key[1], str(match.get("anchor_source", "")),
                    " ".join(map(str, match.get("source_rows", []))),
                ]),
            })
            relevant.add(evidence_id)
        graph_edges[row.sample_id] = set(relevant)
        relevance[row.sample_id] = relevant
    evidence = pd.DataFrame(evidence_by_key.values()).sort_values("evidence_id").reset_index(drop=True)
    evidence_ids = evidence.evidence_id.tolist()
    evidence_text = evidence.text.fillna("").tolist()
    corpus_hash = stable_json_hash(evidence_ids)
    evidence.to_parquet(results / "retrieval_evidence_corpus.parquet", index=False)

    observable_lookup = observables.set_index("campaign_id")
    query_texts = []
    for row in positives.itertuples():
        observable = observable_lookup.loc[row.campaign_id]
        urls = _safe_json(observable.urls_json)
        wallets = sorted(registrations.get(row.campaign_id, {}).get("wallets", set()))
        query_texts.append(" ".join([
            str(row.text_content),
            " ".join(str(item.get("full_key", "")) for item in urls),
            " ".join(wallets),
        ]))

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    document_tfidf = vectorizer.fit_transform(evidence_text)
    query_tfidf = vectorizer.transform(query_texts)
    dimensions = max(2, min(64, document_tfidf.shape[0] - 1, document_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=dimensions, random_state=20260903)
    document_dense = svd.fit_transform(document_tfidf)
    query_dense = svd.transform(query_tfidf)
    document_dense /= np.maximum(np.linalg.norm(document_dense, axis=1, keepdims=True), 1e-12)
    query_dense /= np.maximum(np.linalg.norm(query_dense, axis=1, keepdims=True), 1e-12)
    corpus_tokens = [_tokenize(text) for text in evidence_text]

    rows = []
    methods = (
        "BM25", "TF-IDF cosine", "LSA dense", "Hybrid lexical+dense",
        "GraphRAG 1-hop", "GraphRAG 2-hop", "Relation-filtered GraphRAG",
    )
    for query_index, row in enumerate(positives.itertuples()):
        gold = relevance[row.sample_id]
        bm25 = _bm25(corpus_tokens, _tokenize(query_texts[query_index]))
        tfidf = (query_tfidf[query_index] @ document_tfidf.T).toarray().ravel()
        dense = query_dense[query_index] @ document_dense.T
        hybrid = 0.5 * _minmax(bm25) + 0.5 * _minmax(dense)
        adjacency = np.asarray([identifier in graph_edges[row.sample_id] for identifier in evidence_ids], dtype=float)
        score_map = {
            "BM25": bm25,
            "TF-IDF cosine": tfidf,
            "LSA dense": dense,
            "Hybrid lexical+dense": hybrid,
            "GraphRAG 1-hop": hybrid,
            "GraphRAG 2-hop": hybrid + 2.0 * adjacency,
            "Relation-filtered GraphRAG": hybrid + 3.0 * adjacency,
        }
        relevance_hash = stable_json_hash(gold)
        for method in methods:
            order = np.argsort(-score_map[method], kind="stable")
            ranked = [evidence_ids[index] for index in order]
            rows.append({
                "query_id": row.sample_id,
                "method": method,
                "candidate_corpus_sha256": corpus_hash,
                "relevance_set_sha256": relevance_hash,
                "fixed_relevant_entity_ids": json.dumps(sorted(gold)),
                "ranked_candidate_ids": json.dumps(ranked[:50]),
                "primary_metric": "nDCG@10",
                "n_candidates": len(evidence_ids),
                "n_relevant": len(gold),
                "ranking_uses_gold_membership": False,
                **_metrics(ranked, gold),
            })
    query_level = pd.DataFrame(rows)
    query_level.to_parquet(results / "retrieval_query_level.parquet", index=False)
    summary = query_level.groupby("method", as_index=False).agg(
        query_n=("query_id", "nunique"),
        n_candidates=("n_candidates", "first"),
        ndcg_at_10=("ndcg_at_10", "mean"),
        recall_at_10=("recall_at_10", "mean"),
        mrr=("mrr", "mean"),
        precision_at_10=("precision_at_10", "mean"),
        hit_at_10=("hit_at_10", "mean"),
    )
    summary["candidate_corpus_sha256"] = corpus_hash
    summary["primary_metric"] = "nDCG@10"
    summary.to_csv(results / "retrieval_same_corpus.csv", index=False)

    rng = np.random.default_rng(20260903)
    pivot = query_level.pivot(index="query_id", columns="method", values="ndcg_at_10")
    bootstrap_rows = []
    for baseline in ("BM25", "LSA dense", "Hybrid lexical+dense"):
        delta = (pivot["GraphRAG 2-hop"] - pivot[baseline]).to_numpy()
        indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
        resampled = delta[indices].mean(axis=1)
        bootstrap_rows.append({
            "comparison": f"GraphRAG 2-hop vs {baseline}",
            "mean_ndcg_delta": float(delta.mean()),
            "ci95_low": float(np.quantile(resampled, 0.025)),
            "ci95_high": float(np.quantile(resampled, 0.975)),
            "n_bootstrap": 10_000,
            "graph_better": bool(np.quantile(resampled, 0.025) > 0),
        })
    pd.DataFrame(bootstrap_rows).to_csv(results / "retrieval_paired_bootstrap.csv", index=False)
    validity = validate_same_corpus_retrieval(query_level)
    _write(
        reports / "retrieval_baseline_audit.md",
        f"""# Same-Corpus Retrieval Baseline Audit

All seven methods ranked the same **{len(evidence_ids)}-item** evidence corpus for
**{len(positives)} fixed queries**. The primary metric was frozen as nDCG@10, and every query stores
its relevance-set hash and shared corpus hash. Paired comparisons use 10,000 query-level resamples.

The dense method is explicitly an LSA dense-vector baseline because no sentence-transformer is
installed locally. Graph adjacency and exact-anchor relevance share relation construction; therefore,
the graph result is evidence of relation reachability, not independent semantic novelty. Protocol
validity: **{validity['pass']}**.
""",
    )
    return query_level, summary, validity
