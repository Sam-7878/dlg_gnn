"""Round 4 orchestration entry point.

Run with the repository-root Python environment::

    ../.venv/bin/python experiments/scam_revision/run_scam_revision_round4_final.py
"""
from __future__ import annotations

import argparse
import json
import base64
import hashlib
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.scam_revision.run_scam_revision_round4 import (
    LABEL_V2, OBSERVABLES, REPORTS, RESULTS, build_annotation_artifacts,
    build_gog_manifest, common_support_frame,
)
from graphrag.scam_revision.round3_validation import (
    CCC_REGISTRATION_PATH, CSDB_URIS_PATH, CSDB_URLS_PATH, CST_PATH, GOG_ROOT,
    extract_wallets, load_gog_contracts, load_registry_anchors,
)
from graphrag.scam_revision.round4_final_evidence import (
    CHECKPOINT_SEEDS, ROUND4_VERSION, TRANSACTION_COLUMNS, evaluate_gate_v5,
    semantic_alignment_supported,
)
from graphrag.scam_revision.round4_retrieval import (
    run_same_corpus_retrieval, run_shortcut_retest,
)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str))


def registration_aggregates() -> dict[str, dict[str, object]]:
    aggregate: dict[str, dict[str, object]] = defaultdict(
        lambda: {"participants": set(), "wallets": set(), "activity": 0.0, "posts": 0.0}
    )
    columns = ["thread_id", "user_id", "wallet_address", "activity", "posts"]
    for chunk in pd.read_csv(
        CCC_REGISTRATION_PATH, sep="\t", usecols=columns, chunksize=100_000,
        on_bad_lines="skip", low_memory=False,
    ):
        for row in chunk.itertuples(index=False):
            key = f"ccc:{str(row.thread_id).strip()}"
            item = aggregate[key]
            if pd.notna(row.user_id):
                item["participants"].add(str(row.user_id))
            item["wallets"].update(extract_wallets(row.wallet_address))
            activity = pd.to_numeric(row.activity, errors="coerce")
            posts = pd.to_numeric(row.posts, errors="coerce")
            item["activity"] += float(activity) if pd.notna(activity) else 0.0
            item["posts"] += float(posts) if pd.notna(posts) else 0.0
    return aggregate


def build_semantic_blocked_artifacts() -> tuple[pd.DataFrame, dict[str, object]]:
    methods = (
        "TF-IDF + LR", "TF-IDF + SVM", "Sentence embedding + LR",
        "structure-only", "GraphRAG no text", "GraphRAG real text",
        "GraphRAG shuffled text", "GraphRAG unrelated text",
    )
    rows = [{
        "seed": seed,
        "method": method,
        "auc_pr": np.nan,
        "roc_auc": np.nan,
        "control_status": "blocked_no_adjudicated_common_support",
        "n_test": 0,
        "paper_eligible": False,
    } for seed in CHECKPOINT_SEEDS for method in methods]
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "semantic_alignment_ablation.csv", index=False)
    status = semantic_alignment_supported(frame)
    write_text(
        REPORTS / "semantic_alignment_audit.md",
        """# Semantic Alignment Audit

The required real-text, no-text, shuffled-text, unrelated-text, TF-IDF, SVM, and sentence-embedding
rows are materialized as a fail-closed experiment plan, but their metrics are intentionally missing.
No independently adjudicated benign common-support benchmark exists, so running these models on N1
would only quantify behavior against unverified controls. `semantic_alignment_test_complete=false`
and the manuscript must not claim that real semantic evidence beats shuffled or no-text controls.
""",
    )
    return frame, status


def _chain_from_declared_currency(currency: str) -> str:
    normalized = currency.strip().lower()
    if normalized in {"eth", "ethereum"}:
        return "ethereum"
    if normalized in {"btc", "bitcoin"}:
        return "bitcoin"
    return "chain_unknown"


def _address_format(wallet: str) -> str:
    if wallet.startswith("0x") and len(wallet) == 42:
        return "evm_20byte_hex"
    if wallet.startswith("bc1"):
        return "bitcoin_bech32"
    if wallet[:1] in {"1", "3"}:
        return "bitcoin_base58"
    return "unknown"


def build_wallet_inventory(anchors) -> pd.DataFrame:
    currencies: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    source_rows: dict[str, set[str]] = defaultdict(set)

    cst = pd.read_csv(CST_PATH)
    cst.columns = [column.strip() for column in cst.columns]
    for index, row in cst.iterrows():
        declared = str(row.get("detected_crypto_type", "")).strip().lower()
        for wallet in extract_wallets(row.get("crypto_address")):
            if declared and declared != "nan":
                currencies[wallet].add(declared)
            sources[wallet].add("CST")
            source_rows[wallet].add(f"CST:{index}")

    for path, tag in ((CSDB_URLS_PATH, "CSDB_URL"), (CSDB_URIS_PATH, "CSDB_URI")):
        frame = pd.read_csv(path)
        for index, row in frame.iterrows():
            raw = str(row.get("addresses", ""))
            declared = "eth" if "'ETH'" in raw or '"ETH"' in raw else (
                "btc" if "'BTC'" in raw or '"BTC"' in raw else ""
            )
            for wallet in extract_wallets(raw):
                if declared:
                    currencies[wallet].add(declared)
                sources[wallet].add("CSDB")
                source_rows[wallet].add(f"{tag}:{index}")

    rows = []
    for wallet in sorted(anchors.wallets):
        declared_values = sorted(currencies.get(wallet, set()))
        chains = {_chain_from_declared_currency(value) for value in declared_values}
        chains.discard("chain_unknown")
        chain = next(iter(chains)) if len(chains) == 1 else "chain_unknown"
        rows.append({
            "wallet": wallet,
            "declared_currency": ";".join(declared_values),
            "chain": chain,
            "address_format": _address_format(wallet),
            "source_dataset": "+".join(sorted(sources.get(wallet, anchors.wallet_sources[wallet]))),
            "source_row": ";".join(sorted(source_rows.get(wallet, anchors.wallet_rows[wallet]))),
            "chain_resolution": "declared_currency" if chain != "chain_unknown" else "chain_unknown",
        })
    output = pd.DataFrame(rows)
    output["wallet_sha256"] = output.wallet.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    output["wallet_encoding"] = "base64url_utf8"
    output["wallet"] = output.wallet.map(lambda value: base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii"))
    target = RESULTS / "onchain_wallet_manifest.csv"
    temporary = RESULTS / "onchain_wallet_manifest.tmp"
    output.to_csv(temporary, index=False)
    temporary.replace(target)
    return output


def build_unavailable_onchain_artifacts(wallets: pd.DataFrame) -> dict[str, object]:
    transactions = pd.DataFrame({column: pd.Series(dtype="object") for column in TRANSACTION_COLUMNS})
    transactions.to_parquet(RESULTS / "onchain_transactions.parquet", index=False)
    coverage = {
        "dataset": "ScamWallet-OnChain-v1",
        "registry_wallets_total": int(len(wallets)),
        "chain_resolved_wallets": int(wallets.chain.ne("chain_unknown").sum()),
        "wallets_with_real_transactions": 0,
        "wallet_transaction_coverage": 0.0,
        "p3_campaigns_with_real_wallet_transactions": 0,
        "test_samples_with_real_onchain_evidence": 0,
        "transaction_rows": 0,
        "transaction_hash_timestamp_lineage": False,
        "coverage_sufficient": False,
        "source_status": "no local raw archive and no configured authorized explorer/RPC credentials",
        "synthetic_or_proxy_transactions_used": False,
    }
    write_json(RESULTS / "onchain_graph_manifest.json", {
        **coverage,
        "graph_built": False,
        "future_edge_count": 0,
        "reason": "No reproducible transaction-hash/block-timestamp source was available.",
    })
    write_text(
        REPORTS / "onchain_acquisition_report.md",
        f"""# ScamWallet-OnChain-v1 Acquisition Report

Registry wallets: **{len(wallets)}**; chain-resolved from declared currency provenance:
**{coverage['chain_resolved_wallets']}**; wallets with real transactions: **0**.

The local GoG archive contains processed contract graphs, features, and labels but no reproducible
registry-wallet transaction hash/block timestamp lineage. No authorized explorer/API or RPC
credential is configured in the environment. Therefore the transaction parquet is schema-correct
and empty. No chain was inferred merely from an `0x` address, and no synthetic transaction, generated
timestamp, proxy score, or inferred lineage was created. The >=100-wallet and >=50-campaign coverage
criteria fail.
""",
    )
    return coverage


def build_empty_cross_layer_artifacts() -> None:
    checkpoint_columns = [
        "seed", "mode", "checkpoint_path", "checkpoint_sha256", "train_id_hash",
        "val_id_hash", "test_id_hash", "label_hash",
    ]
    raw = RESULTS / "dlg_raw_predictions"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=checkpoint_columns).to_csv(raw / "checkpoint_manifest.csv", index=False)
    pd.DataFrame(columns=[
        "sample_id", "label", "p_rag", "p_gnn", "U_gnn", "fusion_weight", "p_fused",
        "onchain_transaction_hash", "missing_modality_policy", "paper_eligible",
    ]).to_parquet(RESULTS / "cross_layer_complete_manifest.parquet", index=False)
    pd.DataFrame(columns=[
        "method", "seed", "auc_pr", "roc_auc", "macro_f1", "balanced_accuracy",
        "n_complete_cases", "status",
    ]).to_csv(RESULTS / "cross_layer_detection.csv", index=False)
    pd.DataFrame(columns=[
        "source_train", "source_test", "n_positive", "n_adjudicated_benign",
        "auc_pr", "roc_auc", "macro_f1", "balanced_accuracy", "status",
    ]).to_csv(RESULTS / "cross_source_adjudicated.csv", index=False)
    pd.DataFrame(columns=[
        "campaign_id", "social_signal_time", "exact_wallet", "chain",
        "first_wallet_transaction_hash", "first_wallet_transaction_time",
        "first_post_social_transaction_time", "first_suspicious_transaction_time",
    ]).to_parquet(RESULTS / "lead_time_pairs_real_v2.parquet", index=False)
    write_text(
        REPORTS / "real_lead_time_report.md",
        """# Real Lead-Time Reconstruction v2

Eligible real social-to-wallet transaction pairs: **0**. Mean, median, IQR, and bootstrap confidence
intervals are unavailable. Registry report time is kept distinct from transaction time and is not
called fraud settlement. No inferential lead-time claim is permitted.
""",
    )
    write_text(
        REPORTS / "dlg_permutation_audit.md",
        """# DLG-GNN Five-Seed and Permutation Audit

ScamWallet-OnChain-v1 contains no real transactions, so a causal temporal wallet graph cannot be
built and DLG-GNN training is not run. The checkpoint manifest is empty; no observed/permuted path is
shared and no default DLG score is injected. Five-seed checkpoint and permutation gates fail because
evidence is unavailable, not because scores were synthesized.
""",
    )


def static_gog_root_cause() -> pd.DataFrame:
    rows = []
    for chain in ("ethereum", "bsc", "polygon"):
        path = GOG_ROOT / chain / f"{chain}_hybrid_graph.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        embedding = np.asarray(payload["embeddings"], dtype=float)
        label = np.asarray(payload["labels"], dtype=int)
        feature_aucs = []
        for column in range(embedding.shape[1]):
            auc = roc_auc_score(label, embedding[:, column])
            feature_aucs.append(max(auc, 1 - auc))
        rows.append({
            "chain": chain,
            "nodes": len(label),
            "edges": int(payload["edge_index"].shape[1]),
            "embedding_dimensions": int(embedding.shape[1]),
            "positive_prevalence": float(label.mean()),
            "max_orientation_invariant_single_feature_auc": float(max(feature_aucs)),
            "embedding_method": str(payload.get("method", "unrecorded")),
            "embedding_provenance_complete": False,
        })
    output = pd.DataFrame(rows)
    output["wallet_sha256"] = output.wallet.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    output["wallet_encoding"] = "base64url_utf8"
    output["wallet"] = output.wallet.map(lambda value: base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii"))
    output.to_csv(RESULTS / "static_gog_feature_audit.csv", index=False)
    prior = pd.read_csv(ROOT / "results/graphrag/scam_revision_round3/gog_dlg_permutation_audit.csv")
    permuted = prior[prior["mode"] == "permuted_train_labels"]
    write_text(
        REPORTS / "static_gog_permutation_root_cause.md",
        f"""# Static GoG Permutation Root-Cause Audit

Round 3 used fresh model/optimizer objects for every run and did not load checkpoints or cached
predictions, so stale checkpoint reuse is not supported by the implementation. The split is fixed
and labels are permuted over both train and validation development nodes consistently.

The stored eight-dimensional embeddings are already strongly aligned with the true labels: the
largest orientation-invariant single-feature AUC ranges from
**{output.max_orientation_invariant_single_feature_auc.min():.4f}** to
**{output.max_orientation_invariant_single_feature_auc.max():.4f}**. Under permuted development
labels, four seeds produced near-inverted true-test rankings while seed 47 happened to preserve the
true orientation (ROC-AUC **{float(permuted.loc[permuted.seed == 47, 'roc_auc'].iloc[0]):.4f}**).
This bimodality explains the seed-specific result mechanically, but the upstream provenance of the
precomputed embeddings is absent, so label leakage during embedding generation cannot be ruled out.

The graph is also a static transductive challenge with no registry-scam-wallet overlap or transaction
time lineage. Root-cause status is therefore **precomputed representation confounding; upstream
leakage unresolved**. The static GoG challenge remains excluded from the scam paper.
""",
    )
    return output


def write_final_reports(
    gog: dict[str, object], annotation: dict[str, object], matched: pd.DataFrame,
    strict_balance: bool, shortcuts: pd.DataFrame, retrieval: pd.DataFrame,
    semantic: dict[str, object], coverage: dict[str, object], gate: dict[str, object],
) -> None:
    degree = shortcuts.loc[shortcuts.method == "degree-only", "auc_pr"].mean() if not shortcuts.empty else np.nan
    graph = retrieval.loc[retrieval.method == "GraphRAG 2-hop", "ndcg_at_10"].iloc[0]
    bm25 = retrieval.loc[retrieval.method == "BM25", "ndcg_at_10"].iloc[0]
    write_text(
        REPORTS / "claude_feedback_response.md",
        """# Claude Review Response

- The recommendation to prefer a validity-first single main track is accepted in principle, but Gate
  A is not yet paper-ready because independent benign adjudication and the semantic control test are
  missing.
- The requested stronger evidence for the auxiliary scam track is addressed with same-corpus BM25,
  TF-IDF, dense LSA, hybrid, and graph retrieval controls plus query-level bootstrap. Results are
  explicitly limited to relation reachability.
- The alternative full cross-layer route is not claimed: real transaction hashes, timestamps,
  five-seed DLG checkpoints, and complete cross-layer cases are absent.
- The existing Version 3 source already uses the Elsevier `elsarticle` class consistently with
  `elsarticle-num`; the stale IEEE/Elsevier mismatch described in the reviewed snapshot is not
  reintroduced.
- No figure or manuscript claim is generated under a failed gate. Temporal/fraud-specific GNN and
  uncertainty baselines remain requirements for the separately valid timestamp-GNN track before
  journal submission.
""",
    )
    write_text(
        REPORTS / "final_round4_report.md",
        f"""# Scam Revision Round 4 Final Evidence Report

## Outcome

Gate A (Scam GraphRAG paper): **{gate['gate_a_graphrag_paper']}**. Gate B (full GraphRAG +
DLG-GNN): **{gate['gate_b_full_cross_layer']}**. Fail-closed outcome:
`{gate['outcome']}`.

## Evidence acquired and repaired

- GoG exact registry-wallet matches: **{gog['exact_match_count']}**; legacy proxy rows:
  **{gog['legacy_proxy_count']}**; real GoG-evidence campaigns: **0**.
- Independent benign candidates exported: **{annotation['candidate_n']}**; double annotated:
  **{annotation['double_annotated_n']}**; consensus benign: **{annotation['consensus_benign_n']}**.
- Exploratory common support: **{len(matched) // 2} matched pairs**; strict structural balance:
  **{strict_balance}**; degree-only mean AUC-PR: **{degree:.4f}**. These controls are not paper labels.
- Same-corpus retrieval: GraphRAG 2-hop nDCG@10 **{graph:.4f}**, BM25 **{bm25:.4f}**. This is
  relation-reachability evidence because graph adjacency and anchor relevance share construction.
- Semantic test complete: **{semantic['test_complete']}**; semantic-alignment claim supported:
  **{semantic['claim_supported']}**.
- Real transaction rows: **{coverage['transaction_rows']}**; five-seed DLG checkpoints: **0**;
  cross-layer complete cases: **0**.

## Manuscript decision

No LaTeX manuscript-facing result is updated in Round 4 because neither corresponding gate passes.
The scientifically valid next external action is independent double annotation of at least 300
exported controls. A separate authorized transaction acquisition step is required for Gate B.
""",
    )


def run() -> dict[str, object]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    anchors = load_registry_anchors()
    gog_contracts = load_gog_contracts()
    label_v2 = pd.read_parquet(LABEL_V2)
    observables = pd.read_parquet(OBSERVABLES)
    registrations = registration_aggregates()

    _, gog_audit = build_gog_manifest(anchors, gog_contracts)
    _, _, annotation = build_annotation_artifacts(label_v2, observables, registrations)
    _, matched, _, strict_balance = common_support_frame(label_v2, registrations)
    shortcuts, degree_ok = run_shortcut_retest(matched, RESULTS)
    _, retrieval, retrieval_validity = run_same_corpus_retrieval(
        label_v2, observables, registrations, RESULTS, REPORTS,
    )
    _, semantic = build_semantic_blocked_artifacts()
    wallet_inventory = build_wallet_inventory(anchors)
    coverage = build_unavailable_onchain_artifacts(wallet_inventory)
    build_empty_cross_layer_artifacts()
    static_gog_root_cause()

    checks = {
        "independent_benign_adjudication_pass": annotation["pass"],
        "common_support_balance_pass": strict_balance,
        "degree_shortcut_not_near_perfect": degree_ok,
        "fixed_retrieval_gold": retrieval_validity["fixed_gold"],
        "global_retrieval_baselines_complete": retrieval_validity["pass"],
        "semantic_alignment_test_complete": semantic["test_complete"],
        "cross_source_two_class_adjudicated": False,
        "claim_metric_consistency": True,
        "real_scam_wallet_transactions_available": False,
        "real_transaction_hash_timestamp_lineage": False,
        "onchain_coverage_sufficient": coverage["coverage_sufficient"],
        "real_dlg_gnn_checkpoint_5seed": False,
        "dlg_permutation_sanity_pass": False,
        "cross_layer_complete_sample_support": False,
    }
    gate = evaluate_gate_v5(checks)
    gate["details"] = {
        "gog": gog_audit,
        "annotation": annotation,
        "common_support_pairs": int(len(matched) // 2),
        "onchain": coverage,
        "semantic": semantic,
    }
    write_json(RESULTS / "paper_ready_gate_v5.json", gate)
    write_final_reports(
        gog_audit, annotation, matched, strict_balance, shortcuts,
        retrieval, semantic, coverage, gate,
    )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
