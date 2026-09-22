"""Execute Round 4 final-evidence validation for the scam GraphRAG study.

This runner is deliberately fail-closed.  It produces useful manifests when
human adjudication or real chain history is unavailable, but it never promotes
those missing evidence sources to positive gate results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import ks_2samp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graphrag.scam_revision.round3_validation import (
    CCC_REGISTRATION_PATH, CSDB_URIS_PATH, CSDB_URLS_PATH, CST_PATH, GOG_ROOT,
    ROUND3_RESULTS, extract_wallets, load_gog_contracts, load_registry_anchors,
    stable_id,
)
from graphrag.scam_revision.round4_final_evidence import (
    CHECKPOINT_SEEDS, ROUND4_VERSION, TRANSACTION_COLUMNS, adjudicated_benign_only,
    annotation_summary, balance_diagnostics, common_support_balance_pass,
    evaluate_gate_v5, gog_consistency, semantic_alignment_supported,
    sha256_text, stable_json_hash, validate_same_corpus_retrieval,
)


RESULTS = ROOT / "results" / "graphrag" / "scam_revision_round4"
REPORTS = ROOT / "reports" / "graphrag" / "scam_revision_round4"
OBSERVABLES = ROUND3_RESULTS / "campaign_observables.parquet"
LABEL_V2 = ROUND3_RESULTS / "label_manifest_v2.parquet"
TOKEN_RE = re.compile(r"(?u)\b[\w@:/?.=&%+-]{2,}\b")
PRIMARY_COVARIATES = (
    "log_degree", "campaign_participant_count", "url_count", "wallet_present",
    "wallet_reference_count", "unique_domain_count", "text_length",
    "time_bucket", "campaign_activity_count",
)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def json_dump(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True, default=str))


def safe_json(value: object) -> list[dict]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def campaign_registration_aggregates() -> dict[str, dict[str, object]]:
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
            item["activity"] += float(pd.to_numeric(row.activity, errors="coerce") or 0.0)
            item["posts"] += float(pd.to_numeric(row.posts, errors="coerce") or 0.0)
    return aggregate


def build_gog_manifest(anchors, gog_contracts: set[str]) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    source_offset = 0
    for chunk in pd.read_csv(
        CCC_REGISTRATION_PATH, sep="\t", usecols=["thread_id", "wallet_address"],
        chunksize=100_000, on_bad_lines="skip", low_memory=False,
    ):
        for local_index, row in enumerate(chunk.itertuples(index=False)):
            campaign_id = f"ccc:{str(row.thread_id).strip()}"
            for wallet in sorted(extract_wallets(row.wallet_address) & gog_contracts):
                exact = wallet in anchors.wallets
                chain = "ethereum" if exact and wallet.startswith("0x") else "chain_unknown"
                rows.append({
                    "campaign_id": campaign_id,
                    "registry_wallet": wallet if exact else "",
                    "wallet_chain": chain,
                    "gog_entity_id": wallet,
                    "match_type": (
                        "exact_registry_wallet_to_gog_contract" if exact
                        else "ccc_registration_wallet_to_gog_contract"
                    ),
                    "match_key": wallet,
                    "exact_match": exact,
                    "legacy_proxy": not exact,
                    "source_file": str(CCC_REGISTRATION_PATH),
                    "source_row": f"CCC_REGISTRATION:{source_offset + local_index}",
                })
        source_offset += len(chunk)
    columns = [
        "campaign_id", "registry_wallet", "wallet_chain", "gog_entity_id",
        "match_type", "match_key", "exact_match", "legacy_proxy",
        "source_file", "source_row",
    ]
    manifest = pd.DataFrame(rows, columns=columns)
    manifest.to_csv(RESULTS / "gog_match_manifest.csv", index=False)
    audit = gog_consistency(manifest)
    audit.update({
        "registry_scam_wallets": len(anchors.wallets),
        "local_gog_contracts": len(gog_contracts),
        "registry_wallet_gog_intersection": len(anchors.wallets & gog_contracts),
        "legacy_campaign_count": int(manifest.loc[manifest.legacy_proxy, "campaign_id"].nunique()),
        "definition": "real evidence requires an exact registry-wallet == GoG-contract key",
    })
    write_text(
        REPORTS / "gog_match_definition_audit.md",
        f"""# GoG Match-Definition Audit

The row-level reconstruction found **{audit['exact_match_count']} exact registry-wallet matches** and
**{audit['legacy_proxy_count']} legacy proxy rows** across **{audit['legacy_campaign_count']} campaigns**.
The proxy rows mean only that a CCC registration wallet appears in the local GoG contract inventory;
they do not establish that the wallet is a registry-confirmed scam wallet. Consequently,
`campaigns_with_real_gog_wallet_evidence` is **{audit['campaigns_with_real_gog_wallet_evidence']}**.

Acceptance (`exact_match_count == 0 => real-evidence campaigns == 0`): **{audit['pass']}**.
Every retained row includes its source file and source-row identifier. Legacy proxy rows are excluded
from on-chain coverage, lead time, DLG-GNN, and cross-layer evidence.
""",
    )
    return manifest, audit


def build_annotation_artifacts(
    label_v2: pd.DataFrame, observables: pd.DataFrame, registrations: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    label_v3 = label_v2.copy()
    label_v3["label_manifest_version"] = ROUND4_VERSION
    label_v3["independent_adjudication"] = False
    label_v3["control_verification_status"] = np.select(
        [label_v3.label_tier.eq("N1"), label_v3.label_tier.eq("P3-Strong")],
        ["UNVERIFIED_CONTROL", "REGISTRY_ANCHORED_POSITIVE"], default="NOT_APPLICABLE",
    )
    # N1 labels are removed from the paper-eligible manifest until annotation.
    n1 = label_v3.label_tier.eq("N1")
    label_v3.loc[n1, "main_eligible"] = False
    label_v3.loc[n1, "label"] = pd.NA
    label_v3.loc[n1, "negative_verification"] = (
        "UNVERIFIED_CONTROL: exact-anchor-negative is not independently benign"
    )
    label_v3.to_parquet(RESULTS / "label_manifest_v3.parquet", index=False)

    obs = observables.set_index("campaign_id")
    candidates = []
    for row in label_v3[n1].sort_values("sample_id").itertuples():
        observable = obs.loc[row.campaign_id]
        urls = safe_json(observable.urls_json)
        registration = registrations.get(row.campaign_id, {})
        wallets = sorted(registration.get("wallets", set()))
        candidates.append({
            "sample_id": row.sample_id,
            "campaign_id": row.campaign_id,
            "campaign_time": pd.to_datetime(row.timestamp, unit="s", utc=True).isoformat(),
            "campaign_title/text": row.text_content,
            "promoted_urls": json.dumps([item.get("raw", "") for item in urls], ensure_ascii=False),
            "wallets": json.dumps(wallets),
            "domains": json.dumps(sorted({item.get("registered_domain", "") for item in urls if item.get("registered_domain")})),
            "source_links/identifiers": f"{row.source_file}#{row.source_row_id}",
            "CST_exact_hit": False,
            "CSDB_exact_hit": False,
            "annotation_1": "",
            "annotation_2": "",
            "final_label": "",
            "reason": "",
            "verification_status": "UNVERIFIED_CONTROL",
        })
    candidate_frame = pd.DataFrame(candidates)
    candidate_frame.to_csv(RESULTS / "benign_annotation_candidates.csv", index=False)
    summary = annotation_summary(candidate_frame)
    adjudicated = adjudicated_benign_only(candidate_frame)
    adjudicated.to_parquet(RESULTS / "adjudicated_benign_manifest.parquet", index=False)
    write_text(
        REPORTS / "benign_adjudication_report.md",
        f"""# Independent Benign Adjudication Report

Exported candidates: **{summary['candidate_n']}**. Double-annotated: **{summary['double_annotated_n']}**.
Consensus BENIGN: **{summary['consensus_benign_n']}**. Agreement and Cohen's kappa are unavailable
because no independent human annotations were supplied.

Protocol: two annotators independently choose `BENIGN`, `SCAM`, `AMBIGUOUS`, or
`INSUFFICIENT_EVIDENCE`; only BENIGN/BENIGN rows that receive a final BENIGN adjudication may enter
the main negative class. Absence from CST/CSDB, absence of an exact wallet, and absence of a known
malicious URL are not benign labels. Current gate status: **FAIL (human adjudication pending)**.
""",
    )
    return label_v3, candidate_frame, summary


def common_support_frame(
    label_v2: pd.DataFrame, registrations: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    source = label_v2[
        (label_v2.entity_type == "campaign") & label_v2.label_tier.isin(["P3-Strong", "N1"])
    ].copy()
    source["label"] = source.label.astype(int)
    source["campaign_participant_count"] = source.campaign_id.map(
        lambda key: len(registrations.get(key, {}).get("participants", set()))
    )
    source["campaign_activity_count"] = source.campaign_id.map(
        lambda key: registrations.get(key, {}).get("activity", 0.0)
    ).astype(float)
    source["text_length"] = source.text_content.fillna("").astype(str).str.len()
    source["log_degree"] = np.log1p(
        source.url_count.astype(float) + source.wallet_reference_count.astype(float)
        + source.campaign_participant_count.astype(float)
    )
    source["time_bucket"] = pd.to_datetime(source.timestamp, unit="s", utc=True).dt.year * 4 + (
        pd.to_datetime(source.timestamp, unit="s", utc=True).dt.quarter - 1
    )
    source["wallet_present"] = source.wallet_present.astype(int)

    x = source[list(PRIMARY_COVARIATES)].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    propensity_model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=20260903)
    propensity_model.fit(z, source.label)
    propensity = propensity_model.predict_proba(z)[:, 1]
    source["propensity_score"] = propensity
    source["propensity_logit"] = np.log(np.clip(propensity, 1e-6, 1 - 1e-6) / np.clip(1 - propensity, 1e-6, 1))

    positives = source[source.label == 1].copy()
    negatives = source[source.label == 0].copy()
    caliper = 0.20 * float(source.propensity_logit.std(ddof=1))
    available = set(negatives.index)
    pairs: list[tuple[int, int, float]] = []
    # Exact wallet-presence matching precedes nearest-neighbour propensity matching.
    for pos_index in positives.sort_values("propensity_score", ascending=False).index:
        candidates = negatives.loc[list(available)] if available else negatives.iloc[0:0]
        candidates = candidates[candidates.wallet_present == source.loc[pos_index, "wallet_present"]]
        if candidates.empty:
            continue
        distances = (candidates.propensity_logit - source.loc[pos_index, "propensity_logit"]).abs()
        neg_index = int(distances.idxmin())
        distance = float(distances.loc[neg_index])
        if distance <= caliper:
            pairs.append((int(pos_index), neg_index, distance))
            available.remove(neg_index)

    matched_rows = []
    for pair_no, (pos_index, neg_index, distance) in enumerate(pairs):
        for role, index in (("positive", pos_index), ("control", neg_index)):
            row = source.loc[index].to_dict()
            row.update({
                "match_pair_id": f"pair:{pair_no:04d}", "match_role": role,
                "propensity_distance": distance,
                "cem_exact_wallet_present": True,
                "caliper": caliper,
                "control_status": "REGISTRY_ANCHORED_POSITIVE" if role == "positive" else "UNVERIFIED_CONTROL",
                "paper_eligible": False,
            })
            matched_rows.append(row)
    matched = pd.DataFrame(matched_rows)
    matched.to_parquet(RESULTS / "degree_common_support_manifest.parquet", index=False)

    before = balance_diagnostics(source, PRIMARY_COVARIATES)
    before["stage"] = "before"
    after = balance_diagnostics(matched, PRIMARY_COVARIATES) if not matched.empty else pd.DataFrame()
    if not after.empty:
        after["stage"] = "after"
        for column in PRIMARY_COVARIATES:
            pos = matched.loc[matched.label == 1, column]
            neg = matched.loc[matched.label == 0, column]
            after.loc[after.covariate == column, "ks_statistic"] = ks_2samp(pos, neg).statistic
    diagnostics = pd.concat([before, after], ignore_index=True)
    diagnostics.to_csv(RESULTS / "common_support_balance.csv", index=False)
    strict_balance = common_support_balance_pass(after) if not after.empty else False
    write_text(
        REPORTS / "common_support_audit.md",
        f"""# Degree Common-Support Audit

The pipeline applied exact matching on wallet presence, propensity nearest-neighbour matching, and a
0.2-SD logit caliper. It retained **{len(pairs)} matched pairs** from {len(positives)} positives and
{len(negatives)} unverified controls. Maximum post-match |SMD| is
**{after.smd.abs().max() if not after.empty else float('nan'):.4f}**; strict all-covariate |SMD| < 0.1:
**{strict_balance}**.

This is an exploratory construction audit only. The controls remain `UNVERIFIED_CONTROL`, so even a
balanced manifest is not an independently benign benchmark and is not paper-eligible.
""",
    )
    return source, matched, diagnostics, strict_balance
