"""Ground-truth reconstruction utilities for scam revision Round 3.

The module deliberately keeps registry anchors separate from detector inputs.
Shared hosting/social-platform roots can never promote an entire campaign to a
positive label; they require an exact non-root path (or an exact wallet).
"""
from __future__ import annotations

import hashlib
import json
import re
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/mnt/d/_Work/_data/DLG")
CST_PATH = DATA_ROOT / "CryptoScamTracker" / "dan_dataset.csv"
CSDB_URLS_PATH = DATA_ROOT / "CryptoScamDB" / "urls.csv"
CSDB_URIS_PATH = DATA_ROOT / "CryptoScamDB" / "uris.csv"
CCC_ROOT = DATA_ROOT / "CoordinatedCryptocurrencyCampaigns" / "Bounties(Altcoins)" / "labeled"
CCC_EVENTS_PATH = CCC_ROOT / "events.tsv"
CCC_REGISTRATION_PATH = CCC_ROOT / "comments_registration.tsv"
GOG_ROOT = Path("/mnt/d/_Work/_data/GoG")

ROUND3_RESULTS = ROOT / "results" / "graphrag" / "scam_revision_round3"
ROUND3_REPORTS = ROOT / "reports" / "graphrag" / "scam_revision_round3"

# Provider roots are blocked at registered-domain granularity.  Subdomains such
# as docs.google.com are also listed for audit clarity, although the root rule
# already covers them.
SHARED_PLATFORM_DOMAINS = frozenset({
    "google.com", "docs.google.com", "drive.google.com", "googleusercontent.com",
    "t.me", "telegram.me", "medium.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "youtu.be", "github.com", "github.io",
    "githubusercontent.com", "amazonaws.com", "cloudfront.net", "blogspot.com",
    "tumblr.com", "000webhostapp.com", "facebook.com", "fb.me", "reddit.com",
    "linkedin.com", "bitcointalk.org", "imgur.com", "discord.gg", "discord.com",
    "wordpress.com", "wixsite.com", "weebly.com", "notion.site", "linktr.ee",
})
MULTIPART_SUFFIXES = frozenset({
    "co.uk", "org.uk", "com.au", "com.br", "co.jp", "co.kr", "com.sg",
    "com.cn", "com.tr", "co.in", "co.nz", "com.mx", "com.ar", "com.ua",
})

ETH_RE = re.compile(r"(?i)0x[a-f0-9]{40}")
BTC_RE = re.compile(r"(?i)(?:bc1[a-z0-9]{25,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})")


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:20]}"


def registered_domain(host: str) -> str:
    host = host.lower().strip(".")
    parts = host.split(".")
    if len(parts) < 2:
        return host
    suffix = ".".join(parts[-2:])
    if len(parts) >= 3 and suffix in MULTIPART_SUFFIXES:
        return ".".join(parts[-3:])
    return suffix


@dataclass(frozen=True)
class URLAnchor:
    raw: str
    host: str
    registered_domain: str
    path: str
    query: str
    path_key: str
    full_key: str


def normalize_url(raw_value: object) -> URLAnchor | None:
    if raw_value is None or pd.isna(raw_value):
        return None
    raw = str(raw_value).strip().strip("'\"()[]{}<>.,")
    if not raw or raw.lower() in {"nan", "none", "null"} or raw.startswith("mailto:"):
        return None
    target = raw if re.match(r"(?i)^https?://", raw) else "http://" + raw
    try:
        parsed = urlsplit(target)
        host = (parsed.hostname or "").lower().strip(".")
        if "." not in host:
            return None
        path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
        query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
        path_key = f"{host}{path}"
        full_key = f"{path_key}?{query}" if query else path_key
        return URLAnchor(raw, host, registered_domain(host), path, query, path_key, full_key)
    except (TypeError, ValueError):
        return None


def split_url_field(value: object) -> list[URLAnchor]:
    if value is None or pd.isna(value):
        return []
    result: list[URLAnchor] = []
    for raw in re.split(r"\s*,\s*|\s+", str(value)):
        normalized = normalize_url(raw)
        if normalized is not None:
            result.append(normalized)
    return result


def extract_wallets(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()
    text = str(value)
    return {match.lower() for match in ETH_RE.findall(text)} | {
        match.lower() for match in BTC_RE.findall(text)
    }


def parse_time(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else int(parsed.timestamp())


@dataclass
class RegistryAnchors:
    path_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    full_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    host_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    root_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    wallet_sources: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    path_rows: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    wallet_rows: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    path_report_times: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    host_report_times: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    wallet_report_times: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    urls: dict[str, URLAnchor] = field(default_factory=dict)

    @property
    def wallets(self) -> set[str]:
        return set(self.wallet_sources)


def load_registry_anchors() -> RegistryAnchors:
    anchors = RegistryAnchors()
    cst = pd.read_csv(CST_PATH)
    cst.columns = [column.strip() for column in cst.columns]
    for index, row in cst.iterrows():
        source_row = f"CST:{index}"
        url = normalize_url(row.get("website") or row.get("domain"))
        timestamp = parse_time(row.get("time_captured"))
        if url:
            anchors.urls[url.path_key] = url
            anchors.path_sources[url.path_key].add("CST")
            anchors.full_sources[url.full_key].add("CST")
            anchors.host_sources[url.host].add("CST")
            anchors.root_sources[url.registered_domain].add("CST")
            anchors.path_rows[url.path_key].add(source_row)
            if timestamp is not None:
                anchors.path_report_times[url.path_key].append(timestamp)
                anchors.host_report_times[url.host].append(timestamp)
        for wallet in extract_wallets(row.get("crypto_address")):
            anchors.wallet_sources[wallet].add("CST")
            anchors.wallet_rows[wallet].add(source_row)
            if timestamp is not None:
                anchors.wallet_report_times[wallet].append(timestamp)

    for path, tag in ((CSDB_URLS_PATH, "CSDB_URL"), (CSDB_URIS_PATH, "CSDB_URI")):
        frame = pd.read_csv(path)
        for index, row in frame.iterrows():
            source_row = f"{tag}:{index}"
            url = normalize_url(row.get("url") or row.get("name"))
            if url:
                anchors.urls[url.path_key] = url
                anchors.path_sources[url.path_key].add("CSDB")
                anchors.full_sources[url.full_key].add("CSDB")
                anchors.host_sources[url.host].add("CSDB")
                anchors.root_sources[url.registered_domain].add("CSDB")
                anchors.path_rows[url.path_key].add(source_row)
            for wallet in extract_wallets(row.get("addresses")):
                anchors.wallet_sources[wallet].add("CSDB")
                anchors.wallet_rows[wallet].add(source_row)
    return anchors


def load_gog_contracts() -> set[str]:
    contracts: set[str] = set()
    for path in sorted((GOG_ROOT / "features").glob("*_basic_metrics_processed.csv")):
        for chunk in pd.read_csv(path, usecols=["Contract"], chunksize=100_000):
            contracts.update(str(value).lower() for value in chunk["Contract"].dropna())
    return contracts


def load_campaign_wallet_evidence(
    scam_wallets: set[str], gog_contracts: set[str]
) -> tuple[dict[str, int], dict[str, set[str]], dict[str, set[str]]]:
    wallet_reference_count: dict[str, int] = defaultdict(int)
    scam_matches: dict[str, set[str]] = defaultdict(set)
    gog_matches: dict[str, set[str]] = defaultdict(set)
    for chunk in pd.read_csv(
        CCC_REGISTRATION_PATH, sep="\t", usecols=["thread_id", "wallet_address"],
        chunksize=100_000, on_bad_lines="skip",
    ):
        for thread_id, raw_wallet in zip(chunk["thread_id"], chunk["wallet_address"]):
            campaign_id = f"ccc:{str(thread_id).strip()}"
            found = extract_wallets(raw_wallet)
            wallet_reference_count[campaign_id] += len(found)
            overlap = found & scam_wallets
            if overlap:
                scam_matches[campaign_id].update(overlap)
            gog_overlap = found & gog_contracts
            if gog_overlap:
                gog_matches[campaign_id].update(gog_overlap)
    return dict(wallet_reference_count), dict(scam_matches), dict(gog_matches)


def _campaign_match(
    urls: Iterable[URLAnchor], wallets: set[str], anchors: RegistryAnchors,
) -> tuple[list[dict], list[dict]]:
    strong: list[dict] = []
    weak: list[dict] = []
    for wallet in sorted(wallets & anchors.wallets):
        strong.append({
            "anchor_type": "exact_wallet", "anchor_value": wallet,
            "anchor_source": "+".join(sorted(anchors.wallet_sources[wallet])),
            "source_rows": sorted(anchors.wallet_rows[wallet]),
        })
    for url in urls:
        shared = url.registered_domain in SHARED_PLATFORM_DOMAINS or url.host in SHARED_PLATFORM_DOMAINS
        sources = anchors.path_sources.get(url.path_key, set()) | anchors.full_sources.get(url.full_key, set())
        if sources and (not shared or url.path != "/"):
            strong.append({
                "anchor_type": "exact_shared_path" if shared else "exact_full_url",
                "anchor_value": url.full_key, "anchor_source": "+".join(sorted(sources)),
                "source_rows": sorted(anchors.path_rows.get(url.path_key, set())),
            })
            continue
        host_sources = anchors.host_sources.get(url.host, set())
        if host_sources and not shared:
            strong.append({
                "anchor_type": "dedicated_malicious_host", "anchor_value": url.host,
                "anchor_source": "+".join(sorted(host_sources)),
                "source_rows": [],
            })
            continue
        root_has_anchor = bool(anchors.root_sources.get(url.registered_domain))
        if shared and root_has_anchor:
            weak.append({
                "anchor_type": "shared_platform_root_only", "anchor_value": url.registered_domain,
                "anchor_source": "registry_root_overlap", "source_rows": [],
            })
    # JSON-stable de-duplication.
    def unique(rows: list[dict]) -> list[dict]:
        seen: set[tuple[str, str]] = set(); result = []
        for row in rows:
            key = (row["anchor_type"], row["anchor_value"])
            if key not in seen:
                seen.add(key); result.append(row)
        return result
    return unique(strong), unique(weak)


def _time_matched_controls(campaigns: list[dict], target: int = 1500) -> set[str]:
    """Match exact-anchor-negative controls on observable nuisance features.

    Matching does not make these independently adjudicated benign cases; that
    remains a fail-closed gate item.
    """
    positives = [row for row in campaigns if row["strong_matches"]]
    candidates = [
        row for row in campaigns
        if not row["strong_matches"] and not row["weak_matches"] and row["url_count"] > 0
    ]
    if not positives or not candidates:
        return set()

    def vector(row: dict) -> list[float]:
        return [
            row["timestamp"] / 86_400.0,
            np.log1p(row["url_count"]),
            np.log1p(row["unique_host_count"]),
            np.log1p(row["unique_domain_count"]),
            np.log1p(row["shared_platform_count"]),
            float(row["wallet_reference_count"] > 0),
            np.log1p(len(row["text_content"])),
        ]

    positive_x = np.asarray([vector(row) for row in positives], dtype=float)
    candidate_x = np.asarray([vector(row) for row in candidates], dtype=float)
    combined = np.vstack([positive_x, candidate_x])
    median = np.median(combined, axis=0)
    scale = np.quantile(combined, 0.75, axis=0) - np.quantile(combined, 0.25, axis=0)
    fallback = np.std(combined, axis=0)
    scale = np.where(scale < 1e-9, fallback, scale)
    scale = np.where(scale < 1e-9, 1.0, scale)
    positive_x = (positive_x - median) / scale
    candidate_x = (candidate_x - median) / scale
    weights = np.asarray([2.0, 1.5, 1.5, 1.5, 1.0, 3.0, 1.0])

    selected_indices: set[int] = set()
    nearest_per_positive = max(1, target // len(positives))
    for positive in positive_x:
        distances = np.abs(candidate_x - positive).dot(weights)
        added = 0
        for index in np.argsort(distances, kind="stable"):
            candidate_index = int(index)
            if candidate_index not in selected_indices:
                selected_indices.add(candidate_index)
                added += 1
                if added >= nearest_per_positive:
                    break
        if len(selected_indices) >= target:
            break

    if len(selected_indices) < min(target, len(candidates)):
        minimum_distance = np.full(len(candidates), np.inf)
        for positive in positive_x:
            minimum_distance = np.minimum(minimum_distance, np.abs(candidate_x - positive).dot(weights))
        for index in np.argsort(minimum_distance, kind="stable"):
            selected_indices.add(int(index))
            if len(selected_indices) >= min(target, len(candidates)):
                break
    return {candidates[index]["campaign_id"] for index in selected_indices}


def build_round3_manifests() -> dict[str, object]:
    ROUND3_RESULTS.mkdir(parents=True, exist_ok=True)
    ROUND3_REPORTS.mkdir(parents=True, exist_ok=True)
    (ROUND3_RESULTS / "evaluation_sample_manifests").mkdir(parents=True, exist_ok=True)
    anchors = load_registry_anchors()
    gog_contracts = load_gog_contracts()
    wallet_counts, campaign_scam_wallets, campaign_gog_wallets = load_campaign_wallet_evidence(
        anchors.wallets, gog_contracts
    )

    columns = [
        "thread_id", "post_time ", "title", "clean_title", "categories", "reward_pool",
        "social_media_urls", "other_urls", "spreadsheet_urls", "forum_urls", "image_urls",
    ]
    events = pd.read_csv(CCC_EVENTS_PATH, sep="\t", usecols=columns, on_bad_lines="skip")
    campaigns: list[dict] = []
    anchor_rows: list[dict] = []
    promoted_domain_counts: dict[str, int] = defaultdict(int)
    for source_index, event in events.iterrows():
        campaign_id = f"ccc:{str(event['thread_id']).strip()}"
        timestamp = parse_time(event.get("post_time "))
        if timestamp is None:
            continue
        urls: list[URLAnchor] = []
        for column in ("social_media_urls", "other_urls", "spreadsheet_urls", "forum_urls", "image_urls"):
            urls.extend(split_url_field(event.get(column)))
        urls = list({url.full_key: url for url in urls}.values())
        for url in urls:
            promoted_domain_counts[url.registered_domain] += 1
        matched_wallets = campaign_scam_wallets.get(campaign_id, set())
        strong, weak = _campaign_match(urls, matched_wallets, anchors)
        text = " ".join(str(event.get(column, "")) for column in ("clean_title", "title", "categories", "reward_pool"))
        text = re.sub(r"\s+", " ", text.replace("nan", " ")).strip()
        row = {
            "campaign_id": campaign_id, "source_row_id": f"CCC_EVENTS:{source_index}",
            "timestamp": timestamp, "timestamp_source_column": "post_time",
            "text_content": text, "urls": urls,
            "url_count": len(urls), "unique_host_count": len({url.host for url in urls}),
            "unique_domain_count": len({url.registered_domain for url in urls}),
            "shared_platform_count": sum(url.registered_domain in SHARED_PLATFORM_DOMAINS for url in urls),
            "wallet_reference_count": wallet_counts.get(campaign_id, 0),
            "strong_matches": strong, "weak_matches": weak,
            "gog_wallets": sorted(campaign_gog_wallets.get(campaign_id, set())),
        }
        campaigns.append(row)
        for match in strong + weak:
            anchor_rows.append({
                "campaign_id": campaign_id, "source_row_id": row["source_row_id"],
                "social_signal_time": timestamp, **match,
                "is_strong": match in strong,
                "shared_platform_blocked": match["anchor_type"] == "shared_platform_root_only",
            })

    controls = _time_matched_controls(campaigns, target=1500)
    manifest_rows: list[dict] = []
    for row in campaigns:
        if row["strong_matches"]:
            tier, label, eligible = "P3-Strong", 1, True
            matches = row["strong_matches"]
        elif row["weak_matches"]:
            tier, label, eligible = "P3-Weak", pd.NA, False
            matches = row["weak_matches"]
        elif row["campaign_id"] in controls:
            tier, label, eligible = "N1", 0, True
            matches = []
        else:
            tier, label, eligible = "N2", pd.NA, False
            matches = []
        manifest_rows.append({
            "sample_id": row["campaign_id"], "entity_type": "campaign",
            "campaign_id": row["campaign_id"], "source_dataset": "CCC",
            "source_file": str(CCC_EVENTS_PATH), "source_row_id": row["source_row_id"],
            "label": label, "label_tier": tier, "main_eligible": eligible,
            "anchor_type": matches[0]["anchor_type"] if matches else ("anchor_negative_control" if tier == "N1" else ""),
            "anchor_value": matches[0]["anchor_value"] if matches else "",
            "anchor_source": matches[0]["anchor_source"] if matches else ("CCC exact-anchor-negative" if tier == "N1" else ""),
            "timestamp": row["timestamp"], "timestamp_source": "CCC post_time",
            "real_timestamp": True, "text_content": row["text_content"],
            "url_count": row["url_count"], "unique_host_count": row["unique_host_count"],
            "unique_domain_count": row["unique_domain_count"],
            "shared_platform_count": row["shared_platform_count"],
            "wallet_reference_count": row["wallet_reference_count"],
            "wallet_present": row["wallet_reference_count"] > 0,
            "domain_present": row["unique_domain_count"] > 0,
            "gog_wallet_match_count": len(row["gog_wallets"]),
            "all_anchor_matches": json.dumps(matches, sort_keys=True),
            "negative_verification": (
                "no exact registry URL/host/wallet anchor; time/feature-matched CCC campaign control; not manually adjudicated"
                if tier == "N1" else ""
            ),
            "split_name": "excluded", "label_manifest_version": "scam-r3-label-v2.0",
        })

    # Registry anchors are retained as ground-truth provenance, never as main detector inputs.
    for path_key, sources in anchors.path_sources.items():
        url = anchors.urls[path_key]
        manifest_rows.append({
            "sample_id": stable_id("url", path_key), "entity_type": "url", "campaign_id": "",
            "source_dataset": "+".join(sorted(sources)), "source_file": "CST/CSDB registry",
            "source_row_id": ";".join(sorted(anchors.path_rows[path_key])), "label": 1,
            "label_tier": "P1" if len(sources) > 1 else "P2", "main_eligible": False,
            "anchor_type": "exact_full_url", "anchor_value": path_key,
            "anchor_source": "+".join(sorted(sources)),
            "timestamp": min(anchors.path_report_times.get(path_key, [0])) or pd.NA,
            "timestamp_source": "CST time_captured" if anchors.path_report_times.get(path_key) else "unavailable",
            "real_timestamp": bool(anchors.path_report_times.get(path_key)), "text_content": url.host + " " + url.path,
            "url_count": 1, "unique_host_count": 1, "unique_domain_count": 1,
            "shared_platform_count": int(url.registered_domain in SHARED_PLATFORM_DOMAINS),
            "wallet_reference_count": 0, "wallet_present": False, "domain_present": True,
            "gog_wallet_match_count": 0, "all_anchor_matches": "[]", "negative_verification": "",
            "split_name": "registry_anchor_only", "label_manifest_version": "scam-r3-label-v2.0",
        })
    for wallet, sources in anchors.wallet_sources.items():
        manifest_rows.append({
            "sample_id": f"wallet:{wallet}", "entity_type": "wallet", "campaign_id": "",
            "source_dataset": "+".join(sorted(sources)), "source_file": "CST/CSDB registry",
            "source_row_id": ";".join(sorted(anchors.wallet_rows[wallet])), "label": 1,
            "label_tier": "P1" if len(sources) > 1 else "P2", "main_eligible": False,
            "anchor_type": "exact_wallet", "anchor_value": wallet,
            "anchor_source": "+".join(sorted(sources)),
            "timestamp": min(anchors.wallet_report_times.get(wallet, [0])) or pd.NA,
            "timestamp_source": "CST time_captured" if anchors.wallet_report_times.get(wallet) else "unavailable",
            "real_timestamp": bool(anchors.wallet_report_times.get(wallet)), "text_content": "wallet address",
            "url_count": 0, "unique_host_count": 0, "unique_domain_count": 0,
            "shared_platform_count": 0, "wallet_reference_count": 1, "wallet_present": True,
            "domain_present": False, "gog_wallet_match_count": int(wallet in gog_contracts),
            "all_anchor_matches": "[]", "negative_verification": "", "split_name": "registry_anchor_only",
            "label_manifest_version": "scam-r3-label-v2.0",
        })

    manifest = pd.DataFrame(manifest_rows)
    campaign_main = manifest[(manifest.entity_type == "campaign") & manifest.main_eligible].copy()
    campaign_main = campaign_main.sort_values(["timestamp", "sample_id"])
    n = len(campaign_main); train_end = int(0.70 * n); validation_end = train_end + int(0.15 * n)
    split_by_id = {
        sample_id: ("train" if index < train_end else "validation" if index < validation_end else "test")
        for index, sample_id in enumerate(campaign_main.sample_id)
    }
    manifest.loc[manifest.sample_id.isin(split_by_id), "split_name"] = manifest.loc[
        manifest.sample_id.isin(split_by_id), "sample_id"
    ].map(split_by_id)
    natural = manifest[manifest.sample_id.isin(split_by_id)].copy()
    for split in ("train", "validation", "test"):
        labels = natural.loc[natural.split_name == split, "label"].astype(int)
        if labels.nunique() != 2:
            raise RuntimeError(f"natural temporal split lacks two-class support: {split}")

    manifest.to_parquet(ROUND3_RESULTS / "label_manifest_v2.parquet", index=False)
    campaign_observables = pd.DataFrame([{
        **{key: value for key, value in row.items() if key not in {"urls", "strong_matches", "weak_matches"}},
        "urls_json": json.dumps([url.__dict__ for url in row["urls"]], sort_keys=True),
        "strong_matches_json": json.dumps(row["strong_matches"], sort_keys=True),
        "weak_matches_json": json.dumps(row["weak_matches"], sort_keys=True),
    } for row in campaigns])
    campaign_observables.to_parquet(ROUND3_RESULTS / "campaign_observables.parquet", index=False)
    pd.DataFrame(anchor_rows).to_parquet(ROUND3_RESULTS / "p3_anchor_manifest.parquet", index=False)
    natural.to_parquet(ROUND3_RESULTS / "evaluation_sample_manifests" / "natural_temporal.parquet", index=False)

    # A balanced high-confidence track is explicitly distinct from natural prevalence.
    positives = natural[natural.label.astype(int) == 1]
    negatives = natural[natural.label.astype(int) == 0].sort_values("sample_id").head(len(positives))
    balanced = pd.concat([positives, negatives]).sort_values(["timestamp", "sample_id"])
    balanced.to_parquet(ROUND3_RESULTS / "evaluation_sample_manifests" / "balanced_high_confidence.parquet", index=False)
    test_manifest = natural[natural.split_name == "test"].copy()
    train_manifest = natural[natural.split_name == "train"].copy()
    campaign_disjoint = test_manifest.copy()
    campaign_disjoint["disjointness_verified"] = not bool(
        set(campaign_disjoint.sample_id) & set(train_manifest.sample_id)
    )
    campaign_disjoint["two_class_support"] = campaign_disjoint.label.astype(int).nunique() == 2
    campaign_disjoint["paper_eligible"] = (
        campaign_disjoint.disjointness_verified & campaign_disjoint.two_class_support
    )
    campaign_disjoint["track_interpretation"] = "unseen campaign IDs under the frozen temporal split"
    campaign_disjoint.to_parquet(
        ROUND3_RESULTS / "evaluation_sample_manifests" / "campaign_disjoint.parquet", index=False
    )

    # Full wallet identities are not retained for all CCC controls.  The only
    # verifiable common-support diagnostic is the subset with no wallet
    # reference at all; it must not be described as unseen-wallet transfer.
    wallet_disjoint = test_manifest[test_manifest.wallet_reference_count.astype(int) == 0].copy()
    wallet_disjoint["disjointness_verified"] = True
    wallet_disjoint["two_class_support"] = wallet_disjoint.label.astype(int).nunique() == 2
    wallet_disjoint["paper_eligible"] = False
    wallet_disjoint["track_interpretation"] = (
        "wallet-absent two-class diagnostic; not unseen-wallet generalization"
    )
    wallet_disjoint.to_parquet(
        ROUND3_RESULTS / "evaluation_sample_manifests" / "wallet_disjoint.parquet", index=False
    )

    observable_lookup = campaign_observables.set_index("campaign_id")
    def observed_domains(sample_id: str) -> set[str]:
        return {
            row["registered_domain"] for row in json.loads(observable_lookup.loc[sample_id, "urls_json"])
            if row.get("registered_domain")
        }
    train_domains = set().union(*(observed_domains(sample_id) for sample_id in train_manifest.sample_id))
    domain_mask = test_manifest.sample_id.map(
        lambda sample_id: not bool(observed_domains(sample_id) & train_domains)
    )
    domain_disjoint = test_manifest[domain_mask].copy()
    domain_disjoint["disjointness_verified"] = True
    domain_disjoint["two_class_support"] = domain_disjoint.label.astype(int).nunique() == 2
    domain_disjoint["paper_eligible"] = False
    domain_disjoint["track_interpretation"] = (
        "strict observed-domain-disjoint subset; one-class support, metrics unavailable"
    )
    domain_disjoint.to_parquet(
        ROUND3_RESULTS / "evaluation_sample_manifests" / "domain_disjoint.parquet", index=False
    )
    pd.DataFrame([
        {"track": "campaign_disjoint", "n": len(campaign_disjoint),
         "n_positive": int(campaign_disjoint.label.astype(int).sum()),
         "two_class_support": bool(campaign_disjoint.label.astype(int).nunique() == 2),
         "paper_eligible": True, "interpretation": campaign_disjoint.track_interpretation.iloc[0]},
        {"track": "wallet_disjoint", "n": len(wallet_disjoint),
         "n_positive": int(wallet_disjoint.label.astype(int).sum()),
         "two_class_support": bool(wallet_disjoint.label.astype(int).nunique() == 2),
         "paper_eligible": False, "interpretation": wallet_disjoint.track_interpretation.iloc[0]},
        {"track": "domain_disjoint", "n": len(domain_disjoint),
         "n_positive": int(domain_disjoint.label.astype(int).sum()),
         "two_class_support": bool(domain_disjoint.label.astype(int).nunique() == 2),
         "paper_eligible": False, "interpretation": domain_disjoint.track_interpretation.iloc[0]},
    ]).to_csv(ROUND3_RESULTS / "entity_disjoint_support.csv", index=False)

    audit = pd.DataFrame([
        {
            "registered_domain": domain, "ccc_reference_count": count,
            "is_shared_platform": domain in SHARED_PLATFORM_DOMAINS,
            "registry_host_count": sum(registered_domain(host) == domain for host in anchors.host_sources),
            "eligible_as_root_anchor": False if domain in SHARED_PLATFORM_DOMAINS else True,
        }
        for domain, count in sorted(promoted_domain_counts.items(), key=lambda item: (-item[1], item[0]))
    ])
    audit.to_csv(ROUND3_RESULTS / "shared_platform_domain_audit.csv", index=False)

    # Lead-time lineage: no on-chain timestamp is invented.  Report-time is CST-only.
    lead_rows = []
    campaign_lookup = {row["campaign_id"]: row for row in campaigns}
    for record in manifest[(manifest.entity_type == "campaign") & (manifest.label_tier == "P3-Strong")].itertuples():
        matches = json.loads(record.all_anchor_matches)
        report_times: list[int] = []
        matched_wallets: set[str] = set()
        for match in matches:
            if match["anchor_type"] == "exact_wallet":
                matched_wallets.add(match["anchor_value"])
                report_times.extend(anchors.wallet_report_times.get(match["anchor_value"], []))
            else:
                normalized = normalize_url(match["anchor_value"])
                key = normalized.path_key if normalized else match["anchor_value"].split("?")[0]
                report_times.extend(anchors.path_report_times.get(key, []))
                host = key.split("/")[0]
                report_times.extend(anchors.host_report_times.get(host, []))
        report_time = min(report_times) if report_times else None
        gog_wallet_match = bool(matched_wallets & gog_contracts)
        lead_rows.append({
            "campaign_id": record.campaign_id, "source_file": str(CCC_EVENTS_PATH),
            "source_row_id": record.source_row_id, "social_signal_time": int(record.timestamp),
            "timestamp_source_column": "post_time", "scam_report_time": report_time,
            "report_timestamp_source": "CST time_captured" if report_time else "unavailable",
            "wallet": sorted(matched_wallets)[0] if matched_wallets else "",
            "chain": "ethereum" if any(wallet.startswith("0x") for wallet in matched_wallets) else "unknown",
            "transaction_hash_or_event_id": "", "first_observed_transaction_time": pd.NA,
            "first_suspicious_transaction_time": pd.NA,
            "real_social_time": True, "real_report_time": report_time is not None,
            "real_onchain_time": False, "wallet_exact_match": bool(matched_wallets),
            "gog_contract_match": gog_wallet_match,
            "social_to_report_days": ((report_time - int(record.timestamp)) / 86400) if report_time else pd.NA,
            "social_to_onchain_days": pd.NA, "paper_metric_eligible": False,
            "ineligibility_reason": "GoG processed archive has no transaction hash/timestamp lineage",
        })
    lead = pd.DataFrame(lead_rows)
    lead.to_parquet(ROUND3_RESULTS / "lead_time_pairs_real.parquet", index=False)
    report_eligible = lead[lead.real_report_time]
    summary = pd.DataFrame([{
        "metric": "social_to_registry_report", "eligible_n": len(report_eligible),
        "mean_days": report_eligible.social_to_report_days.mean() if len(report_eligible) else np.nan,
        "median_days": report_eligible.social_to_report_days.median() if len(report_eligible) else np.nan,
        "paper_eligible": len(report_eligible) >= 30,
        "status": "insufficient_real_pairs" if len(report_eligible) < 30 else "available",
    }, {
        "metric": "social_to_onchain", "eligible_n": 0, "mean_days": np.nan, "median_days": np.nan,
        "paper_eligible": False, "status": "unavailable_no_transaction_timestamp_lineage",
    }])
    summary.to_csv(ROUND3_RESULTS / "lead_time_real_summary.csv", index=False)

    tier_counts = manifest.label_tier.value_counts().to_dict()
    dataset_manifest = {
        "version": "scam-r3-label-v2.0", "p3_strong": int(tier_counts.get("P3-Strong", 0)),
        "p3_weak_excluded": int(tier_counts.get("P3-Weak", 0)),
        "n1_hard_controls": int(tier_counts.get("N1", 0)), "n2_excluded": int(tier_counts.get("N2", 0)),
        "p1_registry_anchors": int(tier_counts.get("P1", 0)), "p2_registry_anchors": int(tier_counts.get("P2", 0)),
        "natural_samples": len(natural), "natural_positive_prevalence": float(natural.label.astype(int).mean()),
        "shared_platform_root_anchors_allowed": False,
        "negative_control_limitation": "N1 controls are exact-anchor-negative, time/feature-matched CCC campaigns, not independently adjudicated benign entities",
        "gog_contracts": len(gog_contracts), "registry_scam_wallets": len(anchors.wallets),
        "registry_wallet_gog_matches": len(anchors.wallets & gog_contracts),
        "lead_time_policy": "no synthetic timestamps; on-chain lead time unavailable without transaction lineage",
    }
    (ROUND3_RESULTS / "dataset_manifest_v2.json").write_text(json.dumps(dataset_manifest, indent=2) + "\n")
    return {
        "manifest": manifest, "natural": natural, "balanced": balanced,
        "anchors": anchors, "campaigns": campaigns, "gog_contracts": gog_contracts,
        "campaign_scam_wallets": campaign_scam_wallets, "campaign_gog_wallets": campaign_gog_wallets,
        "dataset_manifest": dataset_manifest,
    }
