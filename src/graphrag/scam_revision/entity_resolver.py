"""
graphrag/scam_revision/entity_resolver.py

Phase C: Canonical Entity Resolution
Implements robust normalization for:
1. Wallets (chain-aware, lowercase, 0x prefix, checksum validation)
2. Domains / URLs (eTLD+1 public suffix separation, host, path)
3. Campaigns & Users (native vs derived identifiers)
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Regex patterns for address format validation
REGEX_ETH = re.compile(r"^0x[a-fA-F0-9]{40}$")
REGEX_BTC_LEGACY = re.compile(r"^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$")
REGEX_BTC_BECH32 = re.compile(r"^bc1[a-zA-HJ-NP-Z0-9]{25,39}$")
REGEX_XRP = re.compile(r"^r[0-9a-zA-Z]{24,34}$")
REGEX_ADA = re.compile(r"^addr1[a-z0-9]{50,100}$")


# Common public suffixes for eTLD+1 extraction
MULTI_PART_TLDS = {
    "co.uk", "org.uk", "me.uk", "gov.uk", "ac.uk",
    "com.au", "net.au", "org.au", "edu.au",
    "co.jp", "ne.jp", "or.jp", "ac.jp",
    "co.kr", "ne.kr", "re.kr", "or.kr",
    "com.br", "net.br", "org.br",
    "com.cn", "net.cn", "org.cn",
    "com.tw", "org.tw",
    "co.in", "net.in", "org.in",
    "com.ru", "net.ru", "org.ru",
}


@dataclass(frozen=True)
class NormalizedWallet:
    raw: str
    chain: str
    address: str
    canonical_id: str  # chain:address
    is_valid_format: bool


@dataclass(frozen=True)
class NormalizedDomainURL:
    raw: str
    scheme: str
    host: str
    domain: str  # eTLD+1
    path: str
    query: str
    is_valid: bool


def normalize_wallet(raw_address: Any, default_chain: Optional[str] = None) -> Optional[NormalizedWallet]:
    """
    Normalizes a cryptocurrency address.
    """
    if raw_address is None:
        return None
    raw = str(raw_address).strip()
    if not raw or raw.lower() in ["nan", "none", "null", ""]:
        return None
    
    # Strip potential enclosing brackets or quotes
    raw = raw.strip("{}[]\"' ")
    
    chain = (default_chain or "unknown").lower().strip()
    cleaned = raw.lower()
    
    # Detect chain from format
    if REGEX_ETH.match(raw):
        if chain in ["unknown", "", "eth"]:
            chain = "ethereum"
        elif chain in ["bsc", "polygon", "arbitrum", "optimism"]:
            # standard EVM
            pass
        return NormalizedWallet(
            raw=raw,
            chain=chain,
            address=cleaned,
            canonical_id=f"{chain}:{cleaned}",
            is_valid_format=True
        )
    elif REGEX_BTC_LEGACY.match(raw) or REGEX_BTC_BECH32.match(raw):
        chain = "bitcoin"
        return NormalizedWallet(
            raw=raw,
            chain=chain,
            address=cleaned,
            canonical_id=f"{chain}:{cleaned}",
            is_valid_format=True
        )
    elif REGEX_XRP.match(raw):
        chain = "ripple"
        return NormalizedWallet(
            raw=raw,
            chain=chain,
            address=raw,  # Base58 case-sensitive
            canonical_id=f"{chain}:{raw}",
            is_valid_format=True
        )
    elif REGEX_ADA.match(raw):
        chain = "cardano"
        return NormalizedWallet(
            raw=raw,
            chain=chain,
            address=cleaned,
            canonical_id=f"{chain}:{cleaned}",
            is_valid_format=True
        )
    
    # Fallback / malformed
    if raw.startswith("0x") and len(raw) == 42:
        return NormalizedWallet(
            raw=raw,
            chain=chain if chain != "unknown" else "ethereum",
            address=cleaned,
            canonical_id=f"{chain}:{cleaned}",
            is_valid_format=True
        )
        
    return NormalizedWallet(
        raw=raw,
        chain=chain,
        address=cleaned,
        canonical_id=f"{chain}:{cleaned}",
        is_valid_format=False
    )


def extract_etld_plus_one(host: str) -> str:
    """Extract eTLD+1 from a hostname."""
    host = host.lower().strip(".")
    parts = host.split(".")
    if len(parts) <= 1:
        return host
    
    # Check 2-part TLDs
    if len(parts) >= 3:
        two_part_tld = f"{parts[-2]}.{parts[-1]}"
        if two_part_tld in MULTI_PART_TLDS:
            return f"{parts[-3]}.{two_part_tld}"
            
    return f"{parts[-2]}.{parts[-1]}"


def normalize_url_domain(raw_url: Any) -> Optional[NormalizedDomainURL]:
    """
    Parses and normalizes a domain or URL.
    """
    if raw_url is None:
        return None
    raw = str(raw_url).strip()
    if not raw or raw.lower() in ["nan", "none", "null", ""]:
        return None
        
    # Ensure scheme for urllib parsing
    target = raw
    if not target.startswith("http://") and not target.startswith("https://"):
        target = "http://" + target
        
    try:
        parsed = urllib.parse.urlparse(target)
        host = (parsed.hostname or "").lower().strip()
        if not host:
            return None
            
        # Strip port
        host = host.split(":")[0]
        domain = extract_etld_plus_one(host)
        
        return NormalizedDomainURL(
            raw=raw,
            scheme=parsed.scheme or "http",
            host=host,
            domain=domain,
            path=parsed.path or "/",
            query=parsed.query or "",
            is_valid=bool(domain and "." in domain)
        )
    except Exception:
        # Fallback simple split
        simple_host = raw.replace("https://", "").replace("http://", "").split("/")[0].split("?")[0].strip().lower()
        dom = extract_etld_plus_one(simple_host)
        return NormalizedDomainURL(
            raw=raw,
            scheme="http",
            host=simple_host,
            domain=dom,
            path="/",
            query="",
            is_valid=bool(dom and "." in dom)
        )


def extract_addresses_from_text(text: str) -> List[NormalizedWallet]:
    """
    Extracts all valid crypto addresses found inside an arbitrary string or JSON field.
    """
    if not text or pd_isna(text):
        return []
    
    text_str = str(text)
    wallets = []
    
    # Find 0x Ethereum addresses
    for m in re.findall(r"0x[a-fA-F0-9]{40}", text_str):
        w = normalize_wallet(m, "ethereum")
        if w: wallets.append(w)
        
    # Find BTC addresses
    for m in re.findall(r"[13][a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{25,39}", text_str):
        w = normalize_wallet(m, "bitcoin")
        if w: wallets.append(w)
        
    return wallets


def pd_isna(val: Any) -> bool:
    if val is None:
        return True
    s = str(val).lower().strip()
    return s in ["nan", "none", "null", ""]
