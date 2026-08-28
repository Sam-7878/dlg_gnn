"""
privacy/vector_codec.py

Serializes risk vectors to bytes and measures actual payload sizes.

Representations (in order of descending privacy protection):
    RAW_CONTEXT     : raw context text (baseline, no privacy)
    FULL_VECTOR     : r_t = [s, q, k, a, h] (full float representation)
    QUANTIZED       : r_t with float32 → int8 quantization
    NOISY           : r_t with Gaussian noise on continuous values
    MINIMAL         : risk_score + category only

Serialization formats:
    JSON            : human-readable (larger payload)
    COMPACT_BINARY  : struct-packed bytes (smallest payload)

The paper claims "96 bytes" for the full risk vector —  this module
measures the actual byte count from real serialization (no hard-coding).
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


# ══════════════════════════════════════════════════════════════════════════════
# Payload breakdown (for paper table generation)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PayloadBreakdown:
    """Per-field byte accounting for the risk vector payload."""
    score_bytes: int = 4       # s_t: float32
    confidence_bytes: int = 4  # q_t: float32
    category_bytes: int = 1    # k_t: uint8
    age_bytes: int = 4         # a_t: float32
    hint_bytes: int = 1        # h_t: uint8
    metadata_bytes: int = 0    # overhead (mode tag, event_id hash, etc.)

    @property
    def total_bytes(self) -> int:
        return (
            self.score_bytes + self.confidence_bytes + self.category_bytes
            + self.age_bytes + self.hint_bytes + self.metadata_bytes
        )

    def to_dict(self) -> Dict[str, int]:
        return {
            "score": self.score_bytes,
            "confidence": self.confidence_bytes,
            "category": self.category_bytes,
            "age": self.age_bytes,
            "relation_hint": self.hint_bytes,
            "metadata": self.metadata_bytes,
            "total": self.total_bytes,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Codec
# ══════════════════════════════════════════════════════════════════════════════

# Compact binary format: 2f2B1f  (score=4, q=4, k=1, h=1, age=4) = 14 bytes base
# + 4 bytes event_id hash + 2 bytes mode tag = 20 bytes minimal
# Full vector JSON is ~96–120 bytes depending on field names

_BINARY_FORMAT = "!2f2Bf"   # network byte order: float, float, uint8, uint8, float
_BINARY_SIZE   = struct.calcsize(_BINARY_FORMAT)   # 14 bytes


class VectorCodec:
    """
    Serializes / deserializes risk vectors and measures real payload bytes.

    Supported modes:
        'json'   : JSON serialization
        'binary' : compact struct-packed bytes
    """

    def __init__(self, mode: str = "json"):
        if mode not in ("json", "binary"):
            raise ValueError(f"mode must be 'json' or 'binary', got {mode!r}")
        self.mode = mode

    # ── Serialization ────────────────────────────────────────────────────────

    def serialize(self, risk_dict: Dict[str, Any]) -> bytes:
        """Serialize a risk vector dict to bytes."""
        if self.mode == "json":
            return self._serialize_json(risk_dict)
        else:
            return self._serialize_binary(risk_dict)

    def _serialize_json(self, d: Dict[str, Any]) -> bytes:
        payload = {
            "event_id": str(d.get("event_id", "")),
            "s": round(float(d.get("local_risk_score", 0.0)), 6),
            "q": round(float(d.get("confidence", 0.0)), 6),
            "k": int(d.get("risk_type_id", 0)),
            "a": int(d.get("context_age_sec", 0)),
            "h": int(d.get("relation_hint_id", 0)),
            "m": str(d.get("privacy_mode", "full")),
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _serialize_binary(self, d: Dict[str, Any]) -> bytes:
        s   = float(d.get("local_risk_score", 0.0))
        q   = float(d.get("confidence", 0.0))
        k   = int(d.get("risk_type_id", 0)) & 0xFF
        h   = int(d.get("relation_hint_id", 0)) & 0xFF
        a   = float(d.get("context_age_sec", 0))
        return struct.pack(_BINARY_FORMAT, s, q, k, h, a)

    # ── Deserialization ───────────────────────────────────────────────────────

    def deserialize(self, data: bytes) -> Dict[str, Any]:
        if self.mode == "json":
            raw = json.loads(data.decode("utf-8"))
            return {
                "event_id": raw.get("event_id"),
                "local_risk_score": raw["s"],
                "confidence": raw["q"],
                "risk_type_id": raw["k"],
                "context_age_sec": raw["a"],
                "relation_hint_id": raw["h"],
                "privacy_mode": raw.get("m", "full"),
            }
        else:
            s, q, k, h, a = struct.unpack(_BINARY_FORMAT, data)
            return {
                "local_risk_score": s,
                "confidence": q,
                "risk_type_id": k,
                "relation_hint_id": h,
                "context_age_sec": a,
            }

    # ── Measurement utilities ─────────────────────────────────────────────────

    def measure_bytes(self, risk_dict: Dict[str, Any]) -> int:
        """Return actual serialized byte count."""
        return len(self.serialize(risk_dict))

    def measure_raw_context_bytes(self, context_text: str, encoding: str = "utf-8") -> int:
        """Return byte length of raw context text."""
        return len(context_text.encode(encoding))

    def payload_breakdown(self) -> PayloadBreakdown:
        """Return the field-level byte breakdown for this codec mode."""
        if self.mode == "binary":
            # struct: 2 float32 (s, q) + 2 uint8 (k, h) + 1 float32 (a)
            return PayloadBreakdown(
                score_bytes=4, confidence_bytes=4,
                category_bytes=1, age_bytes=4, hint_bytes=1,
                metadata_bytes=0,
            )
        else:
            # JSON: approximate field sizes including key names and punctuation
            return PayloadBreakdown(
                score_bytes=12,   # '"s":0.123456' ~12 chars
                confidence_bytes=12,
                category_bytes=6,  # '"k":12' ~6 chars
                age_bytes=12,
                hint_bytes=5,
                metadata_bytes=30, # event_id + mode + brackets + commas
            )

    def measure_batch(self, risk_dicts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Measure payload statistics over a batch of risk dicts."""
        sizes = [self.measure_bytes(d) for d in risk_dicts]
        import statistics
        return {
            "mean_bytes": statistics.mean(sizes),
            "median_bytes": statistics.median(sizes),
            "min_bytes": min(sizes),
            "max_bytes": max(sizes),
            "mode": self.mode,
            "n": len(sizes),
        }
