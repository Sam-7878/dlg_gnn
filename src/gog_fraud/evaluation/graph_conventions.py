"""Dataset graph-direction conventions and stored-edge diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np


SEMANTIC_GRAPH_TYPES = {
    "Elliptic": "directed_transaction_graph",
    "DGraphFin": "directed_financial_interaction_graph",
    "Yelp": "undirected_review_relation_graph",
    "Amazon": "undirected_product_copurchase_graph",
    "BitcoinOTC": "directed_signed_trust_graph",
    "Flickr": "undirected_social_graph",
    "Reddit": "directed_interaction_graph",
    "Cora": "directed_citation_graph",
    "CiteSeer": "directed_citation_graph",
    "PubMed": "directed_citation_graph",
}


def stored_edge_audit(edge_index: Any, num_nodes: int, *, sample_size: int = 100_000) -> dict[str, Any]:
    if hasattr(edge_index, "detach"): edge_index = edge_index.detach().cpu().numpy()
    edges = np.asarray(edge_index, dtype=np.int64)
    source, target = edges
    keys = source * np.int64(num_nodes) + target
    ordered = np.sort(keys)
    if len(keys) > sample_size:
        sample_index = np.linspace(0, len(keys) - 1, sample_size, dtype=np.int64)
        sample_source, sample_target = source[sample_index], target[sample_index]
    else:
        sample_source, sample_target = source, target
    reverse_keys = sample_target * np.int64(num_nodes) + sample_source
    positions = np.searchsorted(ordered, reverse_keys)
    reverse_present = (positions < len(ordered)) & (ordered[np.minimum(positions, len(ordered) - 1)] == reverse_keys) if len(ordered) else np.zeros(len(reverse_keys), dtype=bool)
    reverse_fraction = float(reverse_present.mean()) if len(reverse_present) else float("nan")
    return {
        "stored_edge_type": "edge_index_pairs",
        "contains_reverse_edges": bool(reverse_fraction > 0.95),
        "reverse_edge_fraction_sampled": reverse_fraction,
        "reverse_edge_sample_size": int(len(reverse_present)),
        "self_loop_count": int((source == target).sum()),
    }


def topology_convention(dataset: str, edge_index: Any, num_nodes: int) -> dict[str, Any]:
    semantic = SEMANTIC_GRAPH_TYPES[dataset]
    stored = stored_edge_audit(edge_index, num_nodes)
    semantic_undirected = semantic.startswith("undirected")
    # Existing bidirectional storage must never be mirrored again.
    mode = "undirected_stored_bidirectional_no_mirror" if semantic_undirected and stored["contains_reverse_edges"] else "directed_source_conditioned"
    return {"dataset": dataset, "semantic_graph_type": semantic, **stored, "topology_metric_mode": mode,
            "mirror_for_metric": False}

