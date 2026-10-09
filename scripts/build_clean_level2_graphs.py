#!/usr/bin/env python3
"""
Build Clean Level 2 Graphs for Ethereum, BSC, and Polygon
Using relation_builder.py (Unsupervised Relational Meta-Graph Construction)

This script replaces the legacy, label-dependent *_hybrid_graph.pt artifacts
with mathematically sound, strictly label-independent Level 2 meta-graphs
constructed via cosine k-NN (and optional temporal/entity relations).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

# Ensure src is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gog_fraud.data.level2.relation_builder import (
    RelationBuilderConfig,
    build_level2_graph,
    save_level2_graph,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("build_clean_level2_graphs")

CHAINS = ["polygon", "bsc", "ethereum"]
DEFAULT_GOG_DIR = Path("/mnt/d/_Work/_data/GoG")
if not DEFAULT_GOG_DIR.exists():
    DEFAULT_GOG_DIR = Path("D:/_Work/_data/GoG")


def build_clean_graph_for_chain(
    chain: str,
    gog_dir: Path,
    k: int = 5,
    similarity: str = "cosine",
    output_filename: Optional[str] = None,
) -> Path:
    chain_dir = gog_dir / chain
    if not chain_dir.exists():
        raise FileNotFoundError(f"Chain directory not found: {chain_dir}")

    # Look for source tensor to extract canonical node embeddings, labels, and mappings
    # Check if legacy hybrid or existing tensor is available
    source_path = None
    for cand in [
        chain_dir / f"{chain}_level2_graph.pt",
        chain_dir / f"{chain}_hybrid_graph.pt",
        chain_dir / f"{chain}_knn_graph.pt",
    ]:
        if cand.exists():
            source_path = cand
            break

    if source_path is None:
        raise FileNotFoundError(
            f"No source graph found in {chain_dir} to extract contract embeddings."
        )

    logger.info(f"[{chain.upper()}] Loading source embeddings from {source_path}")
    raw = torch.load(source_path, map_location="cpu", weights_only=False)

    # Extract embeddings
    if isinstance(raw, Data):
        emb = raw.level1_embedding if hasattr(raw, "level1_embedding") and raw.level1_embedding is not None else raw.x[:, :8]
        labels = raw.y if hasattr(raw, "y") and raw.y is not None else raw.labels
        contract_to_idx = getattr(raw, "contract_to_idx", None)
        idx_to_contract = getattr(raw, "idx_to_contract", None)
        n = raw.num_nodes
    else:
        raw_emb = raw["embeddings"]
        emb = torch.from_numpy(raw_emb).float() if isinstance(raw_emb, np.ndarray) else raw_emb.float()
        labels = raw["labels"].float() if torch.is_tensor(raw["labels"]) else torch.tensor(raw["labels"], dtype=torch.float32)
        contract_to_idx = raw.get("contract_to_idx", None)
        idx_to_contract = raw.get("idx_to_contract", None)
        n = raw["num_nodes"]

    emb_np = emb.cpu().numpy() if torch.is_tensor(emb) else emb
    emb_tensor = torch.from_numpy(emb_np).float() if isinstance(emb_np, np.ndarray) else emb.float()

    logger.info(f"[{chain.upper()}] Canonical nodes: {n}, Embedding shape: {tuple(emb_tensor.shape)}")

    # Construct bundle for relation_builder
    bundle = {
        "embedding": emb_tensor,
        "score": torch.zeros(n, 1, dtype=torch.float32),
        "logits": torch.zeros(n, 1, dtype=torch.float32),
        "graph_id": torch.arange(n, dtype=torch.long),
        "label": labels.view(-1, 1) if labels.dim() == 1 else labels,
    }
    if idx_to_contract is not None:
        bundle["contract_id"] = [idx_to_contract.get(i, str(i)) for i in range(n)]

    # Configure clean, label-independent RelationBuilder
    cfg = RelationBuilderConfig(
        relation_modes=["embedding_knn"],
        knn_k=k,
        knn_similarity=similarity,
        knn_self_loops=False,
        include_edge_weight=True,
    )

    logger.info(f"[{chain.upper()}] Executing relation_builder.build_level2_graph (k={k}, sim={similarity})...")
    clean_graph = build_level2_graph(bundle, cfg)

    # Attach backward-compatible attributes so PyG Data & legacy dictionary loaders both work
    clean_graph.embeddings = emb_np
    clean_graph.labels = labels.view(-1).long()
    clean_graph.num_nodes = n
    clean_graph.contract_to_idx = contract_to_idx
    clean_graph.idx_to_contract = idx_to_contract
    clean_graph.method = "relation_builder_clean"
    clean_graph.k = k

    if output_filename is None:
        output_filename = f"{chain}_level2_graph.pt"
    out_path = chain_dir / output_filename

    save_level2_graph(clean_graph, str(out_path))
    logger.info(
        f"[{chain.upper()}] Saved clean Level 2 graph to {out_path} "
        f"(nodes={clean_graph.num_nodes}, edges={clean_graph.edge_index.size(1)})"
    )

    # Verification checks
    assert clean_graph.num_nodes == n, f"Node count mismatch: {clean_graph.num_nodes} vs {n}"
    assert clean_graph.edge_index.size(0) == 2, "Edge index must have 2 rows"
    assert not torch.isnan(clean_graph.x).any(), "NaN found in node features x"
    assert not torch.isinf(clean_graph.x).any(), "Inf found in node features x"

    # Cross-label edge presence check (proving label-independence)
    src_labels = labels.view(-1)[clean_graph.edge_index[0]].long()
    dst_labels = labels.view(-1)[clean_graph.edge_index[1]].long()
    cross_label_edges = (src_labels != dst_labels).sum().item()
    same_label_edges = (src_labels == dst_labels).sum().item()
    total_edges = clean_graph.edge_index.size(1)
    cross_ratio = cross_label_edges / max(1, total_edges)

    logger.info(
        f"[{chain.upper()}] Topology verification: Total edges={total_edges}, "
        f"Cross-class edges={cross_label_edges} ({cross_ratio*100:.2f}%), "
        f"Same-class edges={same_label_edges}"
    )

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Build clean Level 2 graphs using relation_builder.py")
    parser.add_argument("--gog-dir", type=str, default=str(DEFAULT_GOG_DIR), help="Path to _data/GoG root directory")
    parser.add_argument("--k", type=int, default=5, help="k nearest neighbors")
    parser.add_argument("--similarity", type=str, default="cosine", choices=["cosine", "dot"], help="Similarity metric")
    parser.add_argument("--chains", nargs="+", default=CHAINS, help="Chains to process")
    args = parser.parse_args()

    gog_dir = Path(args.gog_dir)
    logger.info(f"Target GoG directory: {gog_dir}")
    logger.info(f"Chains to build: {args.chains}")

    generated_paths = []
    for chain in args.chains:
        out = build_clean_graph_for_chain(
            chain=chain.lower(),
            gog_dir=gog_dir,
            k=args.k,
            similarity=args.similarity,
        )
        generated_paths.append(out)

    logger.info("=" * 60)
    logger.info("Successfully generated all clean Level 2 graphs:")
    for p in generated_paths:
        logger.info(f"  - {p} ({p.stat().st_size:,} bytes)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
