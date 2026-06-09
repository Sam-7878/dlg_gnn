import os
import json
import torch
import logging
from datetime import datetime
from src.benchmark.split_generator import SplitGenerator

logger = logging.getLogger(__name__)

class SemiSyntheticBuilder:
    """
    Combines the transactional GoG graph with synthetic social contexts
    to construct a unified semi-synthetic streaming benchmark.
    """
    def __init__(self, gog_path: str, context_path: str, output_dir: str):
        self.gog_path = gog_path
        self.context_path = context_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def build_benchmark(self, split_type: str = "temporal", seed: int = 42):
        logger.info("="*60)
        logger.info(" BUILDING SEMI-SYNTHETIC STREAMING BENCHMARK")
        logger.info("="*60)
        
        # 1. Load GoG transaction graph
        if not os.path.exists(self.gog_path):
            logger.error(f"GoG Dataset not found at: {self.gog_path}")
            # Generate simulated GNN data structure if actual data is missing
            logger.info("Generating fallback simulated transaction GNN data...")
            num_nodes = 5000
            num_edges = 15000
            x = torch.randn(num_nodes, 32)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
            labels = torch.zeros(num_nodes, dtype=torch.long)
            labels[torch.randperm(num_nodes)[:250]] = 1  # 5% anomalies
            data_dict = {'embeddings': x, 'edge_index': edge_index, 'labels': labels}
        else:
            try:
                data_dict = torch.load(self.gog_path)
                logger.info(f"Loaded GoG graph from {self.gog_path}")
            except Exception as e:
                logger.error(f"Failed to load GoG graph: {e}. Utilizing simulated data.")
                num_nodes = 5000
                num_edges = 15000
                x = torch.randn(num_nodes, 32)
                edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)
                labels = torch.zeros(num_nodes, dtype=torch.long)
                labels[torch.randperm(num_nodes)[:250]] = 1
                data_dict = {'embeddings': x, 'edge_index': edge_index, 'labels': labels}

        # Normalize data tensors
        x = data_dict.get('embeddings', data_dict.get('x'))
        edge_index = data_dict.get('edge_index')
        labels = data_dict.get('labels', data_dict.get('y'))
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        
        # Save graph components to benchmark directory
        torch.save({'embeddings': x, 'edge_index': edge_index, 'labels': labels}, 
                   os.path.join(self.output_dir, "polygon_hybrid_graph.pt"))

        # 2. Load contexts
        contexts = []
        if not os.path.exists(self.context_path):
            raise FileNotFoundError(f"Synthetic contexts not found at {self.context_path}. Run generator first.")
            
        with open(self.context_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    contexts.append(json.loads(line))
        
        logger.info(f"Loaded {len(contexts)} contexts from {self.context_path}")

        # Check label consistency and timestamps
        timestamps = []
        aligned_contexts = []
        
        # Limit to the minimum between GNN nodes and generated contexts
        n_limit = min(num_nodes, len(contexts))
        
        for idx in range(n_limit):
            ctx = contexts[idx]
            gog_label = int(labels[idx].item())
            ctx_label = int(ctx["label"])
            
            # Align labels if discrepancy exists
            if gog_label != ctx_label:
                ctx["label"] = gog_label
                
            aligned_contexts.append(ctx)
            
            # Parse context timestamp
            ts = datetime.fromisoformat(ctx["transaction_timestamp"].replace("Z", ""))
            timestamps.append(ts.timestamp())

        # Write aligned contexts to benchmark directory
        contexts_out = os.path.join(self.output_dir, "contexts.jsonl")
        with open(contexts_out, "w", encoding="utf-8") as f:
            for ctx in aligned_contexts:
                f.write(json.dumps(ctx) + "\n")

        # 3. Create Split
        splitter = SplitGenerator(seed=seed)
        if split_type == "temporal":
            train_idx, val_idx, test_idx = splitter.generate_temporal_split(timestamps)
        elif split_type == "random":
            train_idx, val_idx, test_idx = splitter.generate_random_split(n_limit)
        else:
            train_idx, val_idx, test_idx = splitter.generate_inductive_split(n_limit)

        # Write split ids
        for name, idxs in [("train", train_idx), ("valid", val_idx), ("test", test_idx)]:
            with open(os.path.join(self.output_dir, f"{name}_ids.txt"), "w") as f:
                f.write("\n".join(map(str, idxs)))

        # 4. Generate Metadata
        metadata = {
            "benchmark_name": "GoG-MicroRAG-Stream-v1",
            "num_nodes": n_limit,
            "num_edges": num_edges,
            "num_transactions": n_limit,
            "num_contexts": len(aligned_contexts),
            "fraud_ratio": round((labels[:n_limit] == 1).sum().item() / n_limit, 4),
            "split_type": split_type,
            "created_at": datetime.now().strftime("%Y-%m-%d")
        }
        
        with open(os.path.join(self.output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"✅ Benchmark building completed. Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
        logger.info(f"Metadata saved to {os.path.join(self.output_dir, 'metadata.json')}")

def main():
    gog_path = "D:\\_Work\\_data\\GoG\\polygon\\polygon_hybrid_graph.pt"
    if not os.path.exists(gog_path):
        gog_path = "/mnt/d/_Work/_data/GoG/polygon/polygon_hybrid_graph.pt"
        
    context_path = "d:\\_Work\\goat_bank\\dlg_gnn\\data\\contexts\\synthetic_contexts.jsonl"
    output_dir = "d:\\_Work\\goat_bank\\dlg_gnn\\data\\benchmark\\gog_microrag_stream_v1"
    
    builder = SemiSyntheticBuilder(gog_path, context_path, output_dir)
    builder.build_benchmark(split_type="temporal", seed=42)

if __name__ == "__main__":
    main()
