import os
import torch
import logging
from torch_geometric.datasets import Yelp, Amazon, Reddit
from pygod.generator import gen_contextual_outlier, gen_structural_outlier

logger = logging.getLogger(__name__)

class StreamingDataLoader:
    """
    Streaming data loader for Yelp, Amazon, and Reddit datasets.
    Yields node chunks with synthesized review texts based on node labels.
    Uses generator to prevent memory overflow (OOM) on large graphs.
    """
    def __init__(self, dataset_name: str, root_dir: str, chunk_size: int = 500, max_nodes: int = None):
        self.dataset_name = dataset_name.lower()
        self.root_dir = root_dir
        self.chunk_size = chunk_size
        self.max_nodes = max_nodes
        
        logger.info(f"Loading dataset: {dataset_name} from {root_dir}")
        self.data = self._load_dataset()
        self.total_nodes = self.data.num_nodes
        
        if self.max_nodes and self.max_nodes < self.total_nodes:
            self.node_indices = list(range(self.max_nodes))
            logger.info(f"Subsampled to first {self.max_nodes} nodes for streaming.")
        else:
            self.node_indices = list(range(self.total_nodes))
            logger.info(f"Streaming through all {self.total_nodes} nodes.")

    def _inject_outliers(self, data, contextual_ratio=0.03, structural_ratio=0.03, m_clique=10, k=50):
        """Inject synthetic contextual and structural outliers for anomaly labels."""
        n_contextual = max(10, int(data.num_nodes * contextual_ratio))
        data, yc = gen_contextual_outlier(data, n=n_contextual, k=k, seed=42)
        n_clique = max(1, int((data.num_nodes * structural_ratio) / m_clique))
        data, ys = gen_structural_outlier(data, m=m_clique, n=n_clique, seed=42)
        data.y = torch.logical_or(yc, ys).long()
        return data

    def _load_dataset(self):
        if self.dataset_name == "yelp":
            path = os.path.join(self.root_dir, "Yelp")
            dataset = Yelp(root=path)
            data = dataset[0]
            if data.y.dim() > 1:
                data.y = (data.y.sum(dim=-1) > 0).long()
            data = self._inject_outliers(data, contextual_ratio=0.01, structural_ratio=0.01, m_clique=8)
            return data
            
        elif self.dataset_name == "amazon":
            path = os.path.join(self.root_dir, "Amazon")
            dataset = Amazon(root=path, name="Computers")
            data = dataset[0]
            data = self._inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.02, m_clique=8)
            return data
            
        elif self.dataset_name == "reddit":
            path = os.path.join(self.root_dir, "Reddit")
            dataset = Reddit(root=path)
            data = dataset[0]
            data = self._inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.01, m_clique=10)
            return data
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def __iter__(self):
        """
        Generator yielding chunks of size chunk_size.
        Each chunk contains user reviews with synthesized text depending on their label.
        """
        n = len(self.node_indices)
        for i in range(0, n, self.chunk_size):
            chunk_indices = self.node_indices[i:i + self.chunk_size]
            chunk_data = []
            
            for idx in chunk_indices:
                label = self.data.y[idx].item()
                
                if label == 1:
                    text = (
                        f"User_{idx}: Register now to get a free crypto signup bonus of 5000 USDT! "
                        f"Fully verified and guaranteed returns. Signup here: http://scam-link-{idx}.com"
                    )
                else:
                    text = (
                        f"User_{idx}: I purchased this product. "
                        f"It works fine and I have no complaints about it."
                    )
                
                chunk_data.append({
                    "node_idx": idx,
                    "user_id": f"User_{idx}",
                    "text": text,
                    "label": label
                })
                
            yield chunk_data
