import os
import torch
import logging
import pandas as pd
from torch_geometric.datasets import Yelp, Amazon, Reddit
from pygod.generator import gen_contextual_outlier, gen_structural_outlier

logger = logging.getLogger(__name__)

def resolve_data_root(root_dir: str) -> str:
    """
    Resolves data root path dynamically to handle both WSL2 Linux paths and Windows paths.
    """
    paths = [
        root_dir,
        "/mnt/d/_Work/_data/DLG",
        "D:\\_Work\\_data\\DLG",
        "d:/_Work/_data/DLG",
        "../data/DLG",
        "./data/DLG"
    ]
    for p in paths:
        if os.path.exists(p):
            if any(os.path.exists(os.path.join(p, d)) for d in ["Yelp", "Amazon", "Reddit", "yelp", "amazon", "reddit"]):
                return p
    return root_dir

class UnifiedStreamLoader:
    """
    Unified multi-domain streaming loader for Yelp, Amazon, and Reddit.
    Applies the adapter pattern to yield node chunks with a standardized text schema.
    Uses pandas.read_csv(..., chunksize=X) to guarantee strict memory efficiency.
    """
    def __init__(self, dataset_name: str, root_dir: str, chunk_size: int = 500, max_nodes: int = 2000):
        self.dataset_name = dataset_name.lower()
        self.raw_root = resolve_data_root(root_dir)
        self.chunk_size = chunk_size
        self.max_nodes = max_nodes
        
        # Determine exact dataset subfolder name
        subfolders = {"yelp": "Yelp", "amazon": "Amazon", "reddit": "Reddit"}
        subfolder_name = subfolders.get(self.dataset_name, dataset_name)
        self.dataset_dir = os.path.join(self.raw_root, subfolder_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        
        # Decide CSV output path
        csv_names = {
            "yelp": "yelp_reviews.csv",
            "amazon": "amazon_reviews.csv",
            "reddit": "reddit_posts.csv"
        }
        csv_name = csv_names.get(self.dataset_name, f"{self.dataset_name}_reviews.csv")
        self.csv_path = os.path.join(self.dataset_dir, csv_name)
        
        logger.info(f"UnifiedStreamLoader for [{dataset_name}]. CSV Path: {self.csv_path}")
        
        # Create mock CSV if it does not exist
        if not os.path.exists(self.csv_path):
            self._create_mock_csv()

    def _inject_outliers(self, data, contextual_ratio=0.03, structural_ratio=0.03, m_clique=10, k=50):
        """Inject synthetic contextual and structural outliers for anomaly labels."""
        n_contextual = max(10, int(data.num_nodes * contextual_ratio))
        data, yc = gen_contextual_outlier(data, n=n_contextual, k=k, seed=42)
        n_clique = max(1, int((data.num_nodes * structural_ratio) / m_clique))
        data, ys = gen_structural_outlier(data, m=m_clique, n=n_clique, seed=42)
        data.y = torch.logical_or(yc, ys).long()
        return data

    def _load_pyg_data(self):
        """Loads underlying PyG dataset to align indices and outlier labels."""
        logger.info(f"Mock CSV not found. Loading underlying PyG [{self.dataset_name}] dataset from {self.raw_root}...")
        
        if self.dataset_name == "yelp":
            path = os.path.join(self.raw_root, "Yelp")
            dataset = Yelp(root=path)
            data = dataset[0]
            if data.y.dim() > 1:
                data.y = (data.y.sum(dim=-1) > 0).long()
            data = self._inject_outliers(data, contextual_ratio=0.01, structural_ratio=0.01, m_clique=8)
            return data
            
        elif self.dataset_name == "amazon":
            path = os.path.join(self.raw_root, "Amazon")
            dataset = Amazon(root=path, name="Computers")
            data = dataset[0]
            data = self._inject_outliers(data, contextual_ratio=0.03, structural_ratio=0.02, m_clique=8)
            return data
            
        elif self.dataset_name == "reddit":
            path = os.path.join(self.raw_root, "Reddit")
            dataset = Reddit(root=path)
            data = dataset[0]
            data = self._inject_outliers(data, contextual_ratio=0.02, structural_ratio=0.01, m_clique=10)
            return data
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _create_mock_csv(self):
        """
        Builds and saves the mock CSV with domain-specific schemas mapping to outlier labels.
        """
        data = self._load_pyg_data()
        total_nodes = data.num_nodes
        num_to_generate = min(self.max_nodes, total_nodes)
        
        logger.info(f"Generating {num_to_generate} mock reviews/posts for {self.dataset_name}...")
        
        rows = []
        for idx in range(num_to_generate):
            label = int(data.y[idx].item())
            user_id = f"User_{idx}"
            
            if self.dataset_name == "yelp":
                if label == 1:
                    text = (
                        f"Don't miss this! Earn $5000 easily by reviewing our products. "
                        f"Sign up now at http://yelp-rewards-{idx}.com"
                    )
                else:
                    text = (
                        f"The service at this restaurant was outstanding. "
                        f"The steak was cooked perfectly and the staff was very friendly."
                    )
                rows.append({
                    "node_idx": idx,
                    "user_id": user_id,
                    "review_text": text,
                    "label": label
                })
                
            elif self.dataset_name == "amazon":
                if label == 1:
                    product_review = "Direct purchase available. Claim your half-price iPhone now!"
                    url = f"http://amazon-discounts-{idx}.net/deal"
                else:
                    product_review = "I bought this kitchen mixer last month. It works fine and has various speed settings."
                    url = ""
                rows.append({
                    "node_idx": idx,
                    "user_id": user_id,
                    "product_review": product_review,
                    "url": url,
                    "label": label
                })
                
            elif self.dataset_name == "reddit":
                if label == 1:
                    body = (
                        f"Target price is $100! Everyone buy now and hold the line! "
                        f"Short squeeze incoming at http://reddit-pump-{idx}.xyz"
                    )
                else:
                    body = (
                        f"I'm analyzing the quarterly financial statements. "
                        f"The revenue increased by 5% but operational costs are also up."
                    )
                rows.append({
                    "node_idx": idx,
                    "user_id": user_id,
                    "body": body,
                    "label": label
                })

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        logger.info(f"Successfully saved mock CSV to {self.csv_path}")

    def __iter__(self):
        """
        Generator yielding chunks of unified schema.
        Unified schema: {'node_idx', 'user_id', 'text', 'label'}
        """
        logger.info(f"Streaming [{self.dataset_name}] via pandas.read_csv chunksize={self.chunk_size}")
        
        processed_nodes = 0
        
        # Read the CSV in streaming chunks
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            if processed_nodes >= self.max_nodes:
                break
                
            chunk_data = []
            for _, row in chunk.iterrows():
                if processed_nodes >= self.max_nodes:
                    break
                    
                node_idx = int(row["node_idx"])
                user_id = row["user_id"]
                label = int(row["label"])
                
                # Dynamic adapter schema transformation
                if self.dataset_name == "yelp":
                    text = row["review_text"]
                elif self.dataset_name == "amazon":
                    product_review = row["product_review"]
                    url = row["url"]
                    if pd.isna(url) or not str(url).strip():
                        text = product_review
                    else:
                        text = f"{product_review} (Source URL: {url})"
                elif self.dataset_name == "reddit":
                    text = row["body"]
                else:
                    text = str(row.get("text", ""))
                
                chunk_data.append({
                    "node_idx": node_idx,
                    "user_id": user_id,
                    "text": text,
                    "label": label
                })
                processed_nodes += 1
                
            yield chunk_data

# Kept for legacy compatibility
class StreamingDataLoader(UnifiedStreamLoader):
    pass
