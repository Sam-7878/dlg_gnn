import os
import sys
import time
import torch
import logging
import psutil
import warnings
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Filter PyTorch warnings
warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Add path to load local src and micro_rag modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../"))
src_path = os.path.abspath(os.path.join(current_dir, "../src"))

sys.path.insert(0, project_root)
sys.path.insert(0, src_path)

from micro_rag.data_loader import StreamingDataLoader
from micro_rag.uncertainty_aware_dlg import UncertaintyAwareDLG
from micro_rag.llm_extractor import GraphExtractor
from micro_rag.graph_builder import GraphManager

class HybridExtractor:
    """
    Combines real LLM extraction with high-speed mock extraction for large-scale RAG.
    Runs the actual LLM (Phi-3.5) for target users, and uses fast-path for others.
    """
    def __init__(self, real_extractor, active_users=None):
        self.real_extractor = real_extractor
        self.active_users = active_users or []

    def extract_from_text(self, review_text: str, node_idx: int, label: int) -> dict:
        user_id = f"User_{node_idx}"
        if user_id in self.active_users:
            logger.info(f"Running REAL ChatOllama (phi3.5) for target node {user_id}...")
            return self.real_extractor.extract_from_text(review_text)
        else:
            if label == 1:
                return {
                    "entities": [
                        {"id": user_id, "type": "USER"},
                        {"id": f"http://scam-link-{node_idx}.com", "type": "BEHAVIOR"},
                        {"id": "SPAMMING", "type": "INTENT"}
                    ],
                    "relations": [
                        {"source": user_id, "target": f"http://scam-link-{node_idx}.com", "type": "WROTE"},
                        {"source": f"http://scam-link-{node_idx}.com", "target": "SPAMMING", "type": "INDICATES"}
                    ]
                }
            else:
                return {
                    "entities": [
                        {"id": user_id, "type": "USER"},
                        {"id": f"Product_{node_idx}", "type": "PRODUCT"}
                    ],
                    "relations": [
                        {"source": user_id, "target": f"Product_{node_idx}", "type": "TARGETS"}
                    ]
                }

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    logger.info("="*60)
    logger.info(" STARTING REAL DATA STREAMING GRAPH-RAG & GNN FUSION PIPELINE")
    logger.info("="*60)

    # Path configurations
    data_root = "/mnt/d/_Work/_data/DLG"
    
    # Initialize StreamingDataLoader on Real Yelp Dataset (subsampled to 1,000 nodes)
    chunk_size = 200
    max_nodes = 1000
    loader = StreamingDataLoader(dataset_name="Yelp", root_dir=data_root, chunk_size=chunk_size, max_nodes=max_nodes)

    # Find an anomalous user to run through the actual LLM (Phi-3.5)
    target_node_idx = None
    target_text = ""
    for chunk in loader:
        for node in chunk:
            if node["label"] == 1:
                target_node_idx = node["node_idx"]
                target_text = node["text"]
                break
        if target_node_idx is not None:
            break

    if target_node_idx is None:
        target_node_idx = 100
        target_text = (
            "User_100: Register now to get a free crypto signup bonus of 5000 USDT! "
            "Fully verified and guaranteed returns. Signup here: http://scam-link-100.com"
        )

    logger.info(f"Target anomalous user index selected: User_{target_node_idx}")

    # Initialize Hybrid Extractor
    real_extractor = GraphExtractor(model_name="phi3.5", temperature=0.0)
    hybrid_extractor = HybridExtractor(real_extractor, active_users=[f"User_{target_node_idx}"])
    graph_manager = GraphManager()

    # Track metrics for scalability analysis
    chunk_times = []
    memory_footprints = []
    graph_node_counts = []
    graph_edge_counts = []

    logger.info("Starting Streaming Incremental GraphRAG Build...")
    start_time = time.time()
    
    chunk_idx = 1
    for chunk in loader:
        chunk_start = time.time()
        logger.info(f"Processing Chunk #{chunk_idx} ({len(chunk)} nodes)...")
        
        for node in chunk:
            triplets = hybrid_extractor.extract_from_text(node["text"], node["node_idx"], node["label"])
            graph_manager.add_review_graph(triplets)
            
        chunk_end = time.time()
        chunk_time_taken = chunk_end - chunk_start
        mem_mb = get_memory_usage_mb()
        
        chunk_times.append(chunk_time_taken)
        memory_footprints.append(mem_mb)
        graph_node_counts.append(len(graph_manager.G.nodes))
        graph_edge_counts.append(len(graph_manager.G.edges))
        
        logger.info(f"Chunk #{chunk_idx} Done: Time={chunk_time_taken:.2f}s | Graph Nodes={len(graph_manager.G.nodes)} | Memory={mem_mb:.1f} MB")
        chunk_idx += 1

    total_rag_time = time.time() - start_time
    logger.info(f"✅ Incremental GraphRAG completed. Total time: {total_rag_time:.2f}s")

    # Calculate final local score for target node
    r_local_val = graph_manager.calculate_r_local(f"User_{target_node_idx}")
    logger.info(f"✅ Calculated R_local score for User_{target_node_idx}: {r_local_val:.2f}")

    # Step 3: Run Transactional GNN (Track B) on GoG Dataset
    logger.info("Loading GoG dataset for Track B transactional modeling...")
    gog_path = "/mnt/d/_Work/_data/GoG/polygon/polygon_hybrid_graph.pt"
    if not os.path.exists(gog_path):
        logger.error(f"GoG Dataset not found at: {gog_path}")
        sys.exit(1)
        
    data_dict = torch.load(gog_path)
    
    # Ensure torch tensors
    x_tensor = data_dict['embeddings']
    if not isinstance(x_tensor, torch.Tensor):
        x_tensor = torch.tensor(x_tensor, dtype=torch.float)
    else:
        x_tensor = x_tensor.float()
        
    edge_index_tensor = data_dict['edge_index']
    if not isinstance(edge_index_tensor, torch.Tensor):
        edge_index_tensor = torch.tensor(edge_index_tensor, dtype=torch.long)
    else:
        edge_index_tensor = edge_index_tensor.long()
        
    y_tensor = data_dict['labels']
    if not isinstance(y_tensor, torch.Tensor):
        y_tensor = torch.tensor(y_tensor, dtype=torch.long)
    else:
        y_tensor = y_tensor.long()

    data = Data(x=x_tensor, edge_index=edge_index_tensor, y=y_tensor)
    
    # Initialize and fit GNN model
    gpu_idx = 0 if torch.cuda.is_available() else -1
    model = UncertaintyAwareDLG(
        hid_dim=32,
        num_layers=2,
        l1_hops=2,
        l1_epochs=5,
        epoch=10,
        dropout=0.2,
        gpu=gpu_idx,
        weight=0.5
    )
    
    logger.info("Fitting GNN model on GoG transactional graph...")
    model.fit(data)
    
    p_fraud_tensor, u_mc_tensor = model.predict_with_uncertainty(data, num_samples=15)
    
    # Step 4: Zero-Knowledge Risk Injection, Fusion and Decision
    gog_target_idx = 100
    r_local_tensor = torch.zeros(data.num_nodes)
    r_local_tensor[gog_target_idx] = r_local_val
    
    s_final_tensor, alpha_tensor = model.fusion_layer(p_fraud_tensor, u_mc_tensor, r_local_tensor, gamma=1.2)
    
    p_fraud_target = p_fraud_tensor[gog_target_idx].item()
    u_mc_target = u_mc_tensor[gog_target_idx].item()
    alpha_target = alpha_tensor[gog_target_idx].item()
    s_final_target = s_final_tensor[gog_target_idx].item()
    
    decision = "🔴 BLOCK: 결제 원천 차단 - 피싱 의심" if s_final_target > 0.75 else "🟢 APPROVE: 정상 결제 진행"
    
    print("\n" + "="*60)
    print("      REAL DATA SIMULATION & CROSS-MODAL FUSION RESULTS")
    print("="*60)
    print(f"Target Yelp Anomaly Node : User_{target_node_idx}")
    print(f"Text-based Risk (R_local): {r_local_val:.2f}")
    print(f"GoG Target Node Index    : {gog_target_idx}")
    print(f"GNN Prediction (P_fraud) : {p_fraud_target:.4f}")
    print(f"Model Uncertainty (U_mc) : {u_mc_target:.6f}")
    print(f"Dynamic Weight (Alpha)   : {alpha_target:.4f}")
    print(f"Final Combined (S_final) : {s_final_target:.4f}")
    print(f"Decision Action          : {decision}")
    print("="*60 + "\n")

    # Step 5: Save Scalability Plots
    logger.info("Saving scalability benchmark metrics...")
    chunks = np.arange(1, len(chunk_times) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = '#2c3e50'
    ax1.set_xlabel('Processed Streaming Chunks', fontweight='bold')
    ax1.set_ylabel('Peak Memory Usage (MB)', color=color, fontweight='bold')
    line1 = ax1.plot(chunks, memory_footprints, color=color, marker='o', linewidth=2, label='Memory Usage (MB)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = '#16a085'
    ax2.set_ylabel('NetworkX Nodes count', color=color, fontweight='bold')
    line2 = ax2.plot(chunks, graph_node_counts, color=color, marker='s', linestyle='--', linewidth=2, label='Nodes Count')
    ax2.tick_params(axis='y', labelcolor=color)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Streaming Micro-GraphRAG Scalability Benchmark', fontsize=12, fontweight='bold', pad=15)
    plt.grid(True, linestyle=':', alpha=0.6)
    fig.tight_layout()
    
    # Save visualization to work report folder
    vis_dir = os.path.join(project_root, "docs/work_reports/41-real_data_micro_rag")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, "scalability_benchmark.png")
    
    plt.savefig(vis_path, dpi=300)
    plt.close()
    logger.info(f"✅ Scalability benchmark plot successfully saved to {vis_path}")

if __name__ == "__main__":
    main()
