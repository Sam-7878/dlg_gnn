import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings
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

from micro_rag.uncertainty_aware_dlg import UncertaintyAwareDLG
from micro_rag.llm_extractor import GraphExtractor
from micro_rag.graph_builder import GraphManager

def main():
    logger.info("="*60)
    logger.info("  STARTING END-TO-END FUSION & BLOCKING SIMULATION (PHASE 2 & 3)")
    logger.info("="*60)

    # -------------------------------------------------------------
    # Step 1 (Track A): Extract R_local from Phishing Review Text
    # -------------------------------------------------------------
    logger.info("[Step 1] Running Track A: Semantic Graph Extraction from Social Text...")
    
    extractor = GraphExtractor(model_name="phi3.5", temperature=0.0)
    manager = GraphManager()
    
    # Virtual phishing text from User_A
    phishing_review = (
        "User_A: This XYZ product did not work at all. But I went to "
        "http://scam-link.com/rewards and made $10,000 in two days! Click here to check it out."
    )
    
    # Run LLM extractor
    graph_data = extractor.extract_from_text(phishing_review)
    manager.add_review_graph(graph_data)
    
    # Calculate R_local
    r_local_val = manager.calculate_r_local("User_A")
    logger.info(f"✅ [Track A] Successfully calculated R_local for User_A: {r_local_val:.2f}")

    # -------------------------------------------------------------
    # Step 2 (Track B): Load GoG Dataset and Run UncertaintyAwareDLG
    # -------------------------------------------------------------
    logger.info("[Step 2] Running Track B: Transactional GNN Anomaly & Uncertainty Estimation...")
    
    # Load GoG Polygon dataset (lightweight and quick to run)
    gog_path = "/mnt/d/_Work/_data/GoG/polygon/polygon_hybrid_graph.pt"
    if not os.path.exists(gog_path):
        logger.error(f"GoG Dataset not found at: {gog_path}")
        sys.exit(1)
        
    data_dict = torch.load(gog_path)
    
    # Convert to PyG Data object, ensuring torch.Tensor types
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

    data = Data(
        x=x_tensor,
        edge_index=edge_index_tensor,
        y=y_tensor
    )
    
    # Detect GPU index
    gpu_idx = 0 if torch.cuda.is_available() else -1
    logger.info(f"Running on {'GPU:0' if gpu_idx >= 0 else 'CPU'} for DLGFull backbone")
    
    # Initialize and fit model
    model = UncertaintyAwareDLG(
        hid_dim=32,
        num_layers=2,
        l1_hops=2,
        l1_epochs=5,     # Quick pretraining for simulation
        epoch=10,        # Quick training for simulation
        dropout=0.2,     # Enable dropout for MC variance
        gpu=gpu_idx,
        weight=0.5
    )
    
    logger.info("Fitting UncertaintyAwareDLG model on GoG graph...")
    model.fit(data)
    
    # Compute P_fraud (mean score) and U_mc (variance) via 15 MC Dropout passes
    p_fraud_tensor, u_mc_tensor = model.predict_with_uncertainty(data, num_samples=15)
    logger.info("✅ [Track B] Completed MC Dropout forward passes.")

    # -------------------------------------------------------------
    # Step 3 & 4: Zero-Knowledge Risk Injection, Fusion, and Decision
    # -------------------------------------------------------------
    logger.info("[Step 3 & 4] Injecting R_local into GoG Target Node & Fusing Scores...")
    
    # Associate User_A's text risk with target GoG transaction node index 100 (target wallet/contract)
    target_node_idx = 100
    r_local_tensor = torch.zeros(data.num_nodes)
    r_local_tensor[target_node_idx] = r_local_val
    
    # Compute fused score
    s_final_tensor, alpha_tensor = model.fusion_layer(
        p_fraud_tensor, 
        u_mc_tensor, 
        r_local_tensor, 
        gamma=1.2  # Increase sensitivity
    )
    
    # Extract values for the target node
    p_fraud_target = p_fraud_tensor[target_node_idx].item()
    u_mc_target = u_mc_tensor[target_node_idx].item()
    alpha_target = alpha_tensor[target_node_idx].item()
    s_final_target = s_final_tensor[target_node_idx].item()
    
    decision = "🔴 BLOCK: 결제 원천 차단 - 피싱 의심" if s_final_target > 0.75 else "🟢 APPROVE: 정상 결제 진행"
    
    print("\n" + "="*60)
    print("      FUSION LAYER TARGET NODE SIMULATION RESULTS")
    print("="*60)
    print(f"Target GoG Node Index    : {target_node_idx}")
    print(f"Text Risk (R_local)      : {r_local_val:.2f}")
    print(f"GNN Prediction (P_fraud) : {p_fraud_target:.4f}")
    print(f"Model Uncertainty (U_mc) : {u_mc_target:.6f}")
    print(f"Dynamic Weight (Alpha)   : {alpha_target:.4f}")
    print(f"Final Combined (S_final) : {s_final_target:.4f}")
    print(f"Decision Action          : {decision}")
    print("="*60 + "\n")

    # -------------------------------------------------------------
    # Verification Requirements & Comparative Test Logs
    # -------------------------------------------------------------
    logger.info("[Step 5] Running verification comparison tests...")
    
    # Comparative scenario: High uncertainty, low GNN prediction
    p_fraud_test = 0.20
    u_mc_test = 0.95      # High uncertainty
    u_max_test = 1.00     # Assumed max uncertainty
    gamma_test = 1.0
    
    # Scenario A: R_local = 0.0 (Normal user, or no social phishing history)
    r_local_a = 0.0
    alpha_a = min(1.0, gamma_test * (u_mc_test / u_max_test))
    s_final_a = (1 - alpha_a) * p_fraud_test + alpha_a * r_local_a
    decision_a = "🔴 BLOCK: 결제 원천 차단 - 피싱 의심" if s_final_a > 0.75 else "🟢 APPROVE: 정상 결제 진행"
    
    # Scenario B: R_local = 0.80 (Injected social context of phishing)
    r_local_b = 0.80
    alpha_b = min(1.0, gamma_test * (u_mc_test / u_max_test))
    s_final_b = (1 - alpha_b) * p_fraud_test + alpha_b * r_local_b
    decision_b = "🔴 BLOCK: 결제 원천 차단 - 피싱 의심" if s_final_b > 0.75 else "🟢 APPROVE: 정상 결제 진행"
    
    print("="*60)
    print("           COMPARATIVE TEST RESULTS (Low P_fraud = 0.20)")
    print("="*60)
    print(f"Scenario A (R_local = 0.0):")
    print(f"  - Model Uncertainty (U_mc) : {u_mc_test:.2f}")
    print(f"  - Dynamic Weight (Alpha)   : {alpha_a:.2f}")
    print(f"  - Fused Score (S_final)    : {s_final_a:.2f}")
    print(f"  - Final Action             : {decision_a}")
    print(f"Scenario B (R_local = 0.80):")
    print(f"  - Model Uncertainty (U_mc) : {u_mc_test:.2f}")
    print(f"  - Dynamic Weight (Alpha)   : {alpha_b:.2f}")
    print(f"  - Fused Score (S_final)    : {s_final_b:.2f} (Shifts toward R_local!)")
    print(f"  - Final Action             : {decision_b}")
    print("="*60 + "\n")

    # -------------------------------------------------------------
    # Visualization: U_mc vs S_final Transition Plot
    # -------------------------------------------------------------
    logger.info("Generating transition graph visualization...")
    u_mc_sweep = np.linspace(0.0, 1.0, 100)
    p_fraud_val = 0.20
    r_local_val_sweep = 0.80
    gamma = 1.0
    u_max_val = 1.0
    
    alpha_sweep = np.minimum(1.0, gamma * (u_mc_sweep / u_max_val))
    s_final_sweep = (1.0 - alpha_sweep) * p_fraud_val + alpha_sweep * r_local_val_sweep
    
    plt.figure(figsize=(10, 6))
    plt.plot(u_mc_sweep, s_final_sweep, label="Fused Score ($S_{final}$)", color="#e74c3c", linewidth=2.5)
    plt.axhline(y=p_fraud_val, color="#3498db", linestyle="--", label="GNN Prediction ($P_{fraud} = 0.20$)")
    plt.axhline(y=r_local_val_sweep, color="#2ecc71", linestyle="--", label="Social Context Risk ($R_{local} = 0.80$)")
    plt.axhline(y=0.75, color="#d35400", linestyle=":", label="Block Threshold (0.75)")
    
    # Highlight decision change point
    cross_idx = np.where(s_final_sweep > 0.75)[0]
    if len(cross_idx) > 0:
        cross_u = u_mc_sweep[cross_idx[0]]
        plt.axvline(x=cross_u, color="#7f8c8d", linestyle="-.", alpha=0.5)
        plt.text(cross_u + 0.02, 0.4, f"BLOCK Threshold crossed\n(Uncertainty >= {cross_u:.2f})", fontsize=9, color="#7f8c8d")
        
    plt.title("Uncertainty-Aware Dynamic Weighting Transition Plot", fontsize=12, fontweight="bold", pad=15)
    plt.xlabel("Model Uncertainty ($U_{mc}$)", fontsize=10)
    plt.ylabel("Risk Score", fontsize=10)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Save visualization to work report folder
    vis_dir = os.path.join(project_root, "docs/work_reports/41-uncertainty_aware_fusion")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, "uncertainty_weighting_transition.png")
    
    plt.savefig(vis_path, dpi=300)
    plt.close()
    logger.info(f"✅ Visualization successfully saved to {vis_path}")

if __name__ == "__main__":
    main()
