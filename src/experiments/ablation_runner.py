import os
import sys
import yaml
import pandas as pd
import logging

# Setup paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.insert(0, project_root)

from src.experiments.scenario_manager import ScenarioManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class AblationRunner:
    """
    Executes multiple scenarios to verify the contribution of:
    - GraphRAG
    - Monte Carlo uncertainty
    - Streaming execution
    - Privacy-preserving risk vectorization
    """
    def __init__(self):
        self.configs_dir = os.path.join(project_root, "configs/experiments")
        self.ablation_dir = os.path.join(project_root, "outputs/ablation")
        os.makedirs(self.ablation_dir, exist_ok=True)
        
    def run_all(self) -> pd.DataFrame:
        ablation_results = []
        
        # Define scenarios and corresponding config files / overrides
        scenarios = [
            ("Full Model", "exp_005_full_model.yaml", {}),
            ("Without GraphRAG", "exp_003_mc_streaming.yaml", {}),
            ("Without MC", "exp_004_micro_rag.yaml", {}),
            ("Without Streaming", "exp_001_static_dlg_gnn.yaml", {}),
            ("Without Privacy-Vector (Raw Text)", "exp_005_full_model.yaml", {"risk_injection": {"mode": "raw_context", "fusion_method": "uncertainty_weighted", "enabled": True, "alpha_init": 0.7, "beta_init": 0.3}}),
            ("Without Uncertainty-weighted Fusion", "exp_005_full_model.yaml", {"risk_injection": {"fusion_method": "fixed_weight", "mode": "privacy_preserving_vector", "enabled": True, "alpha_init": 0.7, "beta_init": 0.3}}),
            ("GNN Only", "exp_002_streaming_dlg_gnn.yaml", {})
        ]
        
        for name, config_file, overrides in scenarios:
            cfg_path = os.path.join(self.configs_dir, config_file)
            if not os.path.exists(cfg_path):
                logger.error(f"Config not found: {cfg_path}")
                continue
                
            # Run using ScenarioManager
            manager = ScenarioManager(cfg_path)
            # Apply dynamic overrides
            if overrides:
                for key in overrides:
                    if key in manager.config:
                        manager.config[key].update(overrides[key])
                        
            res = manager.run()
            
            # Extract essential metrics
            ablation_results.append({
                "Setting": name,
                "AUC-ROC": res.get("auc_roc", 0.0),
                "AUC-PR": res.get("auc_pr", 0.0),
                "F1-Score": res.get("f1", 0.0),
                "Recall": res.get("recall", 0.0),
                "Precision": res.get("precision", 0.0),
                "ECE": res.get("ece", 0.0),
                "Brier Score": res.get("brier_score", 0.0),
                "Avg Latency (ms)": res.get("latency_mc_sampling_ms", res.get("latency_gnn_inference_ms", 0.0)) + res.get("latency_risk_vectorizer_ms", 0.0) + res.get("latency_fusion_layer_ms", 0.0),
                "Comm (Bytes)": res.get("risk_vector_bytes", 96) if "Without Privacy-Vector" not in name else res.get("raw_context_bytes", 2048)
            })

        df = pd.DataFrame(ablation_results)
        
        # Save CSV
        csv_out = os.path.join(self.ablation_dir, "ablation_summary.csv")
        df.to_csv(csv_out, index=False)
        logger.info(f"Ablation CSV report saved to {csv_out}")
        
        # Save Markdown
        md_out = os.path.join(self.ablation_dir, "ablation_summary.md")
        with open(md_out, "w", encoding="utf-8") as f:
            f.write("# Ablation Study Summary Table\n\n")
            f.write(df.to_markdown(index=False))
        logger.info(f"Ablation Markdown report saved to {md_out}")
        
        return df

if __name__ == "__main__":
    runner = AblationRunner()
    runner.run_all()
