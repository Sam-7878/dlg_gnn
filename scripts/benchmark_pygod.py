import time
import json
import tracemalloc
import torch
import warnings
import os
import sys

# Ensure our local src is in the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from pygod.utils import load_data
from pygod.models import DOMINANT

from gog_fraud.models.pygod.dlg import DLG

warnings.filterwarnings("ignore")

def track_performance(model, data, model_name="Model"):
    """
    Fits the model and tracks Time, RAM, and VRAM.
    """
    print(f"\n[INFO] Starting training for {model_name}...")
    
    # Setup metrics
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = device.type == 'cuda'
    
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    
    tracemalloc.start()
    start_time = time.time()
    
    # Train the model
    model.fit(data)
    
    end_time = time.time()
    _, peak_ram = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_vram = torch.cuda.max_memory_allocated(device) if is_cuda else 0
    
    # Make predictions
    print(f"[INFO] Evaluating {model_name}...")
    scores = model.decision_function(data)
    
    # Handle PyGOD / torch outputs (might be numpy or tensor)
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
        
    y_true = data.y.cpu().numpy() if isinstance(data.y, torch.Tensor) else data.y
    
    # Metrics
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except ValueError:
        roc_auc = 0.0
        
    try:
        pr_auc = average_precision_score(y_true, scores)
    except ValueError:
        pr_auc = 0.0
        
    # F1 Score requires binary labels. We use median or a simple threshold.
    # PyGOD models typically return anomaly scores. 
    # For a fair comparison, we can use the top K% as anomalies where K is the ground truth anomaly ratio.
    k_ratio = sum(y_true) / len(y_true)
    threshold = sorted(scores, reverse=True)[int(k_ratio * len(scores))]
    preds = (scores >= threshold).astype(int)
    
    f1 = f1_score(y_true, preds)
    
    results = {
        "ROC-AUC": round(roc_auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "F1-Score": round(f1, 4),
        "Time (s)": round(end_time - start_time, 2),
        "Peak RAM (MB)": round(peak_ram / (1024 * 1024), 2),
        "Peak VRAM (MB)": round(peak_vram / (1024 * 1024), 2)
    }
    
    print(f"[SUCCESS] {model_name} Evaluation Complete.")
    return results

def main():
    print("========================================")
    print("      PyGOD Benchmark: DOMINANT vs DLG  ")
    print("========================================")
    
    # 1. Load Data
    print("[INFO] Loading Cora dataset with injected anomalies...")
    # pygod's load_data automatically injects structural and contextual anomalies 
    # into standard datasets like 'cora' if they are not naturally anomalous.
    data = load_data("cora")
    print(f"[INFO] Data Loaded: {data.x.size(0)} nodes, {data.edge_index.size(1)} edges.")
    print(f"[INFO] Anomaly ratio: {data.y.sum().item() / data.y.size(0):.4f}")
    
    # Determine common parameters
    gpu_id = 0 if torch.cuda.is_available() else -1
    epoch = 100
    
    # 2. Initialize Models
    # DOMINANT
    dominant = DOMINANT(epoch=epoch, gpu=gpu_id, verbose=0)
    
    # DLG Wrapper
    dlg = DLG(
        epoch=epoch, 
        gpu=gpu_id, 
        subgraph_batch_size=256, # The specific partitioning advantage
        verbose=0
    )
    
    # 3. Run Benchmark
    benchmark_results = {}
    
    # Run DOMINANT
    res_dominant = track_performance(dominant, data, model_name="DOMINANT")
    benchmark_results["DOMINANT"] = res_dominant
    
    # Run DLG
    res_dlg = track_performance(dlg, data, model_name="DLG")
    benchmark_results["DLG"] = res_dlg
    
    # 4. Output Results
    print("\n========================================")
    print("           BENCHMARK RESULTS            ")
    print("========================================")
    
    # Print Table Header
    metrics_keys = list(benchmark_results["DOMINANT"].keys())
    header = f"{'Model':<12} | " + " | ".join([f"{k:<14}" for k in metrics_keys])
    print(header)
    print("-" * len(header))
    
    for model_name, metrics in benchmark_results.items():
        row = f"{model_name:<12} | " + " | ".join([f"{metrics[k]:<14}" for k in metrics_keys])
        print(row)
        
    print("========================================\n")
    
    # Save to JSON
    save_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), 
        '../docs/work_reports/22-pygod_benchmark_cora_dominant/benchmark_results.json'
    ))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(benchmark_results, f, indent=4)
        
    print(f"[INFO] Results saved to: {save_path}")

if __name__ == "__main__":
    main()
