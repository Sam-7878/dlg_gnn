import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score
)

def calculate_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Computes classic ML and fraud-specific classification metrics.
    """
    # Round predictions based on threshold
    y_pred = (y_prob >= threshold).astype(int)
    
    # Imbalance stats
    n_total = len(y_true)
    n_fraud = int(np.sum(y_true))
    fraud_ratio = n_fraud / max(n_total, 1)

    # Basic Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc_roc = 0.5
        
    try:
        auc_pr = average_precision_score(y_true, y_prob)
    except Exception:
        auc_pr = fraud_ratio

    # Confusion matrix elements for rate calculation
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)

    # Expected Cost calculation
    # C_FN = 10.0 (high cost of missing fraud), C_FP = 1.0 (cost of checking false positive)
    c_fn = 10.0
    c_fp = 1.0
    expected_cost = fn * c_fn + fp * c_fp
    
    # Baseline Cost (e.g. baseline that flags nothing)
    cost_baseline = n_fraud * c_fn

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "auc_roc": round(float(auc_roc), 4),
        "auc_pr": round(float(auc_pr), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "expected_cost": round(float(expected_cost), 2),
        "cost_saving": round(float(cost_baseline - expected_cost), 2)
    }
