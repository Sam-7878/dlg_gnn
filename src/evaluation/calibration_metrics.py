import numpy as np

def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> dict:
    """
    Computes calibration metrics: ECE, MCE, Brier Score, and NLL.
    """
    # Safeguard lengths
    n = len(y_true)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "brier_score": 0.0, "nll": 0.0}

    # 1. Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    mce = 0.0

    for m in range(num_bins):
        bin_lower = bin_boundaries[m]
        bin_upper = bin_boundaries[m + 1]
        
        # Nodes falling into the current probability bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        # Handle boundary case for 1.0
        if m == num_bins - 1:
            in_bin = in_bin | (y_prob == bin_upper)
            
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            abs_diff = np.abs(bin_acc - bin_conf)
            
            ece += (bin_size / n) * abs_diff
            mce = max(mce, abs_diff)

    # 2. Brier Score
    brier = np.mean((y_prob - y_true) ** 2)

    # 3. Negative Log-Likelihood (NLL)
    eps = 1e-15
    y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
    nll = -np.mean(y_true * np.log(y_prob_clipped) + (1.0 - y_true) * np.log(1.0 - y_prob_clipped))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(float(brier), 4),
        "nll": round(float(nll), 4)
    }
