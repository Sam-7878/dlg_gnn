import sys
import os
import time

log_file = open("/tmp/gadnr_trace.log", "w", buffering=1)

def log(msg):
    log_file.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    log_file.flush()

log("Script started")

import torch
log("torch imported")
import numpy as np
log("numpy imported")
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
log("sklearn imported")
from gog_fraud.extensions.defense.defense_registry import load_defense_dataset
log("defense registry imported")
from gog_fraud.models.pygod.gadnr import GADNR
log("GADNR class imported")

def test_direct():
    for name in ["DARPA-TC-THEIA", "LANL-RedTeam"]:
        log(f"Loading {name}...")
        data = load_defense_dataset(name)
        log(f"Loaded {name}: N={data.num_nodes}, E={data.num_edges}")
        log("Creating GADNR instance...")
        model = GADNR(epoch=5, gpu=0, batch_size=0, verbose=0, num_neigh=-1)
        log("Fitting GADNR...")
        t0 = time.time()
        model.fit(data)
        elapsed = time.time() - t0
        log(f"Fit completed in {elapsed:.2f}s!")
        score = model.decision_function(data)
        log("Decision score computed!")
        y = data.y.numpy()
        s = score.cpu().numpy() if isinstance(score, torch.Tensor) else np.array(score)
        roc = roc_auc_score(y, s)
        pr = average_precision_score(y, s)
        log(f"Result {name}: ROC={roc:.4f}, PR={pr:.4f}")

if __name__ == "__main__":
    try:
        test_direct()
        log("All done successfully!")
    except Exception as e:
        import traceback
        log(f"Exception: {e}\n{traceback.format_exc()}")
