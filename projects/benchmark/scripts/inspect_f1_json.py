import json
import pandas as pd
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent.parent
df = pd.read_csv(root / "projects/benchmark/publication/a18/f1_provenance_a18.csv")

samples = df.drop_duplicates(subset=["dataset"])
for idx, row in samples.iterrows():
    ds = row["dataset"]
    m = row["model"]
    s = row["seed"]
    fpath = root / row["metric_path"]
    with open(fpath) as f:
        j = json.load(f)
    print(f"=== Dataset: {ds} ({m}, seed={s}) ===")
    print("  File:", row["metric_path"])
    print("  All Keys:", list(j.keys()))
    if "thresholded" in j:
        t = j["thresholded"]
        if isinstance(t, dict):
            print("  thresholded content:", {k: v for k, v in t.items() if k != "selected_node_ids"})
        else:
            print("  thresholded:", t)
    for k in ["f1", "test_f1", "validation_f1", "val_f1", "oracle_best_f1", "threshold", "best_threshold", "decision_threshold"]:
        if k in j:
            print(f"  {k}: {j[k]}")
    print()
