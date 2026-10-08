import json
from pathlib import Path
import pandas as pd

raw_dir = Path("outputs/benchmark/a03_production/raw")
csv_path = Path("evaluation/benchmark/v2/a03/crypto_production_runs.csv")
df = pd.read_csv(csv_path)

for f in raw_dir.glob("*__DLG-Aug__*.json"):
    data = json.loads(f.read_text())
    res = data.get("result") or data
    if res.get("status") == "success" or data.get("status") == "success":
        c = data["dataset"]
        s = int(data["seed"])
        mask = (df["dataset"] == c) & (df["model"] == "DLG-Aug") & (df["seed"] == s)
        df.loc[mask, "status"] = "success"
        df.loc[mask, "roc_auc"] = res.get("roc_auc")
        df.loc[mask, "pr_auc"] = res.get("pr_auc")
        df.loc[mask, "validation_f1"] = res.get("validation_f1")
        df.loc[mask, "precision_at_k"] = res.get("precision_at_k")
        df.loc[mask, "recall_at_k"] = res.get("recall_at_k")
        df.loc[mask, "topk_f1"] = res.get("topk_f1")

df.to_csv(csv_path, index=False)
print("Updated crypto_production_runs.csv successfully!")
print(df[df["model"] == "DLG-Aug"][["dataset", "seed", "status", "roc_auc", "pr_auc"]])
