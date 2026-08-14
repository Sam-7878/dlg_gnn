from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gog_fraud.data.sci_v2.audit import audit_dataset


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root", required=True)
    p.add_argument("--chains", nargs="+", default=["ethereum", "bsc", "polygon"])
    p.add_argument("--all-folds", action="store_true"); p.add_argument("--strict", action="store_true")
    a = p.parse_args(); print(json.dumps(audit_dataset(a.dataset_root, chains=tuple(a.chains), strict=a.strict), sort_keys=True, indent=2))


if __name__ == "__main__": main()
