from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gog_fraud.data.sci_v2.builder import _resolve_legacy_mapping


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--dataset-root", required=True)
    p.add_argument("--legacy-root", required=True); p.add_argument("--chains", nargs="+", default=["ethereum", "bsc", "polygon"])
    a = p.parse_args(); root = Path(a.dataset_root); legacy = Path(a.legacy_root)
    for chain in a.chains:
        manifest = json.loads((root / f"manifests/{chain}.json").read_text(encoding="utf-8"))
        result = _resolve_legacy_mapping(manifest["records"], legacy / chain / "graphs")
        result["audit_version"] = "legacy-compatibility-v2-list-schema"
        path = root / f"audit/{chain}_legacy_compatibility_v2.json"
        path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"chain": chain, **{k: result[k] for k in ("status", "legacy_graphs", "resolved", "ambiguous", "missing", "label_orientation")}}, sort_keys=True), flush=True)


if __name__ == "__main__": main()
