from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gog_fraud.data.sci_v2.builder import BuildOptions, build_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", required=True); p.add_argument("--legacy-root", required=True)
    p.add_argument("--output-root", required=True); p.add_argument("--labels")
    p.add_argument("--global-mapping-root"); p.add_argument("--chains", nargs="+", default=["ethereum", "bsc", "polygon"])
    p.add_argument("--max-files", type=int); p.add_argument("--strict", action="store_true")
    a = p.parse_args(); raw = Path(a.raw_root)
    summary = build_dataset(BuildOptions(raw, Path(a.legacy_root), Path(a.output_root),
        Path(a.labels or raw.parent / "labels.csv"), Path(a.global_mapping_root or raw.parent / "global_graph"),
        tuple(a.chains), a.max_files, a.strict))
    print(json.dumps(summary, sort_keys=True, indent=2))


if __name__ == "__main__": main()
