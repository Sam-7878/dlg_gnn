from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gog_fraud.data.splits.artifact import build_pooled_split_artifacts, build_split_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chains", nargs="+", default=["ethereum", "bsc", "polygon"])
    args = parser.parse_args()
    for chain in args.chains:
        outputs = build_split_artifacts(
            transaction_root=args.transaction_root, labels_path=args.labels, chain=chain,
            source_manifest=Path(args.manifest_dir) / f"{chain}.json", output_dir=args.output_dir,
        )
        print("\n".join(str(path) for path in outputs))
    if set(args.chains) == {"ethereum", "bsc", "polygon"}:
        print("\n".join(str(path) for path in build_pooled_split_artifacts(split_dir=args.output_dir)))


if __name__ == "__main__":
    main()
