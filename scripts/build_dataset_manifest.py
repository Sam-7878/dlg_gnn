from __future__ import annotations

import argparse
from pathlib import Path

from gog_fraud.data.io.dataset_manifest import build_dataset_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--labels")
    parser.add_argument("--chains", nargs="+", default=["ethereum", "bsc", "polygon"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    for chain in args.chains:
        manifest = build_dataset_manifest(args.source_root, chain=chain, labels_path=args.labels)
        manifest.write(output / f"{chain}.json", output / f"{chain}.csv")


if __name__ == "__main__":
    main()
