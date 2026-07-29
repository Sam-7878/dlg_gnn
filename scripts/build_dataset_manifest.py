"""build_dataset_manifest.py

Chain별 dataset manifest(JSON + CSV)를 생성하는 CLI 스크립트.

주요 기능:
  - chain/file 진행 상황을 stderr에 실시간 출력 (--progress)
  - SHA-256 hash 인덱스를 저장하여 재실행 시 재사용 (--resume / --hash-index)
  - 대용량 스캔에서 조기 종료 가능 (--max-files)
  - 여러 chain을 순서대로 처리

사용 예:
  # 기본 실행 (3 chain 전체)
  python scripts/build_dataset_manifest.py \\
      --source-root /mnt/d/_Work/_data/dataset/transactions \\
      --labels     /mnt/d/_Work/_data/dataset/labels.csv \\
      --output-dir results_sci/manifests

  # 진행 보고 + 재시작 가능
  python scripts/build_dataset_manifest.py \\
      --source-root /mnt/d/_Work/_data/dataset/transactions \\
      --labels     /mnt/d/_Work/_data/dataset/labels.csv \\
      --chains polygon \\
      --output-dir results_sci/manifests \\
      --progress \\
      --hash-index results_sci/manifests/.hash_index.json

  # smoke test (첫 100 파일만)
  python scripts/build_dataset_manifest.py \\
      --source-root /mnt/d/_Work/_data/dataset/transactions \\
      --output-dir /tmp/manifest_test \\
      --max-files 100 \\
      --progress
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gog_fraud.data.io.dataset_manifest import build_dataset_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build chain-level dataset manifests (JSON + CSV).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        required=True,
        help="Root directory containing chain sub-directories or transaction files.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Path to the labels CSV file (optional).",
    )
    parser.add_argument(
        "--chains",
        nargs="+",
        default=["ethereum", "bsc", "polygon"],
        help="Chain names to process (default: ethereum bsc polygon).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where JSON and CSV manifests are written.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        default=False,
        help="Print chain/file progress to stderr.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop scanning after N files per chain (useful for smoke tests). "
            "Hashes and stats will only reflect the first N files."
        ),
    )
    parser.add_argument(
        "--hash-index",
        default=None,
        metavar="PATH",
        help=(
            "Path to a JSON file used as a resumable SHA-256 hash index. "
            "Previously hashed files are reused; new hashes are appended. "
            "Enables fast re-runs without re-hashing unchanged files."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    summary: dict = {"chains": {}, "total_elapsed_s": 0.0}
    overall_start = time.monotonic()

    for chain in args.chains:
        print(f"\n=== chain: {chain} ===", file=sys.stderr, flush=True)
        chain_start = time.monotonic()

        manifest = build_dataset_manifest(
            args.source_root,
            chain=chain,
            labels_path=args.labels,
            max_files=args.max_files,
            progress=args.progress,
            hash_index_path=args.hash_index,
        )

        json_path = output / f"{chain}.json"
        csv_path  = output / f"{chain}.csv"
        manifest.write(json_path, csv_path)

        elapsed = time.monotonic() - chain_start
        summary["chains"][chain] = {
            "transactions":   manifest.transactions,
            "contracts":      manifest.contracts,
            "fraud":          manifest.fraud,
            "benign":         manifest.benign,
            "positive_ratio": manifest.positive_ratio,
            "missing_timestamp": manifest.missing_timestamp,
            "elapsed_s":      round(elapsed, 2),
            "json":           str(json_path),
            "csv":            str(csv_path),
        }

        print(
            f"  -> {manifest.transactions:,} transactions | "
            f"{manifest.fraud} fraud | {manifest.benign} benign | "
            f"{elapsed:.1f}s",
            file=sys.stderr,
            flush=True,
        )

    summary["total_elapsed_s"] = round(time.monotonic() - overall_start, 2)

    summary_path = output / "build_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"\n[done] manifests written to {output}  "
        f"(total {summary['total_elapsed_s']:.1f}s)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
