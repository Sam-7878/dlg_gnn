from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def _read_embedded_label(path: Path) -> str:
    with path.open("rb") as handle:
        handle.seek(0, 2); size = handle.tell(); handle.seek(max(0, size - 4096))
        tail = handle.read().decode("utf-8", errors="replace")
    matches = re.findall(r'"label"\s*:\s*([01])', tail)
    if not matches:
        raise ValueError("label not found in JSON tail")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--transaction-root", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    canonical = Path(args.canonical_root).resolve()
    transactions = Path(args.transaction_root).resolve()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    label_rows: dict[str, list[dict[str, str]]] = {chain: [] for chain in ("ethereum", "bsc", "polygon")}
    with Path(args.labels).open(encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            chain = str(row.get("Chain", "")).lower()
            if chain in label_rows:
                label_rows[chain].append(row)
    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": "gog-processed-v1", "canonical_root": str(canonical),
        "physical_transaction_root": str(transactions), "preprocessing_relationship": "derived_not_symlink_or_copy",
        "source_repository_version": "MISSING_UPSTREAM_VERSION_METADATA", "chains": {},
    }
    exclusions: list[dict[str, str]] = []
    for chain, rows in label_rows.items():
        categories = Counter(str(row.get("Category", "")).strip() or "UNLABELED" for row in rows)
        labels_by_contract = {str(row.get("Contract", "")).lower(): str(row.get("Category", "")) for row in rows}
        raw_files = {path.stem.lower() for path in (transactions / chain).glob("*.csv")}
        for contract in sorted(raw_files - set(labels_by_contract)):
            exclusions.append({"chain": chain, "sample_id": contract, "reason": "raw_transaction_without_label"})
        for contract in sorted(set(labels_by_contract) - raw_files):
            exclusions.append({"chain": chain, "sample_id": contract, "reason": "label_without_raw_transaction"})
        embedded = Counter()
        corrupt_graphs = 0
        graph_files = list((canonical / chain / "graphs").glob("*.json"))
        for path in graph_files:
            try:
                embedded[_read_embedded_label(path)] += 1
            except (OSError, ValueError):
                corrupt_graphs += 1
                exclusions.append({"chain": chain, "sample_id": path.stem, "reason": "corrupt_or_unlabeled_processed_graph"})
        benign = categories.get("0", 0) + categories.get("0.0", 0)
        fraud = sum(count for category, count in categories.items() if category not in {"0", "0.0", "UNLABELED"})
        audit["chains"][chain] = {
            "directory_mapping": {"processed_graphs": f"{chain}/graphs", "raw_transactions": chain},
            "label_rows": len(rows), "raw_transaction_files": len(raw_files), "processed_graph_files": len(graph_files),
            "category_distribution": dict(sorted(categories.items())),
            "binary_mapping": {"0": "benign", "all_nonzero_categories": "fraud", "missing": "unlabeled"},
            "benign": benign, "fraud": fraud, "positive_ratio": fraud / (fraud + benign) if fraud + benign else None,
            "embedded_graph_label_distribution": dict(sorted(embedded.items())), "corrupt_graphs": corrupt_graphs,
            "mapping_limitation": "Processed numeric graph filenames do not carry contract IDs; exact raw-to-graph row correspondence is not independently auditable from current artifacts.",
        }
    audit["exclusion_count"] = len(exclusions)
    (output / "dataset_provenance_label_audit.json").write_text(json.dumps(audit, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with (output / "dataset_exclusions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["chain", "sample_id", "reason"]); writer.writeheader(); writer.writerows(exclusions)
    lines = [
        "# Dataset Provenance and Label Audit", "",
        f"- Canonical processed root: `{canonical}`", f"- Physical raw transaction root: `{transactions}`",
        "- Relationship: preprocessing-derived artifacts; the roots are neither a symlink nor byte-identical copies.",
        "- Dataset version: `gog-processed-v1` (local designation)",
        "- Upstream repository/version: `MISSING_UPSTREAM_VERSION_METADATA`", "",
        "| Chain | Labels | Raw CSV | Processed graphs | Benign | Fraud | Positive ratio | Corrupt graphs |", "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for chain, row in audit["chains"].items():
        lines.append(f"| {chain} | {row['label_rows']} | {row['raw_transaction_files']} | {row['processed_graph_files']} | {row['benign']} | {row['fraud']} | {row['positive_ratio']:.6f} | {row['corrupt_graphs']} |")
    lines += ["", "Binary label rule: category `0` is benign; every non-zero category is fraud. This explains why the aggregate positive ratio may exceed 0.5 and must be described as a fraud-oriented labeled corpus, not population prevalence.", "", f"Exclusions recorded: {len(exclusions)}. See `dataset_exclusions.csv`.", "", "Limitation: numeric processed graph filenames omit contract identity and preprocessing version metadata. Exact raw-to-graph correspondence and historical feature timestamps cannot be independently proven from the current legacy artifacts."]
    (output / "Dataset_Provenance_and_Label_Audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
