#!/usr/bin/env python3
"""Gate A23-G2: Automated prohibited-phrase scan across active manuscript TeX, CSV, and Markdown files."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PUB_A22 = ROOT / "projects" / "benchmark" / "publication" / "a22"
PUB_A23 = ROOT / "projects" / "benchmark" / "publication" / "a23"
SUB_PREPRINTS = ROOT / "projects" / "benchmark" / "submission" / "preprints"
SUB_APPLSCI = ROOT / "projects" / "benchmark" / "submission" / "applsci"

PROHIBITED_PHRASES = [
    "proves capacity",
    "driven by regularized",
    "stem from regularized",
    "stem from high-capacity",
    "alignment disproven",
    "proves ap gain is not driven by",
    "proves training epoch budget alone",
    "confirms that performance gains on this network stem from",
]


def main() -> None:
    targets = [
        SUB_PREPRINTS / "DLG-Benchmark_A16_Preprints.tex",
        SUB_APPLSCI / "DLG-Benchmark_A16_AppliedSciences.tex",
        PUB_A23 / "ablation_provenance_noncausal.csv",
        PUB_A23 / "ablation_provenance_noncausal.md",
        PUB_A23 / "dataset_graph_identity_canonical.csv",
        PUB_A23 / "dataset_graph_identity_canonical.md",
        PUB_A23 / "s3_label_provenance_gate.csv",
        PUB_A23 / "gadnr_failure_evidence_note.md",
        PUB_A23 / "g1_exact_loss_canonical_trace.md",
        PUB_A22 / "a22_ablation_provenance_audit.csv",
        PUB_A22 / "a22_mechanistic_ablation_note.md",
    ]

    matches = []
    scanned_files = []
    for p in targets:
        if not p.exists():
            continue
        rel = p.relative_to(ROOT).as_posix()
        scanned_files.append(rel)
        text_lower = p.read_text(encoding="utf-8").lower()
        for phrase in PROHIBITED_PHRASES:
            if phrase in text_lower:
                matches.append({"file": rel, "prohibited_phrase": phrase})

    result = {
        "gate": "A23-G2",
        "status": "PASS" if len(matches) == 0 else "FAIL",
        "scanned_files_count": len(scanned_files),
        "scanned_files": scanned_files,
        "prohibited_phrases_checked": PROHIBITED_PHRASES,
        "matches_found_count": len(matches),
        "matches": matches,
    }

    out_path = PUB_A23 / "a23_g2_prohibited_phrase_scan.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    assert len(matches) == 0, f"Found prohibited phrases: {matches}"
    print(f"A23-G2 prohibited phrase scan PASSED (0 matches across {len(scanned_files)} files).")


if __name__ == "__main__":
    main()

