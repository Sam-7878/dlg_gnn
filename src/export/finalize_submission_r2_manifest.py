"""Freeze the final R2 claim and artifact manifests after manuscript compilation."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from validation.sci_v3_final_common import atomic_csv, atomic_json, sha256_file


ROOT = Path("results/sci_v3_submission_r2")
MANUSCRIPT = Path(
    "docs/work_reports/110_stream_mc_sci_v3_submission_r2/"
    "_41_01_DLG_StreamMC"
)


def main() -> None:
    claims_path = ROOT / "claim_manifest_v2.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))["claims"]
    claims = [item for item in claims if item["claim_id"] != "C-RUNTIME-REPEATED-POLICY"]
    claims.append({
        "claim_id": "C-RUNTIME-REPEATED-POLICY",
        "status": "SUPPORTED_LIMITED",
        "claim": (
            "warm-started validation-calibrated routing reduces measured mean latency "
            "relative to full-deep execution on the fixed 500-event prefix"
        ),
        "evidence": "runtime/five_repeat_policy_summary.csv",
        "boundary": (
            "five seeds times five repeats; all-benign runtime prefix; classification "
            "accuracy is not inferred from the timing population"
        ),
    })
    atomic_json(claims_path, {"version": 2, "claims": claims})
    atomic_csv(ROOT / "claim_manifest_v2.csv", pd.DataFrame(claims))

    include_roots = [
        ROOT / "cascade",
        ROOT / "runtime",
        ROOT / "statistics",
        ROOT / "reproducibility",
        ROOT / "validation",
        ROOT / "manuscript/tables",
        ROOT / "manuscript/figures",
    ]
    artifacts: list[dict[str, object]] = []
    for folder in include_roots:
        for path in sorted(item for item in folder.rglob("*") if item.is_file()):
            artifacts.append({
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    for path in (MANUSCRIPT / "DLG-StreamMC.tex", MANUSCRIPT / "references.bib", MANUSCRIPT / "DLG-StreamMC.pdf"):
        artifacts.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    atomic_json(ROOT / "manuscript/final_artifact_manifest.json", {
        "version": 1,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    })
    print(json.dumps({"claims": len(claims), "artifacts": len(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
