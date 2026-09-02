"""Fail-closed manuscript validator and compile gate for SCI-v3 R2."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from validation.sci_v3_final_common import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sci_v3_submission_r2/closure.yaml"))
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")); root = Path(cfg["output_root"])
    manuscript = Path(cfg["manuscript"]); directory = manuscript.parent; text = manuscript.read_text(encoding="utf-8")
    errors, warnings = [], []
    placeholders = re.findall(r"(?i)\b(?:placeholder|insert(?:ed)? directly|to be (?:added|inserted)|TBD|TODO)\b", text)
    if placeholders: errors.append(f"unresolved placeholder language: {sorted(set(placeholders))}")
    if "\\usepackage[hidelinks]{}" in text: errors.append("empty usepackage remains")
    if "elsarticle-num" in text: errors.append("non-IEEE bibliography style remains")
    if "XGBoost &\n  0.440766" in text or "0.026201" in text: errors.append("superseded tabular cascade numbers remain")
    packages = re.findall(r"\\usepackage(?:\[[^]]*\])?\{([^}]+)\}", text)
    duplicates = sorted({name for name in packages if packages.count(name) > 1})
    if duplicates: errors.append(f"duplicate packages: {duplicates}")
    labels = re.findall(r"\\label\{([^}]+)\}", text); refs = re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", text)
    duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
    if duplicate_labels: errors.append(f"duplicate labels: {duplicate_labels}")
    missing_refs = sorted(set(refs) - set(labels))
    if missing_refs: errors.append(f"unresolved labels: {missing_refs}")
    bib = directory / "references.bib"
    if not bib.exists(): errors.append("references.bib missing")
    else:
        bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
        cite_groups = re.findall(r"\\cite\{([^}]+)\}", text); cite_keys = {key.strip() for group in cite_groups for key in group.split(",")}
        missing_cites = sorted(cite_keys - bib_keys)
        if missing_cites: errors.append(f"missing bibliography keys: {missing_cites}")
    required = [root / "claim_manifest_v2.json", root / "statistics/claim_status.json", root / "cascade/leakage_audit.json",
                directory / "generated_r2/tables/table_calibrated_production_cascade.tex",
                directory / "generated_r2/figures/figure_streaming_memory.pdf"]
    errors.extend(f"missing artifact: {path}" for path in required if not path.exists())
    gate = json.loads((root / "cascade/acceptance_gate.json").read_text(encoding="utf-8"))
    if gate["gate"] == "FAIL-C" and "acceptance gate is therefore FAIL-C" not in text:
        errors.append("FAIL-C removal is not disclosed in manuscript")
    metrics = pd.read_csv(root / "cascade/calibrated_cascade_metrics.csv")
    fast = metrics[metrics.model == "ProductionLevel1GIN"].f1.mean(); cascade = metrics[metrics.model.str.contains("GATv2")].f1.mean()
    for value in (fast, cascade, cascade-fast):
        if f"{value:.3f}" not in text: errors.append(f"canonical manuscript value missing: {value:.3f}")

    compile_status, compile_output = "SKIPPED", ""
    if not args.skip_compile and not errors:
        latexmk = shutil.which("latexmk")
        if latexmk:
            result = subprocess.run([latexmk, "-pdf", "-interaction=nonstopmode", "-halt-on-error", manuscript.name],
                                    cwd=directory, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
            compile_status = "PASS" if result.returncode == 0 else "FAIL"
            compile_output = result.stdout[-12000:]
            if result.returncode: errors.append("latexmk compilation failed")
        else:
            compile_status = "UNAVAILABLE"
            warnings.append("latexmk is not installed in Ubuntu 24.04; structural validation completed")
    report = {"status": "PASS" if not errors else "FAIL", "compile_status": compile_status,
              "errors": errors, "warnings": warnings, "manuscript": str(manuscript), "compile_output_tail": compile_output}
    atomic_json(root / "validation/manuscript_validation.json", report)
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
