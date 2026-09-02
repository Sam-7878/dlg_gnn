"""Record the successful Tectonic compile as the final manuscript gate."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from validation.sci_v3_final_common import atomic_json, sha256_file


def main() -> None:
    directory = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC")
    main_tex = directory / "DLG-StreamMC.tex"; appendix = directory / "DLG_StreamMC_Appendices_Reader_Guide_v1.tex"
    bib, pdf, log = directory / "references.bib", directory / "DLG-StreamMC.pdf", directory / "DLG-StreamMC.log"
    errors = []
    if not pdf.exists() or pdf.stat().st_size < 10_000: errors.append("compiled PDF missing or implausibly small")
    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    fatal_patterns = [r"LaTeX Error", r"Undefined control sequence", r"Emergency stop", r"Citation `[^']+' .* undefined",
                      r"Reference `[^']+' .* undefined", r"There were undefined references"]
    for pattern in fatal_patterns:
        if re.search(pattern, log_text, flags=re.IGNORECASE): errors.append(f"compile log matches: {pattern}")
    source = main_tex.read_text(encoding="utf-8") + "\n" + appendix.read_text(encoding="utf-8")
    cites = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source) for key in group.split(",")}
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
    if cites - bib_keys: errors.append(f"missing cite keys: {sorted(cites-bib_keys)}")
    overfull = [float(value) for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)]
    max_overfull = max(overfull, default=0.0)
    if max_overfull > 10.0: errors.append(f"material overfull box remains: {max_overfull:.3f}pt")
    page_count = None
    try:
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True, stderr=subprocess.STDOUT, timeout=20)
        match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE); page_count = int(match.group(1)) if match else None
    except Exception:
        pass
    binary = Path("/tmp/tectonic"); binary_sha = sha256_file(binary) if binary.exists() else None
    expected_binary_sha = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"
    if binary_sha != expected_binary_sha: errors.append("Tectonic binary hash differs from official release asset")
    report = {"status": "PASS" if not errors else "FAIL", "compiler": "Tectonic 0.17.0 x86_64-unknown-linux-musl",
              "compiler_sha256": binary_sha, "errors": errors, "pdf": str(pdf), "pdf_sha256": sha256_file(pdf) if pdf.exists() else None,
              "tex_sha256": sha256_file(main_tex), "bib_sha256": sha256_file(bib), "pages": page_count,
              "max_overfull_pt": max_overfull, "undefined_citations_or_references": False if not errors else None}
    root = Path("results/sci_v3_submission_r2/validation")
    atomic_json(root / "tectonic_compile_validation.json", report)
    structural_path = root / "manuscript_validation.json"
    structural = json.loads(structural_path.read_text(encoding="utf-8"))
    if not errors:
        structural["compile_status"] = "PASS_TECTONIC_0.17.0"
        structural["warnings"] = [warning for warning in structural.get("warnings", []) if "latexmk" not in warning]
        structural["compile_report"] = str(root / "tectonic_compile_validation.json")
        atomic_json(structural_path, structural)
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
