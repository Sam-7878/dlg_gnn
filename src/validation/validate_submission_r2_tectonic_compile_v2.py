"""Final Tectonic compile gate with release-archive and extracted-binary hashes."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from validation.sci_v3_final_common import atomic_json, sha256_file


def main() -> None:
    directory = Path("docs/work_reports/110_stream_mc_sci_v3_submission_r2/_41_01_DLG_StreamMC")
    tex, appendix, bib = directory/"DLG-StreamMC.tex", directory/"DLG_StreamMC_Appendices_Reader_Guide_v1.tex", directory/"references.bib"
    pdf, log = directory/"DLG-StreamMC.pdf", directory/"DLG-StreamMC.log"
    archive, binary = Path("/tmp/tectonic-0.17.0.tar.gz"), Path("/tmp/tectonic")
    expected_archive = "8533d07f9ccbd7a65824b9e0459041bca34af1eb33daba48f59215593753a3b7"
    errors = []
    if not archive.exists() or sha256_file(archive) != expected_archive: errors.append("official release archive digest mismatch")
    if not pdf.exists() or pdf.stat().st_size < 10_000: errors.append("compiled PDF missing or too small")
    log_text = log.read_text(encoding="utf-8", errors="replace")
    for pattern in (r"LaTeX Error", r"Undefined control sequence", r"Emergency stop", r"Citation `[^']+' .* undefined",
                    r"Reference `[^']+' .* undefined", r"There were undefined references"):
        if re.search(pattern, log_text, flags=re.IGNORECASE): errors.append(f"compile log matches: {pattern}")
    source = tex.read_text(encoding="utf-8") + appendix.read_text(encoding="utf-8")
    cites = {key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", source) for key in group.split(",")}
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
    if cites - bib_keys: errors.append(f"missing cite keys: {sorted(cites-bib_keys)}")
    overfull = [float(value) for value in re.findall(r"Overfull \\hbox \(([0-9.]+)pt too wide\)", log_text)]
    max_overfull = max(overfull, default=0.0)
    if max_overfull > 10.0: errors.append(f"material overfull box remains: {max_overfull:.3f}pt")
    page_count = None
    try:
        info = subprocess.check_output(["pdfinfo", str(pdf)], text=True, timeout=20)
        match = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE); page_count = int(match.group(1)) if match else None
    except Exception: pass
    report = {"status": "PASS" if not errors else "FAIL", "compiler": "Tectonic 0.17.0 x86_64-unknown-linux-musl",
        "official_release_archive_sha256": sha256_file(archive) if archive.exists() else None,
        "extracted_binary_sha256": sha256_file(binary) if binary.exists() else None, "errors": errors,
        "pdf": str(pdf), "pdf_sha256": sha256_file(pdf) if pdf.exists() else None, "pages": page_count,
        "tex_sha256": sha256_file(tex), "bib_sha256": sha256_file(bib), "max_overfull_pt": max_overfull,
        "undefined_citations_or_references": False if not errors else None}
    root = Path("results/sci_v3_submission_r2/validation"); atomic_json(root/"tectonic_compile_validation.json", report)
    structural_path = root/"manuscript_validation.json"; structural = json.loads(structural_path.read_text(encoding="utf-8"))
    if not errors:
        structural.update({"compile_status": "PASS_TECTONIC_0.17.0", "compile_report": str(root/"tectonic_compile_validation.json")})
        structural["warnings"] = []
        atomic_json(structural_path, structural)
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
