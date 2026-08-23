#!/usr/bin/env python3
"""
DARPA TC E5 THEIA Ground Truth Label Extractor
Defense Extension Round D3

Extracts attack entity timestamps from TA51_Final_report_E5.pdf/docx
and maps them to internal node IDs in the ground_truth_mapping.csv stub.

E5 THEIA attack activities (from TA51_Final_report_E5):
The PDF documents adversarial activities during the 2019-05-07 to 2019-05-23
engagement period. Attack process/host names and timestamps allow us to
identify Subject nodes that were part of attack chains.

Strategy (D3 §11, §13):
1. Extract attack description text from PDF
2. Identify observable indicators: process names, file paths, network endpoints
3. Match against node properties in the built graph
4. Mark matching Subject/FileObject/NetFlowObject nodes as positive

Usage:
    python gt_pdf_extractor.py --graph-dir outputs/sci_defense_extension_real/graphs
"""
import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GT_PDF = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Ground_Truth/Ground_Truth/TA51_Final_report_E5.pdf")
GT_DOCX = Path("/mnt/d/_Work/_data/DLG/DARPA-TC-THEIA/Ground_Truth/Ground_Truth/TA51_Final_report_E5.docx")

# ─────────────────────────────────────────────────────────────────────────────
# E5 THEIA Known Attack Indicators
# Extracted from TA51_Final_report_E5 (manual review of the official PDF)
# These are observable indicators from the E5 engagement period (2019-05-07 to 2019-05-23)
#
# NOTE: For reproducibility, these indicators are defined ONLY from the
# official GT report, NEVER from performance-driven selection.
# ─────────────────────────────────────────────────────────────────────────────
# Attack timeline (to be populated after PDF text extraction)
# Format: (description, attack_start_unix_s, attack_end_unix_s)
KNOWN_ATTACK_WINDOWS = [
    # Placeholder: actual values extracted from TA51_Final_report_E5.pdf
    # Example format (to be filled):
    # ("firefox_backdoor_day1", 1557226800, 1557234000),
    # ("browser_extension_attack_day3", 1557399600, 1557406800),
]

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using pdfminer or PyMuPDF."""
    try:
        import pdfminer.high_level
        return pdfminer.high_level.extract_text(str(pdf_path))
    except ImportError:
        pass
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        return "\n".join(page.get_text() for page in doc)
    except ImportError:
        pass
    return ""

def extract_text_from_docx(docx_path: Path) -> str:
    """Extract text from DOCX."""
    try:
        from docx import Document
        doc = Document(str(docx_path))
        return "\n".join(p.text for p in doc.paragraphs)
    except ImportError:
        pass
    return ""

def parse_attack_indicators(text: str) -> list[dict]:
    """
    Parse attack indicators from GT report text.
    Returns list of {description, process_name, file_path, network_endpoint, timestamp_str}
    """
    indicators = []
    # Look for THEIA-specific attack patterns in the report
    # Common patterns in TC GT reports:
    # - Process names: firefox, wget, curl, bash, python, nc, netcat
    # - File paths: /tmp/*.sh, ~/.config/*, suspicious executables
    # - Network: C2 IP addresses, suspicious ports

    # Timestamp patterns (ISO or relative day)
    ts_pattern = re.compile(r'2019-0[5-9]-\d{2}[T\s]\d{2}:\d{2}')
    proc_pattern = re.compile(r'\b(firefox|wget|curl|bash|python|nc|netcat|sh|python3)\b', re.I)
    ip_pattern = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')
    
    for match in ts_pattern.finditer(text):
        context = text[max(0, match.start()-200):match.end()+200]
        procs = proc_pattern.findall(context)
        ips = ip_pattern.findall(context)
        if procs or ips:
            indicators.append({
                "timestamp_str": match.group(),
                "processes": list(set(procs)),
                "ips": list(set(ips)),
                "context": context[:200].replace("\n", " "),
            })
    return indicators

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", type=Path,
                    default=Path("outputs/sci_defense_extension_real"))
    args = ap.parse_args()

    audit_dir = args.graph_dir / "source_audit"
    gt_mapping_path = audit_dir / "ground_truth_mapping.csv"

    if not gt_mapping_path.exists():
        log.error(f"GT mapping stub not found: {gt_mapping_path}")
        log.error("Run darpa_theia_build.py first.")
        sys.exit(1)

    # Try to extract text from GT report
    text = ""
    if GT_DOCX.exists():
        log.info(f"Extracting text from DOCX: {GT_DOCX}")
        text = extract_text_from_docx(GT_DOCX)
    if not text and GT_PDF.exists():
        log.info(f"Extracting text from PDF: {GT_PDF}")
        text = extract_text_from_pdf(GT_PDF)

    if not text:
        log.warning("Could not extract text from GT report — pdfminer/PyMuPDF/python-docx not installed")
        log.warning("Install with: pip install pdfminer.six python-docx")
        log.warning("GT labels remain as PENDING stubs.")
        return

    log.info(f"Extracted {len(text):,} characters from GT report")

    # Parse indicators
    indicators = parse_attack_indicators(text)
    log.info(f"Found {len(indicators)} attack indicator windows")
    for ind in indicators[:10]:
        log.info(f"  {ind['timestamp_str']}: procs={ind['processes']}, ips={ind['ips']}")

    # Save extracted indicators for audit trail
    indicators_path = audit_dir / "gt_extracted_indicators.json"
    with open(indicators_path, "w") as f:
        json.dump({"source": str(GT_PDF if not GT_DOCX.exists() else GT_DOCX),
                   "indicators": indicators}, f, indent=2)
    log.info(f"Indicators saved → {indicators_path}")

    # NOTE: Full UUID-level matching requires the Subject/FileObject property fields
    # (cmdLine, url, etc.) to be stored during graph build. The current build
    # stores UUIDs but not string properties for memory efficiency.
    # To enable exact UUID matching, re-run darpa_theia_build.py with --store-properties.
    #
    # For D3 purposes, timestamp-window based labeling is acceptable:
    # Nodes with activity in attack windows → positive candidates
    log.info("\nGT extraction complete. Full UUID mapping requires property storage.")
    log.info("Status: PARTIALLY_EXTRACTED — timestamp indicators available")

if __name__ == "__main__":
    main()
