#!/usr/bin/env python3
"""
inspect_sections_and_claims.py

Inspects DLG-Benchmark.tex for:
1. Section/subsection hierarchy and roadmap alignment
2. Section 5.7 LANL text and metrics
3. Overclaiming words ("proves", "explains why", "causes", "oversmoothing explains", "overfitting")
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEX_PATH = REPO_ROOT / "docs" / "papers" / "_42_Benchmark" / "DLG-Benchmark.tex"

def main():
    text = TEX_PATH.read_text(encoding="utf-8")
    lines = text.splitlines()

    print("=== Section / Subsection Hierarchy ===")
    for i, line in enumerate(lines, 1):
        m = re.match(r'^\s*\\(section|subsection|subsubsection)\*?\{([^}]+)\}', line)
        if m:
            level = m.group(1)
            title = m.group(2)
            indent = "  " if level == "subsection" else ("    " if level == "subsubsection" else "")
            print(f"Line {i:4d}: {indent}{level.upper()}: {title}")

    print("\n=== Overclaiming Words Audit ===")
    patterns = [
        (r'\bproves\b', "proves"),
        (r'\bproof\b', "proof"),
        (r'\bexplains why\b', "explains why"),
        (r'\bcauses\b', "causes"),
        (r'\boversmoothing explains\b', "oversmoothing explains"),
        (r'\boverfitting\b', "overfitting"),
    ]

    for pat, label in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        print(f"Pattern '{label}': {len(matches)} occurrences")
        for m in matches[:5]:
            start = max(0, m.start() - 60)
            end = min(len(text), m.end() + 60)
            snippet = text[start:end].replace('\n', ' ')
            # find line number
            line_no = text[:m.start()].count('\n') + 1
            print(f"  Line {line_no}: ...{snippet}...")

if __name__ == "__main__":
    main()
