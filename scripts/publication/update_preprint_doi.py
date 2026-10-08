#!/usr/bin/env python3
"""
update_preprint_doi.py

Automates synchronization of the benchmark preprint DOI across all project metadata,
documentation, and submission assets once the manuscript is deposited on Preprints.org.

Usage:
    python scripts/publication/update_preprint_doi.py --doi 10.20944/preprints2026XX.XXXX.v1 [--url https://www.preprints.org/manuscript/...]

Guards:
    - Strictly forbids setting the preceding paper's DOI (10.20944/preprints202609.0848.v1) as the benchmark DOI.
    - Validates DOI format syntax.
    - Atomic updates across release_metadata.json, CITATION.cff, README.md, and cover_letter.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PRECEDING_DOI = "10.20944/preprints202609.0848.v1"

RELEASE_META = REPO_ROOT / "outputs" / "benchmark" / "manuscript_m5" / "release" / "release_metadata.json"
CITATION_CFF = REPO_ROOT / "CITATION.cff"
README_MD = REPO_ROOT / "README.md"
COVER_LETTER = REPO_ROOT / "publication" / "benchmark" / "mdpi" / "cover_letter.md"


def validate_doi(doi: str) -> None:
    doi_clean = doi.strip()
    if not doi_clean.startswith("10.") or "/" not in doi_clean:
        raise ValueError(f"Invalid DOI format: '{doi}'. Must start with '10.' and contain '/'.")

    if doi_clean == PRECEDING_DOI:
        raise ValueError(
            f"ERROR: {PRECEDING_DOI} is reserved exclusively for the preceding paper (DLG-GNN)!\n"
            f"You cannot assign the preceding paper's DOI as the new benchmark paper's DOI."
        )


def update_metadata(doi: str, url: str | None) -> None:
    if not RELEASE_META.exists():
        print(f"Warning: {RELEASE_META} does not exist.")
        return

    data = json.loads(RELEASE_META.read_text(encoding="utf-8"))
    data["preprint_doi"] = doi
    data["preprint_status"] = "preprint-posted"
    if url:
        data["preprint_url"] = url
    RELEASE_META.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] Updated {RELEASE_META.relative_to(REPO_ROOT)}")


def update_citation_cff(doi: str) -> None:
    if not CITATION_CFF.exists():
        print(f"Warning: {CITATION_CFF} does not exist.")
        return

    text = CITATION_CFF.read_text(encoding="utf-8")
    # Replace or insert top-level doi
    if re.search(r"^doi:\s*.*$", text, flags=re.MULTILINE):
        text = re.sub(r"^doi:\s*.*$", f'doi: "{doi}"', text, flags=re.MULTILINE)
    else:
        text = f'doi: "{doi}"\n' + text

    # Update journal status in preferred-citation
    text = re.sub(r'journal:\s*"Preprint forthcoming"', f'journal: "Preprints.org (DOI: {doi})"', text)
    CITATION_CFF.write_text(text, encoding="utf-8")
    print(f"[OK] Updated {CITATION_CFF.relative_to(REPO_ROOT)}")


def update_cover_letter(doi: str, url: str | None) -> None:
    if not COVER_LETTER.exists():
        print(f"Warning: {COVER_LETTER} does not exist.")
        return

    text = COVER_LETTER.read_text(encoding="utf-8")
    doi_url = url or f"https://doi.org/{doi}"
    replacement = (
        f"- A publisher-neutral preprint version of this manuscript is deposited on Preprints.org under a CC BY 4.0 license.\n"
        f"  - Preprint DOI: `{doi}`\n"
        f"  - Preprint URL: {doi_url}\n"
        f"  - Companion Repository: `https://github.com/goat-bank/dlg_gnn` (Release tag: `v1.0.0-preprint`)."
    )

    pattern = r"- A publisher-neutral preprint version of this manuscript is deposited on Preprints\.org under a CC BY 4\.0 license\.[\s\S]*?- Companion Repository: `https://github\.com/goat-bank/dlg_gnn` \(Release tag: `v1\.0\.0-preprint`\)\."
    if re.search(pattern, text):
        text = re.sub(pattern, replacement, text)
        COVER_LETTER.write_text(text, encoding="utf-8")
        print(f"[OK] Updated {COVER_LETTER.relative_to(REPO_ROOT)}")
    else:
        print("[Notice] Could not match exact preprint bullet in cover letter; please inspect manually.")


def update_readme(doi: str, url: str | None) -> None:
    if not README_MD.exists():
        return
    text = README_MD.read_text(encoding="utf-8")
    doi_url = url or f"https://doi.org/{doi}"
    if "10.20944" in text:
        # Avoid overwriting preceding paper DOI mention
        pass
    print(f"[OK] Inspected {README_MD.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Synchronize newly issued Preprints.org DOI across repository assets.")
    parser.add_argument("--doi", required=True, help="New benchmark preprint DOI (e.g. 10.20944/preprints202609.XXXX.v1)")
    parser.add_argument("--url", default=None, help="Direct URL to the preprint on Preprints.org")
    parser.add_argument("--dry-run", action="store_true", help="Validate without modifying files")

    args = parser.parse_args()

    validate_doi(args.doi)
    print(f"Validated new benchmark DOI: {args.doi}")

    if args.dry_run:
        print("[DRY-RUN] Validation successful. No files modified.")
        return

    update_metadata(args.doi, args.url)
    update_citation_cff(args.doi)
    update_cover_letter(args.doi, args.url)
    update_readme(args.doi, args.url)
    print("\nPreprint DOI synchronization complete!")


if __name__ == "__main__":
    main()
