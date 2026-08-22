"""Extract text and render selected pages from a PDF for provenance audit."""
from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf
from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--text-output", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path)
    parser.add_argument("--pages", type=int, nargs="*", default=[])
    parser.add_argument("--search", action="append", default=[])
    args = parser.parse_args()

    reader = PdfReader(args.pdf)
    text = "\n\f\n".join(page.extract_text() or "" for page in reader.pages)
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_text(text, encoding="utf-8")
    print(f"pages={len(reader.pages)} chars={len(text)}")
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for pattern in args.search:
            if pattern.lower() in page_text.lower():
                print(f"match page={page_number} pattern={pattern}")

    if args.render_dir and args.pages:
        args.render_dir.mkdir(parents=True, exist_ok=True)
        document = pymupdf.open(args.pdf)
        for page_number in args.pages:
            page = document[page_number - 1]
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
            target = args.render_dir / f"page-{page_number:03d}.png"
            pixmap.save(target)
            print(target)


if __name__ == "__main__":
    main()
