"""Render every PDF page to PNG for visual inspection.

A build that exits zero can still produce a table running off the page,
an unreadable figure label, or a heading stranded at the foot of a page.
Those are only findable by looking. This renders each page so they can
be inspected in CI artifacts or locally.

Also reports mechanical signals that correlate with layout problems:
page count, ink coverage per page (a near-empty page usually means a
float placement problem), and whether any page is entirely blank.

    python scripts/render_paper_pages.py [--dpi 150]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDF = PROJECT_ROOT / "paper" / "main.pdf"
OUT = PROJECT_ROOT / "outputs" / "paper_build" / "pages"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not PDF.exists():
        sys.exit(f"No PDF at {PDF}. Run scripts/build_paper.py first.")

    try:
        import fitz  # PyMuPDF
    except ImportError:
        sys.exit("PyMuPDF not installed. pip install pymupdf")

    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("page_*.png"):
        stale.unlink()

    document = fitz.open(PDF)
    page_total = document.page_count
    zoom = args.dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    print(f"Rendering {page_total} pages at {args.dpi} dpi\n")
    print(f"{'page':>5}  {'ink %':>7}  {'chars':>7}  note")

    problems: list[str] = []
    for number, page in enumerate(document, start=1):
        pixmap = page.get_pixmap(matrix=matrix)
        path = OUT / f"page_{number:02d}.png"
        pixmap.save(path)

        # Ink coverage: fraction of non-white pixels. Very low on a page
        # that is not the last one usually means a float pushed content
        # off and left a gap.
        samples = pixmap.samples
        stride = max(1, len(samples) // 30000)
        sampled = samples[::stride]
        non_white = sum(1 for value in sampled if value < 250)
        coverage = 100 * non_white / max(len(sampled), 1)
        text = page.get_text().strip()

        note = ""
        if not text and coverage < 1:
            note = "BLANK PAGE"
            problems.append(f"page {number} is blank")
        elif coverage < 2.0 and number < page_total:
            note = "sparse - check float placement"
            problems.append(f"page {number} is sparse ({coverage:.1f}% ink)")
        print(f"{number:>5}  {coverage:>6.1f}%  {len(text):>7}  {note}")

    document.close()
    print(f"\nWrote {page_total} images to "
          f"{OUT.relative_to(PROJECT_ROOT)}")

    if problems:
        print("\nLayout signals worth a look:")
        for problem in problems:
            print(f"  - {problem}")
    print("\nMechanical checks do not replace looking at the pages.")


if __name__ == "__main__":
    main()
