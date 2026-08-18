"""Build the paper and enforce the acceptance gate.

One command. Regenerates macros and tables from stored results, compiles
with latexmk under halt-on-error, resolves the bibliography and
cross-references, and **fails** on any undefined citation, undefined
reference, undefined control sequence, or overfull box beyond a stated
threshold. Saves the full log either way.

    python scripts/build_paper.py [--skip-regen] [--overfull-limit PT]

Exit codes: 0 clean build, 1 build or gate failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"
LOG_DIR = PROJECT_ROOT / "outputs" / "paper_build"

# A few overfull points are invisible; anything larger is a real
# layout problem worth failing on.
DEFAULT_OVERFULL_LIMIT_PT = 5.0

FATAL_PATTERNS = [
    (r"Undefined control sequence", "undefined control sequence"),
    (r"LaTeX Error", "LaTeX error"),
    (r"Emergency stop", "emergency stop"),
    (r"! Missing", "missing delimiter or argument"),
]
WARNING_PATTERNS = [
    (r"Citation `([^']+)' on page \d+ undefined", "undefined citation"),
    (r"Reference `([^']+)' on page \d+ undefined", "undefined reference"),
    (r"There were undefined references", "undefined references"),
    (r"Label\(s\) may have changed", "labels changed; rerun needed"),
]


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, capture_output=True, text=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-regen", action="store_true",
                        help="skip regenerating macros/tables (CI without data)")
    parser.add_argument("--overfull-limit", type=float,
                        default=DEFAULT_OVERFULL_LIMIT_PT)
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    # ------------------------------------------------------------------
    # 1. Regenerate every number from stored results
    # ------------------------------------------------------------------
    if args.skip_regen:
        print("Skipping regeneration (--skip-regen).")
    else:
        print("Regenerating macros and tables from stored results...")
        result = run([sys.executable, "scripts/generate_paper_outputs.py"], PROJECT_ROOT)
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])
            sys.exit("Failed to regenerate paper outputs.")
        print("  macros.tex and tables regenerated")

    # ------------------------------------------------------------------
    # 2. Validate source before invoking TeX
    # ------------------------------------------------------------------
    print("\nValidating LaTeX source...")
    result = run([sys.executable, "scripts/check_paper.py"], PROJECT_ROOT)
    print("  " + result.stdout.strip().splitlines()[-1])
    if result.returncode != 0:
        print(result.stdout)
        sys.exit("Source validation failed.")

    # ------------------------------------------------------------------
    # 2b. Every number in the prose is classified, and no empirical
    #     result is hand-typed. A number that reappears as a literal
    #     after an edit must fail here, not at the next manual read.
    # ------------------------------------------------------------------
    print("\nAuditing numeric literals...")
    result = run([sys.executable, "scripts/number_census.py"], PROJECT_ROOT)
    print("  " + result.stdout.strip().splitlines()[-1])
    if result.returncode != 0:
        print(result.stdout)
        sys.exit("Number census failed.")

    # ------------------------------------------------------------------
    # 3. Compile
    # ------------------------------------------------------------------
    # pdfTeX stamps /CreationDate and /ModDate into the PDF trailer from
    # the wall clock, so two builds of identical source differ in bytes
    # and therefore in hash. SOURCE_DATE_EPOCH pins those; FORCE_SOURCE_DATE
    # makes pdfTeX honour it for \pdfcreationdate as well. Both are set
    # from the frozen manuscript release date so the value is derived,
    # not another number to keep in sync by hand.
    import yaml  # noqa: PLC0415

    config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    release = dt.date.fromisoformat(str(config["manuscript"]["release_date"]))
    epoch = int(dt.datetime(release.year, release.month, release.day,
                            tzinfo=dt.timezone.utc).timestamp())
    os.environ["SOURCE_DATE_EPOCH"] = str(epoch)
    os.environ["FORCE_SOURCE_DATE"] = "1"
    print(f"\nSOURCE_DATE_EPOCH={epoch} ({release.isoformat()}), "
          f"FORCE_SOURCE_DATE=1")

    latexmk = shutil.which("latexmk")
    if latexmk is None:
        sys.exit(
            "latexmk not found. Install TeX Live or TinyTeX, or run this in CI.\n"
            "The PDF cannot be built or inspected without it."
        )

    # Clean auxiliary files first. A stale .bbl, .aux or PDF can mask a
    # failure completely: the previous build produced a PDF with
    # unresolved references because a broken .bib left no .bbl and the
    # old artifacts filled the gap.
    print("\nCleaning auxiliary files...")
    run([latexmk, "-C"], PAPER)
    for pattern in ("*.aux", "*.bbl", "*.blg", "*.out", "*.toc", "*.fls",
                    "*.fdb_latexmk", "main.pdf"):
        for stale in PAPER.glob(pattern):
            stale.unlink()
            print(f"  removed {stale.name}")

    print("\nCompiling with latexmk (halt-on-error)...")
    result = run(
        [latexmk, "-pdf", "-halt-on-error", "-interaction=nonstopmode",
         "-file-line-error", "main.tex"],
        PAPER,
    )
    log_path = PAPER / "main.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    saved_log = LOG_DIR / "main.log"
    saved_log.write_text(log_text, encoding="utf-8")
    (LOG_DIR / "latexmk_stdout.txt").write_text(
        result.stdout + "\n" + result.stderr, encoding="utf-8"
    )
    print(f"  log saved to {saved_log.relative_to(PROJECT_ROOT)}")

    if result.returncode != 0:
        print(result.stdout[-3000:])
        failures.append("latexmk exited non-zero")

    # ------------------------------------------------------------------
    # 4. Gate on the log
    # ------------------------------------------------------------------
    print("\nChecking the log...")
    for pattern, label in FATAL_PATTERNS:
        hits = re.findall(pattern, log_text)
        if hits:
            failures.append(f"{label}: {len(hits)} occurrence(s)")
    for pattern, label in WARNING_PATTERNS:
        hits = re.findall(pattern, log_text)
        if hits:
            unique = sorted(set(hits)) if hits and isinstance(hits[0], str) else []
            detail = f" ({', '.join(unique[:5])})" if unique and unique[0] else ""
            failures.append(f"{label}: {len(hits)}{detail}")

    overfull = [
        (float(pts), line)
        for pts, line in re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too wide[^\n]*", log_text)
        or []
    ] if False else []
    overfull_raw = re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide\)", log_text)
    bad_boxes = [float(pt) for pt in overfull_raw if float(pt) > args.overfull_limit]
    print(f"  overfull hboxes: {len(overfull_raw)} total, "
          f"{len(bad_boxes)} over {args.overfull_limit}pt")
    if bad_boxes:
        failures.append(
            f"{len(bad_boxes)} overfull box(es) exceed {args.overfull_limit}pt "
            f"(worst {max(bad_boxes):.1f}pt)"
        )

    # ------------------------------------------------------------------
    # 4b. BibTeX must actually have run and produced a bibliography
    # ------------------------------------------------------------------
    print("\nChecking the bibliography...")
    source_text = "".join(
        (PAPER / name).read_text(encoding="utf-8")
        for name in ("main.tex", "_sections_intro.tex", "_sections_results.tex")
    )
    has_citations = bool(re.search(r"\\cite[tp]?\{", source_text))
    bbl = PAPER / "main.bbl"
    blg = PAPER / "main.blg"

    if has_citations:
        if not bbl.exists() or bbl.stat().st_size == 0:
            failures.append(
                "manuscript contains citations but main.bbl is missing or "
                "empty: BibTeX did not produce a bibliography"
            )
            print("  main.bbl: MISSING OR EMPTY")
        else:
            print(f"  main.bbl: {bbl.stat().st_size} bytes")

    if blg.exists():
        blg_text = blg.read_text(encoding="utf-8", errors="replace")
        shutil.copy2(blg, LOG_DIR / "main.blg")
        bib_errors = re.findall(r"^(I found no|Warning--|.*---line \d+ of file)",
                                blg_text, re.MULTILINE)
        hard_errors = [line for line in blg_text.splitlines()
                       if "error" in line.lower() or line.startswith("I found no")]
        print(f"  main.blg: {len(bib_errors)} warning(s), "
              f"{len(hard_errors)} error line(s)")
        if hard_errors:
            for line in hard_errors[:5]:
                print(f"    {line.strip()}")
            failures.append(f"BibTeX reported {len(hard_errors)} error(s)")
    elif has_citations:
        failures.append("no main.blg: BibTeX never ran")

    pdf = PAPER / "main.pdf"
    if pdf.exists():
        size_kb = pdf.stat().st_size / 1024
        print(f"  PDF produced: {size_kb:.0f} KB")
        shutil.copy2(pdf, LOG_DIR / "main.pdf")

        # ------------------------------------------------------------------
        # 4c. The rendered PDF must contain no unresolved markers. This is
        # what caught "Equation (??)" after the log gate passed it.
        # ------------------------------------------------------------------
        try:
            import pymupdf

            document = pymupdf.open(pdf)
            text = "".join(page.get_text() for page in document)
            document.close()
            unresolved = re.findall(r"\(\?\?\)|\[\?\?\]|\?\?(?=[\s.,;)])", text)
            print(f"  unresolved '??' markers in the PDF text: {len(unresolved)}")
            if unresolved:
                failures.append(
                    f"{len(unresolved)} unresolved reference marker(s) rendered "
                    "in the PDF"
                )
            missing_citation = re.findall(r"\[\?\]", text)
            if missing_citation:
                failures.append(
                    f"{len(missing_citation)} unresolved citation marker(s) '[?]'"
                )
        except ImportError:
            print("  (pymupdf not installed; skipping PDF text scan)")
    else:
        failures.append("no PDF produced")

    # ------------------------------------------------------------------
    print()
    if failures:
        print("BUILD GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("BUILD GATE PASSED: PDF built with no undefined citations,")
    print("references, control sequences, or significant overfull boxes.")


if __name__ == "__main__":
    main()
