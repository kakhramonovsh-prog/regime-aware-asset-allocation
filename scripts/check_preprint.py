"""Certify the manuscript as ready for preprint or journal upload.

Separate from the build gate. A manuscript can compile cleanly and still
be unfit to upload: missing an ORCID, missing the declarations a
publisher requires, or still carrying an unqualified preregistration
claim. Those failures are invisible to latexmk and expensive to discover
after a DOI has been minted, because a DOI is permanent even if the
paper is later withdrawn.

    python scripts/check_preprint.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"
SOURCES = ("main.tex", "_sections_intro.tex", "_sections_results.tex")

# Every declaration a Springer Nature submission must carry.
REQUIRED_DECLARATIONS = {
    "Funding": r"\\paragraph\{Funding\.\}",
    "Competing interests": r"\\paragraph\{Competing interests\.\}",
    "Data and code availability": r"\\paragraph\{Data and code availability\.\}",
    "Author contributions": r"\\paragraph\{Author contributions\.\}",
    "AI-assisted tools": r"\\paragraph\{Declaration of AI-assisted tools\.\}",
}


def main() -> None:
    problems: list[str] = []
    text = "".join((PAPER / name).read_text(encoding="utf-8") for name in SOURCES)

    config = yaml.safe_load(
        (PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    manuscript = config.get("manuscript", {})

    print("Author identifiers")
    for field in ("orcid", "corresponding_email"):
        value = str(manuscript.get(field, "PENDING")).strip()
        if not value or value.upper() == "PENDING":
            print(f"  [BLOCK] {field} is unset")
            problems.append(
                f"{field} is still PENDING; register it and set "
                f"config/config.yaml manuscript.{field}"
            )
        else:
            print(f"  [ok] {field}: {value}")

    print("\nRequired declarations")
    for label, pattern in REQUIRED_DECLARATIONS.items():
        if re.search(pattern, text):
            print(f"  [ok] {label}")
        else:
            print(f"  [BLOCK] {label} missing")
            problems.append(f"declaration missing: {label}")

    # An unqualified "preregistered" claim implies a public registry.
    # This plan was time-stamped privately, so the word is not accurate
    # without the qualification that accompanies it in the design section.
    print("\nRegistration claim")
    stray = []
    for name in SOURCES:
        for i, line in enumerate((PAPER / name).read_text(encoding="utf-8").splitlines(), 1):
            # Bare "registered"/"registration" makes the same claim
            # without the prefix. The abstract read "were registered
            # before estimation", which a prefix-only scan missed.
            if re.search(r"preregist|\bwere registered\b|\bwas registered\b"
                         r"|\bregistration\b|\bregistry\b", line, re.I):
                if "described as prospectively specified rather" in line or \
                   "than preregistered" in line:
                    continue
                stray.append(f"{name}:{i}: {line.strip()[:70]}")
    if stray:
        for item in stray:
            print(f"  [BLOCK] {item}")
        problems.extend(f"unqualified preregistration claim at {s}" for s in stray)
    else:
        print("  [ok] no unqualified preregistration claim")

    print("\nProspective-specification disclosure")
    # The sentence wraps across source lines, so compare on normalized
    # whitespace rather than the raw text.
    flat = " ".join(text.split())
    if "time-stamped in a private version-controlled repository" in flat:
        print("  [ok] disclosure present")
    else:
        problems.append("prospective-specification disclosure missing")
        print("  [BLOCK] disclosure missing")

    print("\n" + "=" * 60)
    if problems:
        print("NOT READY FOR UPLOAD:")
        for item in problems:
            print(f"  - {item}")
        sys.exit(1)
    print("PREPRINT CHECK PASSED: identifiers set, declarations present, "
          "registration claim accurate.")


if __name__ == "__main__":
    main()
