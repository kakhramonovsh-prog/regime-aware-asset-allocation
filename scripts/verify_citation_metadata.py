"""Verify bibliography metadata against publisher records via Crossref.

Every DOI is registered with Crossref by its publisher, so the Crossref
record is the publisher's own deposited metadata rather than a
third-party transcription. This script resolves each DOI in
``paper/references.bib`` and compares author surnames, title, container,
volume, issue, pages and year field by field.

It verifies metadata only. Whether a cited *claim* is actually supported
by the article is a separate judgement recorded in
``paper/citation_audit.csv``; no automated check can make it.

    python scripts/verify_citation_metadata.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BIB = PROJECT_ROOT / "paper" / "references.bib"

# Crossref asks for a contact address so they can reach you about
# excessive polling; it also routes the request to their faster pool.
# Read from the environment rather than hard-coded: this file is public,
# and a literal address here is scraped.
MAILTO = os.environ.get("CROSSREF_MAILTO", "")
USER_AGENT = (f"regime-aware-asset-allocation/1.0 (mailto:{MAILTO})" if MAILTO
              else "regime-aware-asset-allocation/1.0")


def parse_bib(text: str) -> dict[str, dict[str, str]]:
    """Parse the subset of BibTeX this repository writes."""
    entries: dict[str, dict[str, str]] = {}
    for block in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.DOTALL):
        kind, key, body = block.groups()
        fields: dict[str, str] = {"__type__": kind.lower()}
        # The final field carries no trailing comma or newline, so the
        # body must be terminated before matching or the last field
        # (usually the DOI) is silently dropped.
        for name, value in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*\n",
                                      body + "\n", re.DOTALL):
            collapsed = " ".join(value.split())
            fields[name.lower()] = collapsed
        entries[key.strip()] = fields
    return entries


def crossref(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())["message"]
    except urllib.error.HTTPError as error:
        print(f"    HTTP {error.code}")
    except Exception as error:  # noqa: BLE001 - report and continue
        print(f"    {type(error).__name__}: {error}")
    return None


# Differences adjudicated during the citation pass of 2026-08-16 against
# a second and sometimes third source. Each entry records WHY the .bib
# is right and Crossref's field is not simply copied. Anything not
# listed here is a new discrepancy and fails the check.
ADJUDICATED: dict[str, dict[str, str]] = {
    "ang2002": {"container": "Journal's formal name carries 'The'; Crossref drops the article"},
    "demiguel2009": {
        "container": "as ang2002",
        "title": "Crossref embeds HTML italics around N",
        "year": "2009 is the RFS 22(5) print issue; Crossref records 2007 advance access. Semantic Scholar gives 2009",
    },
    "harvey2016": {
        "container": "as ang2002",
        "year": "2016 is the RFS 29(1) print issue; Crossref records 2015 advance access. OpenAlex gives 2016",
    },
    "diebold1995": {"container": "Crossref returns the HTML entity &amp; for the ampersand"},
    "dempster1977": {
        "title": "Crossref embeds HTML italics around EM",
        "container": "journal renamed since 1977; the .bib uses the name at publication",
    },
    "bailey2014": {
        "pages": "Crossref carries the start page only; 458-471 confirmed from the open-access AMS PDF (14 pages from 458)",
        "authors": "accent folding in the comparison, not a name difference",
    },
    "engle1982": {"pages": "Crossref start page only; 987-1007 confirmed via Semantic Scholar"},
    "hamilton1989": {"pages": "Crossref start page only; 357-384 confirmed via Semantic Scholar"},
    "newey1987": {"pages": "Crossref start page only; 703-708 confirmed via RePEc and the Econometric Society"},
    "holm1979": {"__unresolved__": "no DOI: 10.2307/4615733 appears in OpenAlex but does not resolve. Verified via OpenAlex and JSTOR stable 4615733"},
}


def adjudicated(key: str, issue: str) -> str | None:
    """Return the recorded reason if this exact field was adjudicated."""
    field = issue.split(":", 1)[0].strip()
    return ADJUDICATED.get(key, {}).get(field)


def norm(value: str) -> str:
    """Compare on letters and digits only."""
    value = re.sub(r"\\[a-zA-Z]+\s*", "", value)          # LaTeX commands
    value = value.replace("{", "").replace("}", "")
    value = value.replace("--", "-").replace("\u2013", "-")
    return re.sub(r"[^a-z0-9]", "", value.lower())


def surnames(authors: str) -> list[str]:
    people = [a.strip() for a in re.split(r"\band\b", authors)]
    out = []
    for person in people:
        surname = person.split(",")[0] if "," in person else person.split()[-1]
        out.append(norm(surname))
    return out


def compare(key: str, fields: dict[str, str], record: dict) -> list[str]:
    """Field-by-field comparison; returns human-readable mismatches."""
    issues: list[str] = []

    published = record.get("title") or [""]
    if norm(fields.get("title", "")) != norm(published[0]):
        issues.append(f"title: bib={fields.get('title')!r} "
                      f"crossref={published[0]!r}")

    bib_names = surnames(fields.get("author", ""))
    cr_names = [norm(a.get("family", "")) for a in record.get("author", [])]
    if bib_names != cr_names:
        issues.append(f"authors: bib={bib_names} crossref={cr_names}")

    container = (record.get("container-title") or [""])[0]
    bib_container = fields.get("journal") or fields.get("booktitle") or ""
    if norm(bib_container) != norm(container):
        issues.append(f"container: bib={bib_container!r} crossref={container!r}")

    for bib_field, cr_field in (("volume", "volume"), ("number", "issue")):
        bib_value = fields.get(bib_field, "")
        cr_value = str(record.get(cr_field, ""))
        if bib_value and cr_value and norm(bib_value) != norm(cr_value):
            issues.append(f"{bib_field}: bib={bib_value!r} crossref={cr_value!r}")

    bib_pages, cr_pages = fields.get("pages", ""), str(record.get("page", ""))
    if bib_pages and cr_pages and norm(bib_pages) != norm(cr_pages):
        issues.append(f"pages: bib={bib_pages!r} crossref={cr_pages!r}")

    bib_year = fields.get("year", "")
    parts = (record.get("issued", {}).get("date-parts") or [[None]])[0]
    cr_year = str(parts[0]) if parts and parts[0] else ""
    if bib_year and cr_year and bib_year != cr_year:
        issues.append(f"year: bib={bib_year!r} crossref={cr_year!r}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="write the full report")
    args = parser.parse_args()

    entries = parse_bib(BIB.read_text(encoding="utf-8"))
    print(f"Parsed {len(entries)} bibliography entries\n")

    report: dict[str, dict] = {}
    clean, flagged, unresolved = [], [], []

    for key in sorted(entries):
        fields = entries[key]
        doi = fields.get("doi", "")
        print(f"{key}  doi={doi or '(none)'}")
        if not doi:
            if adjudicated(key, "__unresolved__"):
                clean.append(key)
                report[key] = {"status": "adjudicated (no doi)", "issues": []}
                print(f"    no DOI, adjudicated: "
                      f"{adjudicated(key, '__unresolved__')}\n")
            else:
                unresolved.append(key)
                report[key] = {"status": "no doi", "issues": []}
                print("    NO DOI - verify manually against the publisher page\n")
            continue

        record = crossref(doi)
        time.sleep(0.4)          # be polite to a free public API
        if record is None:
            unresolved.append(key)
            report[key] = {"status": "unresolved", "issues": []}
            print("    UNRESOLVED\n")
            continue

        issues = compare(key, fields, record)
        report[key] = {
            "status": "mismatch" if issues else "match",
            "issues": issues,
            "crossref": {
                "title": (record.get("title") or [""])[0],
                "container": (record.get("container-title") or [""])[0],
                "volume": record.get("volume"),
                "issue": record.get("issue"),
                "page": record.get("page"),
                "year": ((record.get("issued", {}).get("date-parts")
                          or [[None]])[0] or [None])[0],
                "publisher": record.get("publisher"),
                "type": record.get("type"),
            },
        }
        unexplained = []
        for issue in issues:
            reason = adjudicated(key, issue)
            if reason:
                print(f"    adjudicated {issue.split(':', 1)[0]}: {reason}")
            else:
                unexplained.append(issue)
                print(f"    MISMATCH {issue}")
        report[key]["status"] = "mismatch" if unexplained else "match"
        report[key]["unexplained"] = unexplained
        if unexplained:
            flagged.append(key)
        else:
            clean.append(key)
            if not issues:
                print("    match")
        print()

    print("=" * 70)
    print(f"match: {len(clean)}   mismatch: {len(flagged)}   "
          f"unresolved: {len(unresolved)}")
    if flagged:
        print(f"mismatched: {flagged}")
    if unresolved:
        print(f"unresolved: {unresolved}")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n",
                             encoding="utf-8")
        print(f"\nWrote {args.json}")

    sys.exit(1 if flagged or unresolved else 0)


if __name__ == "__main__":
    main()
