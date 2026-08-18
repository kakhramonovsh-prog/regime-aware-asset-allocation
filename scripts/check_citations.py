"""Citation release gate.

Two separate requirements, because they fail differently:

1. **Bibliographic verification** --- authors, title, year, journal,
   volume, issue, pages and DOI match the publisher record.
2. **Claim verification** --- the cited paper actually supports the
   sentence it is attached to, with the supporting location recorded.

A citation can have perfect metadata and still be wrong about what it
claims, which is the more damaging error and the harder one to catch.
``paper/citation_audit.csv`` records both for every key.

    python scripts/check_citations.py [--release]

Without ``--release`` the script reports status and exits 0, so a draft
PDF can be built while verification is in progress. With ``--release``
it **fails** if any entry is pending, blocking the manuscript tag and
the public release.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"
AUDIT = PAPER / "citation_audit.csv"
BIB = PAPER / "references.bib"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="store_true",
                        help="fail if any citation is unverified")
    args = parser.parse_args()

    import pandas as pd

    audit = pd.read_csv(AUDIT)
    bib_text = BIB.read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,\s]+)\s*,", bib_text))
    audit_keys = set(audit["citation_key"])

    problems: list[str] = []

    # Every key actually CITED must exist in both the bibliography and
    # the audit. A key present in one but not the other is how an
    # unverified citation slips into the manuscript.
    sources = "".join(
        (PAPER / name).read_text(encoding="utf-8")
        for name in ("main.tex", "_sections_intro.tex", "_sections_results.tex")
    )
    cited = {
        key.strip()
        for group in re.findall(r"\\cite[tp]?\{([^}]+)\}", sources)
        for key in group.split(",")
    }
    print(f"Keys cited in the manuscript: {len(cited)}")
    if cited - bib_keys:
        problems.append(f"cited but absent from .bib: {sorted(cited - bib_keys)}")
    if cited - audit_keys:
        problems.append(f"cited but absent from the audit: {sorted(cited - audit_keys)}")

    missing_from_audit = bib_keys - audit_keys
    missing_from_bib = audit_keys - bib_keys
    if missing_from_audit:
        problems.append(f"in .bib but not audited: {sorted(missing_from_audit)}")
    if missing_from_bib:
        problems.append(f"audited but not in .bib: {sorted(missing_from_bib)}")

    verified = audit[audit["status"] == "verified"]
    pending = audit[audit["status"] != "verified"]

    print(f"Citations: {len(audit)} total, {len(verified)} verified, "
          f"{len(pending)} pending\n")

    if len(pending):
        print("PENDING (bibliographic and/or claim verification incomplete):")
        for _, row in pending.iterrows():
            print(f"  {row['citation_key']:16s} {row['supporting_location'][:78]}")

    # Every verified entry must actually carry the evidence of verification.
    for _, row in verified.iterrows():
        if str(row.get("metadata_verified")) != "yes":
            problems.append(f"{row['citation_key']}: marked verified but "
                            "metadata_verified is not 'yes'")
        if not str(row.get("supporting_location", "")).strip():
            problems.append(f"{row['citation_key']}: no supporting location "
                            "recorded for the cited claim")
        if str(row.get("supporting_location", "")).startswith("TO CHECK"):
            problems.append(f"{row['citation_key']}: marked verified but the "
                            "claim check is still outstanding")

    # BibTeX's plainnat style lowercases every title word that is not
    # brace-protected, so acronyms and proper nouns are silently
    # downcased in the rendered bibliography: "the em algorithm",
    # "united kingdom inflation", "the sharpe ratio". BibTeX succeeds, so
    # no build gate sees it; it was found by reading page 14 of the PDF.
    print("\nTitle case protection:")
    protect = {"EM", "GARCH", "ARCH", "VIX", "ETF", "US", "UK", "CAPM", "QLIKE",
               "OLS", "GMM", "HMM", "DCC", "VAR", "CRSP", "SP", "N", "I", "II",
               "Sharpe", "Markowitz", "Ledoit", "Wolf", "Gaussian", "Bayesian",
               "Monte", "Carlo", "United", "Kingdom", "States", "American",
               "European", "Treasury", "Holm", "Newey", "West", "Hamilton",
               "Diebold", "Mariano", "Politis", "Romano", "Engle", "Bollerslev"}
    unprotected: list[str] = []
    for key, title in re.findall(r"@\w+\{([^,]+),.*?title\s*=\s*\{(.*?)\}\s*,\s*\n",
                                 bib_text, re.DOTALL):
        # Strip already-protected spans, then look at what is left.
        exposed = re.sub(r"\{[^{}]*\}", " ", title)
        for word in re.findall(r"[A-Za-z][\w'-]*", exposed):
            if word in protect or (len(word) > 1 and word.isupper()):
                unprotected.append(f"{key}: '{word}' is not brace-protected")
    if unprotected:
        for item in unprotected:
            print(f"  [UNPROTECTED] {item}")
        problems.extend(unprotected)
    else:
        print("  every acronym and proper noun in a title is brace-protected")

    # The .bib itself must not advertise pending work at release time.
    pending_markers = bib_text.count("PENDING VERIFICATION")
    print(f"\n'PENDING VERIFICATION' markers in references.bib: {pending_markers}")

    if problems:
        print("\nAUDIT PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")

    if args.release:
        blocking = list(problems)
        if len(pending):
            blocking.append(f"{len(pending)} citation(s) not fully verified")
        if pending_markers:
            blocking.append(
                f"{pending_markers} 'PENDING VERIFICATION' marker(s) remain in "
                "references.bib"
            )
        if blocking:
            print("\nRELEASE BLOCKED:")
            for item in blocking:
                print(f"  - {item}")
            print("\nEvery citation needs both bibliographic verification and "
                  "claim verification before the manuscript tag.")
            sys.exit(1)
        print("\nRELEASE CHECK PASSED: all citations verified, both metadata "
              "and cited claim.")
        return

    if problems:
        sys.exit(1)
    print("\nDraft mode: pending citations are allowed. Run with --release "
          "before tagging the manuscript.")


if __name__ == "__main__":
    main()
