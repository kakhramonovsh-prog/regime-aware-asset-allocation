"""Exhaustive census of every numeric literal in the manuscript.

Substring matching for hand-typed results is unsound: two unrelated
quantities can legitimately share the same rounded digits, so a match
proves nothing and a miss proves nothing. The volatility interval
``[-0.327, -0.033]`` was caught only because ``0.327`` happened to
collide with the p-value, and the original twelve-value inventory had
missed it.

This replaces that heuristic. Every numeric literal in the prose is
extracted and must be classified in ``paper/number_census.csv`` as
exactly one of:

* ``empirical_result``       --- must come from a generated macro
* ``methodological_assumption`` --- lambda, block length, cost level
* ``sample_or_date``         --- years, counts of observations
* ``equation_constant``      --- exponents, subscripts, fixed constants
* ``literature_fact``        --- a number attributed to a cited paper

An ``empirical_result`` carries a result ID, macro name, raw value,
displayed value, unit and source artifact, and must NOT appear as a
literal in the prose. Any unclassified literal fails the census.

    python scripts/number_census.py [--write-template]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"
CENSUS = PAPER / "number_census.csv"
INVENTORY = PAPER / "result_inventory.csv"

INVENTORY_FIELDS = ("result_id", "macro", "raw_value", "display_value",
                    "unit", "source_artifact")

SOURCES = ["main.tex", "_sections_intro.tex", "_sections_results.tex"]

# Matches integers and decimals, including LaTeX thousands separators
# such as 10{,}000. Excludes numbers glued to letters (macro names,
# label keys) and LaTeX lengths.
NUMBER = re.compile(r"(?<![\w.\\])(\d+(?:\{,\}\d+)*(?:\.\d+)?)(?![\w])")

# Spelled-out quantities are numbers too. "Eleven of thirteen point
# estimates are positive" and "only one survives at the 5% level" are
# empirical results with no digit in them, so a digit-only census would
# certify a manuscript that still hand-types its counts.
WORD_NUMBERS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand",
]
WORD_NUMBER = re.compile(
    r"(?<![\w-])(" + "|".join(WORD_NUMBERS) + r")(?![\w-])", re.IGNORECASE
)


def prose_text() -> str:
    """Manuscript prose with comments, inputs and macro definitions removed."""
    text = "".join((PAPER / name).read_text(encoding="utf-8") for name in SOURCES)
    text = re.sub(r"(?<!\\)%.*", "", text)          # LaTeX comments
    text = re.sub(r"\\input\{[^}]+\}", "", text)     # generated tables
    text = re.sub(r"\\label\{[^}]+\}", "", text)     # label keys
    text = re.sub(r"\\cite[tp]?\{[^}]+\}", "", text)  # citation keys
    text = re.sub(r"\\includegraphics(\[[^\]]*\])?\{[^}]+\}", "", text)
    text = re.sub(r"\\(usepackage|documentclass|geometry)(\[[^\]]*\])?\{[^}]+\}", "", text)
    # Typesetting lengths (\vspace{0.5em}, \\[0.3em]) are not quantities
    # the reader sees; leaving them in floods the census with bare zeros.
    text = re.sub(r"\\(vspace|hspace|vskip|hskip)\*?\{[^}]*\}", "", text)
    text = re.sub(r"\\\\\[[^\]]*\]", "", text)
    return text


# Generated macros are invoked both as ``\primarySharpeDiff{}`` and,
# inside equations, as ``\primarySharpeDiff`` with no braces. Matching
# only the braced form silently exempts every equation from the check.
MACRO_USE = re.compile(r"\\([A-Za-z]+)")

# Generated names are camelCase: a lowercase run followed by a capital.
# LaTeX's own commands (\textbf, \emph, \sqrt) are all lowercase, so
# this separates "a result macro" from "any command" without a list.
GENERATED_NAME = re.compile(r"^[a-z]+[A-Z]")

# Generated macros that are document metadata rather than reported
# results, so they carry no raw value, unit or source artifact. Keep
# this set tiny: anything that states a quantity about the data belongs
# in the result inventory, not here.
NON_RESULT_MACROS = {"\\releaseDate"}


def macros_used() -> set[str]:
    """Generated macros the prose actually invokes, as ``\\name``."""
    known = (set(pd.read_csv(INVENTORY)["macro"])
             if INVENTORY.exists() else set())
    used = {f"\\{name}" for name in MACRO_USE.findall(prose_text())}
    return used & known


def check_inventory() -> list[str]:
    """Every empirical result carries a complete, traceable record.

    The census proves no result is typed as a literal. That is only half
    the guarantee: the other half is that each result the prose *does*
    cite resolves to a generated value with a recorded provenance.
    """
    if not INVENTORY.exists():
        return [f"{INVENTORY.name} is missing; run generate_paper_outputs.py"]

    inventory = pd.read_csv(INVENTORY)
    problems = []

    missing_columns = [c for c in INVENTORY_FIELDS if c not in inventory.columns]
    if missing_columns:
        return [f"result inventory lacks columns {missing_columns}"]

    for _, row in inventory.iterrows():
        for field in INVENTORY_FIELDS:
            if pd.isna(row[field]) or not str(row[field]).strip():
                problems.append(f"result {row['result_id']!r} is missing {field}")

    duplicates = inventory["result_id"][inventory["result_id"].duplicated()]
    if len(duplicates):
        problems.append(f"duplicate result ids: {sorted(set(duplicates))}")

    known = set(inventory["macro"])
    invoked = {f"\\{name}" for name in MACRO_USE.findall(prose_text())
               if GENERATED_NAME.match(name)}
    undefined = sorted(invoked - known - NON_RESULT_MACROS)
    if undefined:
        problems.append(
            f"prose invokes {len(undefined)} result macro(s) absent from the "
            f"inventory: {undefined}"
        )
    return problems


def literals() -> list[str]:
    """Every numeric quantity in the prose, digits and words alike."""
    text = prose_text()
    words = [f"word:{match.lower()}" for match in WORD_NUMBER.findall(text)]
    return NUMBER.findall(text) + words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-template", action="store_true",
                        help="emit a template census for any unclassified literal")
    args = parser.parse_args()



    found = literals()
    distinct = sorted(set(found), key=lambda v: (len(v), v))
    counts = {value: found.count(value) for value in distinct}

    if args.write_template or not CENSUS.exists():
        rows = [{
            "literal": value, "occurrences": counts[value],
            "classification": "", "result_id": "", "macro": "",
            "raw_value": "", "display_value": "", "unit": "",
            "source_artifact": "", "note": "",
        } for value in distinct]
        template = PAPER / "number_census_template.csv"
        pd.DataFrame(rows).to_csv(template, index=False)
        print(f"Wrote template with {len(rows)} distinct literals -> "
              f"{template.relative_to(PROJECT_ROOT)}")
        if not CENSUS.exists():
            print("Classify every row, save as number_census.csv, and rerun.")
            return

    census = pd.read_csv(CENSUS).fillna("")
    classified = {str(row["literal"]): row for _, row in census.iterrows()}

    valid = {"empirical_result", "methodological_assumption", "sample_or_date",
             "equation_constant", "literature_fact"}
    problems: list[str] = []

    unclassified = [v for v in distinct if v not in classified]
    if unclassified:
        problems.append(
            f"{len(unclassified)} literal(s) absent from the census: "
            f"{unclassified[:12]}"
        )

    stale = [v for v in classified if v not in set(distinct)]

    empirical = []
    for value in distinct:
        if value not in classified:
            continue
        row = classified[value]
        kind = str(row["classification"]).strip()
        if kind not in valid:
            problems.append(f"{value!r}: classification {kind!r} is not one of "
                            f"{sorted(valid)}")
            continue
        if kind == "empirical_result":
            empirical.append(value)
            for field in ("result_id", "macro", "raw_value", "display_value",
                          "unit", "source_artifact"):
                if not str(row[field]).strip():
                    problems.append(f"{value!r}: empirical result missing {field}")

    # An empirical result must not appear as a literal at all: it should
    # have been emitted by its macro.
    if empirical:
        problems.append(
            f"{len(empirical)} empirical result(s) appear as literals in the "
            f"prose instead of via macros: {empirical}"
        )

    by_kind: dict[str, int] = {}
    for value in distinct:
        if value in classified:
            kind = str(classified[value]["classification"]).strip()
            by_kind[kind] = by_kind.get(kind, 0) + counts[value]

    print(f"Numeric literals in prose: {len(found)} occurrences, "
          f"{len(distinct)} distinct\n")
    for kind in sorted(by_kind):
        print(f"  {kind:28s} {by_kind[kind]:4d} occurrence(s)")
    if stale:
        print(f"\n  note: {len(stale)} census row(s) no longer appear in the "
              f"prose: {stale[:8]}")

    problems.extend(check_inventory())

    if INVENTORY.exists():
        inventory = pd.read_csv(INVENTORY)
        cited = macros_used()
        print(f"\nEmpirical results in the inventory: {len(inventory)}")
        print(f"  cited in the prose:      {len(cited)}")
        print(f"  cited only in tables:    {len(inventory) - len(cited)}")
        print(f"  distinct source files:   "
              f"{inventory['source_artifact'].nunique()}")

    if problems:
        print("\nCENSUS PROBLEMS:")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("\nCENSUS PASSED: every numeric literal is classified, and no "
          "empirical result is typed into the prose.")


if __name__ == "__main__":
    main()
